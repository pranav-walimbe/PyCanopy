"""
SpatialOptimizer: cost-based plan transformer.

Passes (in order):
  1. _assign_selectivity: estimate selectivity for each node from engine stats
  2. _cost_sort: reorder scalar vs. spatial based on selectivity
  3. _fusion_pass: merge consecutive fusable spatial nodes
  4. _join_side_pass: set flip=True on symmetric joins where query side is larger
  5. _detect_fanout: find the longest shared plan prefix across branches
"""

from __future__ import annotations

import dataclasses
import json
import math

from pycanopy.nodes import (
    ContainsNode,
    FusedSpatialNode,
    IntersectsSelfJoinNode,
    KnnJoinNode,
    KnnNode,
    Plan,
    PluginPath,
    PolygonKnnJoinNode,
    PolygonWithinDistanceJoinNode,
    RangeNode,
    ScalarNode,
    SelectNode,
    WithinDistanceJoinNode,
    WithinDistanceOfPointNode,
    WithinJoinNode,
)

# Spatial nodes with selectivity below this threshold are too selective to fuse
_FUSION_SELECTIVITY_FLOOR = 0.05

# Datasets smaller than this always use BruteForce
_FUSION_MIN_N = 500

# Spatial selectivity threshold where slicing sf.df directly (IO path) is cheaper
_IO_SELECTIVITY_THRESHOLD = 0.05


_BINARY_OP_COST: dict[str, int] = {
    "Gt": 1,
    "Lt": 1,
    "Eq": 1,
    "GtEq": 1,
    "LtEq": 1,
    "NotEq": 1,
    "Plus": 1,
    "Minus": 1,
    "Multiply": 2,
    "Divide": 2,
    "Modulus": 2,
    "And": 1,
    "Or": 1,
}

_FUNCTION_KEY_COST: dict[str, int] = {
    "Pow": 4,
    "StringExpr": 10,
    "ListExpr": 8,
}

_BOOLEAN_FUNCTION_COST: dict[str, int] = {
    "IsBetween": 2,
    "IsIn": 5,
    "IsNull": 1,
    "IsNotNull": 1,
}


def _function_cost(fn: object) -> int:
    # Cost of a Polars function expression from its serialized AST node
    if not isinstance(fn, dict):
        return 1
    for key, val in fn.items():
        if key in _FUNCTION_KEY_COST:
            return _FUNCTION_KEY_COST[key]
        if key == "Boolean":
            if isinstance(val, str):
                return _BOOLEAN_FUNCTION_COST.get(val, 1)
            if isinstance(val, dict):
                return _BOOLEAN_FUNCTION_COST.get(next(iter(val), ""), 1)
    return 1


def _walk_ast_cost(node: object) -> int:
    # Recursively sum the estimated cost of a serialized Polars expression AST
    if isinstance(node, dict):
        if "BinaryExpr" in node:
            expr = node["BinaryExpr"]
            return (
                _BINARY_OP_COST.get(expr.get("op", ""), 0)
                + _walk_ast_cost(expr.get("left"))
                + _walk_ast_cost(expr.get("right"))
            )
        if "Function" in node:
            fn = node["Function"]
            return _function_cost(fn.get("function", {})) + sum(
                _walk_ast_cost(inp) for inp in fn.get("input", [])
            )
        if "Cast" in node:
            return 2 + _walk_ast_cost(node["Cast"].get("expr"))
        return sum(_walk_ast_cost(v) for v in node.values())
    if isinstance(node, list):
        return sum(_walk_ast_cost(item) for item in node)
    return 0


def _scalar_cost(expr) -> int:
    # Estimate a scalar expression's cost from its serialized AST, defaulting to 1
    try:
        tree = json.loads(expr.meta.serialize(format="json"))
        return max(1, _walk_ast_cost(tree))
    except Exception:
        return 1


class SpatialOptimizer:
    """Transforms a raw SpatialLazyFrame plan into an execution-ordered plan."""

    def optimize(self, plan: Plan, engine) -> Plan:
        """Run all optimisation passes and return the execution-ordered plan.

        Args:
            plan: Raw plan in declaration order.
            engine: Engine instance, used to read dataset statistics.

        Returns:
            Optimised plan ready for the executor.
        """
        if not plan:
            return plan
        # A trailing SelectNode is a terminal projection, not a predicate, so the cost passes skip it
        select_tail = None
        if isinstance(plan[-1], SelectNode):
            select_tail = plan[-1]
            plan = plan[:-1]
        if plan:
            plan = self._assign_selectivity(plan, engine)
            plan = self._cost_sort(plan)
            plan = self._fusion_pass(plan, engine)
            plan = self._join_side_pass(plan, engine)
        if select_tail is not None:
            plan = [*plan, select_tail]
        return plan

    def _assign_selectivity(self, plan: Plan, engine) -> Plan:
        # Estimate and attach selectivity to each node
        n = engine.n
        extent = engine.extent
        result = []
        for node in plan:
            if isinstance(node, RangeNode):
                sel = self._range_selectivity(node, extent)
                node = dataclasses.replace(node, selectivity=sel)
            elif isinstance(node, ContainsNode):
                node = dataclasses.replace(node, selectivity=1.0 / max(n, 1))
            elif isinstance(node, KnnNode):
                node = dataclasses.replace(node, selectivity=min(1.0, node.k / max(n, 1)))
            elif isinstance(node, WithinDistanceOfPointNode):
                node = dataclasses.replace(node, selectivity=self._disk_selectivity(node, extent))
            elif isinstance(node, ScalarNode):
                node = dataclasses.replace(node, cost=_scalar_cost(node.expr))
            # join nodes have no selectivity field
            result.append(node)
        return result

    def _range_selectivity(
        self,
        node: RangeNode,
        extent: tuple[float, float, float, float] | None,
    ) -> float:
        # Selectivity of a range query as its area divided by the dataset extent area
        if extent is None:
            return 1.0
        min_x, min_y, max_x, max_y = extent
        total_area = (max_x - min_x) * (max_y - min_y)
        if total_area <= 0.0:
            return 1.0
        query_area = max(0.0, node.max_x - node.min_x) * max(0.0, node.max_y - node.min_y)
        return min(1.0, query_area / total_area)

    def _disk_selectivity(
        self,
        node: WithinDistanceOfPointNode,
        extent: tuple[float, float, float, float] | None,
    ) -> float:
        # Selectivity of a radius query as its disk area divided by the dataset extent area
        if extent is None:
            return 1.0
        min_x, min_y, max_x, max_y = extent
        total_area = (max_x - min_x) * (max_y - min_y)
        if total_area <= 0.0:
            return 1.0
        return min(1.0, math.pi * node.distance * node.distance / total_area)

    def _cost_sort(self, plan: Plan) -> Plan:
        # Reorder nodes so cheaper operations run first. Joins and KNN are barriers, and within
        # a run scalars go first (Polars handles them cheaply) then spatials by ascending selectivity.
        result: Plan = []
        run: Plan = []
        _barrier_types = (
            KnnNode,
            KnnJoinNode,
            WithinJoinNode,
            WithinDistanceJoinNode,
            PolygonWithinDistanceJoinNode,
            PolygonKnnJoinNode,
            IntersectsSelfJoinNode,
        )
        for node in plan:
            if isinstance(node, _barrier_types):
                result.extend(self._sort_run(run))
                result.append(node)
                run = []
            else:
                run.append(node)
        result.extend(self._sort_run(run))
        return result

    def _sort_run(self, run: Plan) -> Plan:
        # Sort a barrier-separated run: scalars first by cost, then spatials by selectivity
        scalars = [n for n in run if isinstance(n, ScalarNode)]
        spatials = [n for n in run if not isinstance(n, ScalarNode)]
        return sorted(scalars, key=lambda n: n.cost) + sorted(spatials, key=lambda n: n.selectivity)

    def _fusion_pass(self, plan: Plan, engine) -> Plan:
        # Merge consecutive fusable spatial filter nodes into a FusedSpatialNode, splitting the
        # run at any node that is not fusable (too selective, or dataset below _FUSION_MIN_N).
        if engine.n < _FUSION_MIN_N:
            return plan

        result: Plan = []
        i = 0
        while i < len(plan):
            node = plan[i]
            if not self._is_fusable(node):
                result.append(node)
                i += 1
                continue

            run = [node]
            i += 1
            while i < len(plan) and self._is_fusable(plan[i]):
                run.append(plan[i])
                i += 1

            if len(run) == 1:
                result.append(run[0])
            else:
                result.append(FusedSpatialNode(predicates=run))

        return result

    def _is_fusable(self, node) -> bool:
        # A range or contains node is fusable when its selectivity clears the fusion floor
        return (
            isinstance(node, (RangeNode, ContainsNode))
            and node.selectivity >= _FUSION_SELECTIVITY_FLOOR
        )

    def _join_side_pass(self, plan: Plan, engine) -> Plan:
        # Set flip=True on symmetric join nodes when the query side is over half the dataset,
        # so the existing Engine index is not abandoned for a marginal size difference.
        result = []
        for node in plan:
            if isinstance(node, (WithinJoinNode, WithinDistanceJoinNode)):
                if len(node.query_df) > engine.n // 2:
                    node = dataclasses.replace(node, flip=True)
            result.append(node)
        return result

    def _detect_fanout(self, plans: list[Plan]) -> int:
        # Return the length of the longest plan prefix shared by all plans, where a node is
        # shared when it is the same Python object across every plan (plans reuse references).
        if len(plans) < 2:
            return 0
        min_len = min(len(p) for p in plans)
        for i in range(min_len):
            if not all(p[i] is plans[0][i] for p in plans[1:]):
                return i
        return min_len

    def _select_plugin_path(self, plan: Plan, engine) -> PluginPath:
        # Choose the IO path when a spatial filter is selective enough (K << N) to beat
        # rebuilding a local index, otherwise EXPR. Joins and KNN always use EXPR.
        if any(
            isinstance(
                n,
                (
                    KnnNode,
                    KnnJoinNode,
                    WithinJoinNode,
                    WithinDistanceJoinNode,
                    PolygonWithinDistanceJoinNode,
                    PolygonKnnJoinNode,
                ),
            )
            for n in plan
        ):
            return PluginPath.EXPR

        for node in plan:
            sel: float | None = None
            if isinstance(node, (RangeNode, ContainsNode)):
                sel = node.selectivity
            elif isinstance(node, FusedSpatialNode):
                sel = min(p.selectivity for p in node.predicates)
            if sel is not None and sel < _IO_SELECTIVITY_THRESHOLD:
                return PluginPath.IO

        return PluginPath.EXPR
