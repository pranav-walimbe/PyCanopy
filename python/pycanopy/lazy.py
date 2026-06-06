"""SpatialLazyFrame — immutable plan builder. No execution until .collect()."""

from __future__ import annotations

import polars as pl

from pycanopy.executor import _ROW_IDX, SpatialExecutor
from pycanopy.nodes import (
    ContainsNode,
    FusedSpatialNode,
    KnnJoinNode,
    KnnNode,
    Plan,
    PluginPath,
    RangeNode,
    ScalarNode,
    WithinDistanceJoinNode,
    WithinJoinNode,
)
from pycanopy.optimizer import SpatialOptimizer


def _fmt_expr(expr: pl.Expr) -> str:
    s = str(expr)
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return s.strip()


def _fmt_node(node) -> str:
    if isinstance(node, ScalarNode):
        return f"FILTER [{_fmt_expr(node.expr)}]"
    if isinstance(node, RangeNode):
        return (
            f"RANGE_QUERY [({node.min_x:.4g}, {node.min_y:.4g})"
            f" → ({node.max_x:.4g}, {node.max_y:.4g})]"
        )
    if isinstance(node, ContainsNode):
        return f"CONTAINS [({node.qx:.4g}, {node.qy:.4g})]"
    if isinstance(node, KnnNode):
        approx = ", approximate" if node.approximate else ""
        return f"KNN [k={node.k}, ({node.qx:.4g}, {node.qy:.4g}){approx}]"
    if isinstance(node, FusedSpatialNode):
        count = len(node.predicates)
        pred_strs = []
        for pred in node.predicates:
            if isinstance(pred, RangeNode):
                pred_strs.append(
                    f"  ({pred.min_x:.4g}, {pred.min_y:.4g}) → ({pred.max_x:.4g}, {pred.max_y:.4g})"
                )
            elif isinstance(pred, ContainsNode):
                pred_strs.append(f"  ({pred.qx:.4g}, {pred.qy:.4g})")
        return "\n".join([f"FUSED_SPATIAL [x{count}]", *pred_strs])
    if isinstance(node, KnnJoinNode):
        approx = ", approximate" if node.approximate else ""
        return f"KNN_JOIN [k={node.k}, query_rows={len(node.query_df):,}, barrier{approx}]"
    if isinstance(node, WithinJoinNode):
        flip = ", flip" if node.flip else ""
        return f"WITHIN_JOIN [query_rows={len(node.query_df):,}, barrier{flip}]"
    if isinstance(node, WithinDistanceJoinNode):
        flip = ", flip" if node.flip else ""
        return f"WITHIN_DIST_JOIN [dist={node.distance:.4g}, query_rows={len(node.query_df):,}, barrier{flip}]"
    return f"UNKNOWN [{type(node).__name__}]"


def _fmt_plan(plan: Plan, path: PluginPath | None, n: int, *, optimized: bool) -> str:
    path_suffix = ""
    if path is not None:
        path_label = "EXPR" if path == PluginPath.EXPR else "IO"
        path_suffix = f"; path: {path_label}"
    df_line = f"DF [N={n:,}{path_suffix}]"

    if not plan:
        return df_line

    # Polars convention: outermost (last executed) op at top, source at bottom.
    # Each op is followed by FROM at the same indent level, then its source indented
    # one level deeper — matching how Polars formats single-path plans.
    reversed_plan = list(reversed(plan))
    lines = []
    for depth, node in enumerate(reversed_plan):
        indent = "  " * depth
        node_str = _fmt_node(node)
        first, *rest = node_str.split("\n")
        lines.append(f"{indent}{first}")
        for r in rest:
            lines.append(f"{indent}{r}")
        lines.append(f"{indent}FROM")
    lines.append(f"{'  ' * len(reversed_plan)}{df_line}")
    return "\n".join(lines)


class SpatialLazyFrame:
    """Builds a spatial query plan declaratively. Declaration order is not execution order.

    All methods return a new SpatialLazyFrame with the node appended; no mutation.
    The optimizer reorders filter nodes by cost when .collect() is called.
    Join nodes (knn_join, within_join) act as barriers and are never reordered.

    Args:
        sf: The SpatialFrame that owns the Engine and DataFrame.
        plan: Current list of plan nodes (do not mutate directly).
    """

    def __init__(self, sf: SpatialFrame, plan: Plan) -> None:  # noqa: F821
        self._sf = sf
        self._plan = plan

    def filter(self, expr: pl.Expr) -> SpatialLazyFrame:
        """Add a scalar Polars expression filter.

        The expression is stored as a plan node and evaluated by Polars (not the
        spatial engine). The optimizer may reorder it relative to spatial nodes
        based on selectivity estimates.

        Args:
            expr: Any Polars expression that evaluates to a boolean column.

        Returns:
            New SpatialLazyFrame with the scalar node appended.
        """
        return SpatialLazyFrame(self._sf, [*self._plan, ScalarNode(expr)])

    def range_query(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> SpatialLazyFrame:
        """Add a bounding-box spatial filter.

        Args:
            min_x: Left edge of the query rectangle.
            min_y: Bottom edge of the query rectangle.
            max_x: Right edge of the query rectangle.
            max_y: Top edge of the query rectangle.

        Returns:
            New SpatialLazyFrame with the range node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, RangeNode(min_x, min_y, max_x, max_y)],
        )

    def contains(self, x: float, y: float) -> SpatialLazyFrame:
        """Add a point-in-polygon filter (polygon dataset only).

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.

        Returns:
            New SpatialLazyFrame with the contains node appended.
        """
        return SpatialLazyFrame(self._sf, [*self._plan, ContainsNode(x, y)])

    def knn(
        self,
        x: float,
        y: float,
        k: int,
        approximate: bool = False,
    ) -> SpatialLazyFrame:
        """Add a k-nearest-neighbour lookup.

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            k: Number of neighbours to return.
            approximate: Skip exact geometric refinement for speed.

        Returns:
            New SpatialLazyFrame with the knn node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, KnnNode(x, y, k, approximate)],
        )

    def knn_join(
        self,
        query_df: pl.DataFrame,
        x_col: str,
        y_col: str,
        k: int,
        approximate: bool = False,
    ) -> SpatialLazyFrame:
        """Spatial join: for each row in query_df find its k nearest neighbours
        in this Engine's dataset.

        Acts as a barrier — no plan nodes are reordered past a join.
        Result has query_df columns followed by Engine df columns (right-side
        columns that conflict are prefixed with 'right_').

        Args:
            query_df: DataFrame of query points.
            x_col: Column in query_df holding x coordinates.
            y_col: Column in query_df holding y coordinates.
            k: Number of neighbours per query row.
            approximate: Skip exact geometric refinement for speed.

        Returns:
            New SpatialLazyFrame with the knn join node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, KnnJoinNode(query_df, x_col, y_col, k, approximate)],
        )

    def within_distance_join(
        self,
        query_df: pl.DataFrame,
        x_col: str,
        y_col: str,
        distance: float,
    ) -> SpatialLazyFrame:
        """Spatial join: for each point in query_df find Engine points within `distance`.

        Acts as a barrier — no plan nodes are reordered past a join.
        Result has query_df columns followed by Engine df columns (right-side
        columns that conflict are prefixed with 'right_').

        Args:
            query_df: DataFrame of query points.
            x_col: Column in query_df holding x coordinates.
            y_col: Column in query_df holding y coordinates.
            distance: Maximum Euclidean distance for a match.

        Returns:
            New SpatialLazyFrame with the within-distance join node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, WithinDistanceJoinNode(query_df, x_col, y_col, distance)],
        )

    def within_join(
        self,
        query_df: pl.DataFrame,
        x_col: str,
        y_col: str,
    ) -> SpatialLazyFrame:
        """Spatial join: for each point in query_df find the Engine polygons
        that contain it. Engine must be a polygon dataset.

        Acts as a barrier — no plan nodes are reordered past a join.
        Result has query_df columns followed by Engine df columns (right-side
        columns that conflict are prefixed with 'right_').

        Args:
            query_df: DataFrame of query points.
            x_col: Column in query_df holding x coordinates.
            y_col: Column in query_df holding y coordinates.

        Returns:
            New SpatialLazyFrame with the within join node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, WithinJoinNode(query_df, x_col, y_col)],
        )

    def explain(self, optimized: bool = True) -> str:
        """Return a human-readable description of the query plan.

        Mirrors the interface of Polars' LazyFrame.explain(). By default shows
        the plan after the optimizer has run: reordered operations, any fused
        spatial predicates, and the chosen execution path (EXPR or IO). Pass
        optimized=False to see the plan in declaration order instead.

        Args:
            optimized: If True, show the optimizer's execution order with path
                selection. If False, show operations in declaration order.

        Returns:
            Multi-line string describing the plan. Print it for readable output.
        """
        engine = self._sf.engine
        if optimized:
            opt = SpatialOptimizer()
            plan = opt.optimize(self._plan, engine)
            path = opt._select_plugin_path(plan, engine)
        else:
            plan = self._plan
            path = None
        return _fmt_plan(plan, path, engine.n, optimized=optimized)

    def collect(self) -> pl.DataFrame:
        """Optimise and execute the plan. Returns a Polars DataFrame.

        Triggers:
          1. SpatialOptimizer: selectivity estimation, cost-based sort, fusion pass.
          2. SpatialOptimizer: plugin path selection (EXPR vs IO).
          3. SpatialExecutor: emits the optimised plan via the chosen plugin path.
        """
        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        optimized = optimizer.optimize(self._plan, self._sf.engine)
        plugin_path = optimizer._select_plugin_path(optimized, self._sf.engine)
        return executor.execute(optimized, self._sf, plugin_path)

    @staticmethod
    def collect_all(frames: list[SpatialLazyFrame]) -> list[pl.DataFrame]:
        """Collect multiple SpatialLazyFrames, caching any shared plan prefix.

        When frames were branched from the same base SpatialLazyFrame they share
        plan nodes as identical Python objects (SpatialLazyFrame builds plans via
        [*self._plan, new_node], which reuses references rather than copying). This
        method detects that shared prefix, emits it once as a cached Polars
        LazyFrame, builds each branch's suffix from the cache, then collects all
        branches in a single pl.collect_all() call.

        Falls back to independent collect() calls when no common prefix is found.

        All frames must belong to the same SpatialFrame.

        Args:
            frames: SpatialLazyFrames to collect. Must share a SpatialFrame.

        Returns:
            List of DataFrames in the same order as frames.

        Raises:
            ValueError: If frames is empty or frames belong to different SpatialFrames.
        """
        if not frames:
            raise ValueError("collect_all requires at least one frame")
        if len(frames) == 1:
            return [frames[0].collect()]

        sf = frames[0]._sf
        if not all(f._sf is sf for f in frames[1:]):
            raise ValueError("All frames in collect_all must belong to the same SpatialFrame")

        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        plans = [f._plan for f in frames]
        prefix_len = optimizer._detect_fanout(plans)

        if prefix_len == 0:
            return [f.collect() for f in frames]

        # Optimise the shared prefix as a standalone plan and cache its Polars chain.
        prefix_plan = plans[0][:prefix_len]
        optimized_prefix = optimizer.optimize(prefix_plan, sf.engine)
        base_lf = sf.df.with_row_index(_ROW_IDX).lazy()
        cached_lf = executor._emit_chain(optimized_prefix, sf, base_lf, PluginPath.EXPR).cache()

        # Build each branch's suffix chain starting from the cached result.
        branch_lfs: list[pl.LazyFrame] = []
        for frame in frames:
            suffix_plan = frame._plan[prefix_len:]
            if not suffix_plan:
                branch_lfs.append(cached_lf)
                continue
            optimized_suffix = optimizer.optimize(suffix_plan, sf.engine)
            branch_lfs.append(
                executor._emit_chain(optimized_suffix, sf, cached_lf, PluginPath.EXPR)
            )

        collected = pl.collect_all(branch_lfs)
        return [df.drop(_ROW_IDX) if _ROW_IDX in df.columns else df for df in collected]
