"""SpatialExecutor — walks the optimised plan and emits a Polars LazyFrame chain."""

from __future__ import annotations

import numpy as np
import polars as pl

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

# Internal column name used to track original row positions through scalar filters.
# Engine results are always indices into the original dataset, so we need to
# correlate them with post-filter rows using this persistent index.
_ROW_IDX = "__orig_row__"


def _range_plugin_expr(
    x_col: str,
    y_col: str,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> pl.Expr:
    """Boolean mask expression for a bounding-box filter via map_batches.

    is_elementwise=False makes Polars treat this as a column-level barrier: any
    elementwise scalar predicates emitted before it in the LazyFrame chain execute
    first, so the local index is built on the M already-filtered rows rather than N.
    """

    def _apply(s: pl.Series) -> pl.Series:
        from pycanopy.engine import Engine

        df = s.struct.unnest()
        xs = df[x_col].to_numpy()
        ys = df[y_col].to_numpy()
        local_eng = Engine.from_coords(xs, ys)
        hits = local_eng.range_query(min_x, min_y, max_x, max_y)
        mask = np.zeros(len(xs), dtype=bool)
        if hits:
            mask[hits] = True
        return pl.Series("", mask, dtype=pl.Boolean)

    return pl.struct(pl.col(x_col), pl.col(y_col)).map_batches(
        _apply, return_dtype=pl.Boolean, is_elementwise=False
    )


def _contains_plugin_expr(x_col: str, y_col: str, qx: float, qy: float) -> pl.Expr:
    """Boolean mask expression for a point exact-match filter via map_batches."""

    def _apply(s: pl.Series) -> pl.Series:
        from pycanopy.engine import Engine

        df = s.struct.unnest()
        xs = df[x_col].to_numpy()
        ys = df[y_col].to_numpy()
        local_eng = Engine.from_coords(xs, ys)
        hits = local_eng.contains(qx, qy)
        mask = np.zeros(len(xs), dtype=bool)
        if hits:
            mask[hits] = True
        return pl.Series("", mask, dtype=pl.Boolean)

    return pl.struct(pl.col(x_col), pl.col(y_col)).map_batches(
        _apply, return_dtype=pl.Boolean, is_elementwise=False
    )


def _fused_plugin_expr(
    x_col: str, y_col: str, predicates: list[RangeNode | ContainsNode]
) -> pl.Expr:
    """Boolean mask expression that applies all fused predicates with one index build.

    Building the local index once for all predicates is the core benefit of fusion:
    each additional predicate costs only a query, not an index rebuild.
    """

    def _apply(s: pl.Series) -> pl.Series:
        from pycanopy.engine import Engine

        df = s.struct.unnest()
        xs = df[x_col].to_numpy()
        ys = df[y_col].to_numpy()
        local_eng = Engine.from_coords(xs, ys)

        mask = np.ones(len(xs), dtype=bool)
        for pred in predicates:
            if isinstance(pred, RangeNode):
                hits = local_eng.range_query(pred.min_x, pred.min_y, pred.max_x, pred.max_y)
            elif isinstance(pred, ContainsNode):
                hits = local_eng.contains(pred.qx, pred.qy)
            else:
                continue
            pred_mask = np.zeros(len(xs), dtype=bool)
            if hits:
                pred_mask[hits] = True
            mask &= pred_mask

        return pl.Series("", mask, dtype=pl.Boolean)

    return pl.struct(pl.col(x_col), pl.col(y_col)).map_batches(
        _apply, return_dtype=pl.Boolean, is_elementwise=False
    )


class SpatialExecutor:
    """Translates the optimised plan into Polars operations and executes them.

    Two execution paths are available, chosen by the optimizer via PluginPath:

    EXPR path (default): spatial nodes emit map_batches(is_elementwise=False)
        expressions. Polars runs scalar filters first (barrier semantics), then the
        spatial closure builds a fresh local index on the M remaining rows. A
        persistent _ROW_IDX column is added for KNN nodes that still need global-index
        correlation after any scalar pre-filtering.

    IO path: the pre-built Engine (on N rows) is queried directly to get K candidate
        indices. sf.df is sliced to those K rows. Scalar filters run on the K-row
        slice. No _ROW_IDX column needed. Best when spatial selectivity is tight
        (K << N) and re-building a fresh index on M rows would be wasteful.
    """

    def execute(self, plan: Plan, sf, plugin_path: PluginPath = PluginPath.EXPR) -> pl.DataFrame:
        """Execute the optimised plan against sf.

        Args:
            plan: Execution-ordered plan from SpatialOptimizer.
            sf: SpatialFrame owning the Engine and DataFrame.
            plugin_path: Whether to use the expression plugin or IO plugin path.

        Returns:
            Filtered or joined Polars DataFrame.
        """
        # EXPR path requires x_col/y_col as real DataFrame columns (point datasets).
        # Polygon SpatialFrames use synthetic coordinate column names that don't
        # exist in df; degrade gracefully to the IO path in that case.
        # Exception: join nodes are handled directly by the executor and do not
        # use the plugin expression machinery, so they stay on EXPR path.
        has_joins = any(
            isinstance(n, (KnnJoinNode, WithinJoinNode, WithinDistanceJoinNode)) for n in plan
        )
        if plugin_path == PluginPath.EXPR and sf.x_col not in sf.df.columns and not has_joins:
            plugin_path = PluginPath.IO

        if plugin_path == PluginPath.IO:
            return self._execute_io(plan, sf)

        lf = sf.df.with_row_index(_ROW_IDX).lazy()
        lf = self._emit_chain(plan, sf, lf, plugin_path)
        df = lf.collect()
        if _ROW_IDX in df.columns:
            df = df.drop(_ROW_IDX)
        return df

    def _execute_io(self, plan: Plan, sf) -> pl.DataFrame:
        """IO path: query the pre-built Engine first, slice df, then apply scalars.

        All spatial nodes are resolved against the global Engine (no index rebuild).
        Results are AND-intersected. Scalar nodes run on the small candidate slice.
        """
        candidate_indices: set[int] | None = None
        scalar_nodes: list[ScalarNode] = []

        for node in plan:
            hits: list[int] | None = None
            if isinstance(node, RangeNode):
                hits = sf.engine.range_query(node.min_x, node.min_y, node.max_x, node.max_y)
            elif isinstance(node, ContainsNode):
                hits = sf.engine.contains(node.qx, node.qy)
            elif isinstance(node, FusedSpatialNode):
                for pred in node.predicates:
                    if isinstance(pred, RangeNode):
                        pred_hits = sf.engine.range_query(
                            pred.min_x, pred.min_y, pred.max_x, pred.max_y
                        )
                    elif isinstance(pred, ContainsNode):
                        pred_hits = sf.engine.contains(pred.qx, pred.qy)
                    else:
                        continue
                    pred_set = set(pred_hits)
                    candidate_indices = (
                        pred_set if candidate_indices is None else candidate_indices & pred_set
                    )
                continue
            elif isinstance(node, ScalarNode):
                scalar_nodes.append(node)
                continue

            if hits is not None:
                hits_set = set(hits)
                candidate_indices = (
                    hits_set if candidate_indices is None else candidate_indices & hits_set
                )

        if candidate_indices is None:
            lf = sf.df.lazy()
        else:
            lf = sf.df[sorted(candidate_indices)].lazy()

        for node in scalar_nodes:
            lf = lf.filter(node.expr)

        return lf.collect()

    def _emit_chain(
        self, plan: Plan, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        for node in plan:
            lf = self._emit_node(node, sf, lf, plugin_path)
        return lf

    def _emit_node(self, node, sf, lf: pl.LazyFrame, plugin_path: PluginPath) -> pl.LazyFrame:
        if isinstance(node, ScalarNode):
            return lf.filter(node.expr)
        if isinstance(node, RangeNode):
            return self._emit_range(node, sf, lf, plugin_path)
        if isinstance(node, ContainsNode):
            return self._emit_contains(node, sf, lf, plugin_path)
        if isinstance(node, KnnNode):
            return self._emit_knn(node, sf, lf)
        if isinstance(node, FusedSpatialNode):
            return self._emit_fused(node, sf, lf, plugin_path)
        if isinstance(node, KnnJoinNode):
            return self._emit_knn_join(node, sf, lf)
        if isinstance(node, WithinJoinNode):
            return self._emit_within_join(node, sf, lf)
        if isinstance(node, WithinDistanceJoinNode):
            return self._emit_within_distance_join(node, sf, lf)
        raise TypeError(f"Unknown plan node type: {type(node)}")

    # --- filter nodes ---

    def _emit_range(
        self, node: RangeNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        if plugin_path == PluginPath.EXPR:
            return lf.filter(
                _range_plugin_expr(
                    sf.x_col, sf.y_col, node.min_x, node.min_y, node.max_x, node.max_y
                )
            )
        indices = sf.engine.range_query(node.min_x, node.min_y, node.max_x, node.max_y)
        return self._filter_by_indices(lf, indices)

    def _emit_contains(
        self, node: ContainsNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        if plugin_path == PluginPath.EXPR:
            return lf.filter(_contains_plugin_expr(sf.x_col, sf.y_col, node.qx, node.qy))
        indices = sf.engine.contains(node.qx, node.qy)
        return self._filter_by_indices(lf, indices)

    def _emit_knn(self, node: KnnNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        indices = sf.engine.knn(node.qx, node.qy, node.k, node.approximate)
        return self._filter_by_indices(lf, indices)

    def _emit_fused(
        self, node: FusedSpatialNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        if plugin_path == PluginPath.EXPR:
            return lf.filter(_fused_plugin_expr(sf.x_col, sf.y_col, node.predicates))
        for pred in node.predicates:
            lf = self._emit_node(pred, sf, lf, plugin_path)
        return lf

    def _filter_by_indices(self, lf: pl.LazyFrame, indices: list[int]) -> pl.LazyFrame:
        if not indices:
            return lf.filter(pl.lit(False))
        return lf.filter(pl.col(_ROW_IDX).is_in(indices))

    # --- join nodes ---

    def _emit_knn_join(self, node: KnnJoinNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        """For each row in query_df find k nearest in Engine's dataset.

        The current lf (representing the filtered target) is replaced by the
        join result. __orig_row__ is not present in the output.
        """
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()
        n_queries = len(node.query_df)

        # batch_knn_join returns flat (n_queries * k,) array
        match_indices = sf.engine.batch_knn_join(query_xs, query_ys, node.k, node.approximate)

        # Expand query rows: each query row repeats k times
        query_row_indices = np.repeat(np.arange(n_queries), node.k).tolist()
        target_row_indices = match_indices.tolist()

        query_part = node.query_df[query_row_indices]
        target_part = sf.df[target_row_indices]

        target_part = _resolve_column_conflicts(query_part, target_part)
        return pl.concat([query_part, target_part], how="horizontal").lazy()

    def _emit_within_join(self, node: WithinJoinNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        """For each point in query_df find the Engine polygons that contain it.

        Engine must be a polygon dataset. Returns one row per (query, polygon) match.
        The current lf is replaced by the join result.
        """
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()

        # batch_contains returns flat (M * 2,) array: [q0, e0, q1, e1, ...]
        pairs_flat = sf.engine.batch_contains(query_xs, query_ys)

        if len(pairs_flat) == 0:
            # No matches — return empty DataFrame with correct schema
            empty_q = node.query_df.clear()
            empty_t = _resolve_column_conflicts(empty_q, sf.df.clear())
            return pl.concat([empty_q, empty_t], how="horizontal").lazy()

        pairs = pairs_flat.reshape(-1, 2)
        query_row_indices = pairs[:, 0].tolist()
        target_row_indices = pairs[:, 1].tolist()

        query_part = node.query_df[query_row_indices]
        target_part = sf.df[target_row_indices]

        target_part = _resolve_column_conflicts(query_part, target_part)
        return pl.concat([query_part, target_part], how="horizontal").lazy()

    def _emit_within_distance_join(
        self, node: WithinDistanceJoinNode, sf, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """For each query point find Engine points within node.distance.

        When node.flip is True the query side is indexed and engine points are
        iterated — cheaper when len(query_df) << engine.n.
        """
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()

        pairs_flat = sf.engine.batch_within_distance(query_xs, query_ys, node.distance, node.flip)

        if len(pairs_flat) == 0:
            empty_q = node.query_df.clear()
            empty_t = _resolve_column_conflicts(empty_q, sf.df.clear())
            return pl.concat([empty_q, empty_t], how="horizontal").lazy()

        pairs = pairs_flat.reshape(-1, 2)
        query_row_indices = pairs[:, 0].tolist()
        target_row_indices = pairs[:, 1].tolist()

        query_part = node.query_df[query_row_indices]
        target_part = sf.df[target_row_indices]

        target_part = _resolve_column_conflicts(query_part, target_part)
        return pl.concat([query_part, target_part], how="horizontal").lazy()


def _resolve_column_conflicts(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """Prefix any right-side columns that also appear in left with 'right_'."""
    overlap = set(left.columns) & set(right.columns)
    if overlap:
        return right.rename({c: f"right_{c}" for c in overlap})
    return right
