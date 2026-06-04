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
    RangeNode,
    ScalarNode,
    WithinJoinNode,
)

# Internal column name used to track original row positions through scalar filters.
# Engine results are always indices into the original dataset, so we need to
# correlate them with post-filter rows using this persistent index.
_ROW_IDX = "__orig_row__"


class SpatialExecutor:
    """Translates the optimised plan into Polars operations and executes them.

    Scalar nodes become .filter() calls delegated directly to Polars.
    Spatial filter nodes query the Engine and filter the LazyFrame by original row index.
    Join nodes produce a new DataFrame and replace the current LazyFrame.

    A persistent _ROW_IDX column is added at the start and dropped at the end
    for filter operations. Join nodes replace the LazyFrame entirely so the
    column may or may not be present in the final result.
    """

    def execute(self, plan: Plan, sf) -> pl.DataFrame:
        """Execute the optimised plan against sf.

        Args:
            plan: Execution-ordered plan from SpatialOptimizer.
            sf: SpatialFrame owning the Engine and DataFrame.

        Returns:
            Filtered or joined Polars DataFrame.
        """
        lf = sf.df.with_row_index(_ROW_IDX).lazy()
        lf = self._emit_chain(plan, sf, lf)
        df = lf.collect()
        if _ROW_IDX in df.columns:
            df = df.drop(_ROW_IDX)
        return df

    def _emit_chain(self, plan: Plan, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        for node in plan:
            lf = self._emit_node(node, sf, lf)
        return lf

    def _emit_node(self, node, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        if isinstance(node, ScalarNode):
            return lf.filter(node.expr)
        if isinstance(node, RangeNode):
            return self._emit_range(node, sf, lf)
        if isinstance(node, ContainsNode):
            return self._emit_contains(node, sf, lf)
        if isinstance(node, KnnNode):
            return self._emit_knn(node, sf, lf)
        if isinstance(node, FusedSpatialNode):
            return self._emit_fused(node, sf, lf)
        if isinstance(node, KnnJoinNode):
            return self._emit_knn_join(node, sf, lf)
        if isinstance(node, WithinJoinNode):
            return self._emit_within_join(node, sf, lf)
        raise TypeError(f"Unknown plan node type: {type(node)}")

    # --- filter nodes ---

    def _emit_range(self, node: RangeNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        indices = sf.engine.range_query(node.min_x, node.min_y, node.max_x, node.max_y)
        return self._filter_by_indices(lf, indices)

    def _emit_contains(self, node: ContainsNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        indices = sf.engine.contains(node.qx, node.qy)
        return self._filter_by_indices(lf, indices)

    def _emit_knn(self, node: KnnNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        indices = sf.engine.knn(node.qx, node.qy, node.k, node.approximate)
        return self._filter_by_indices(lf, indices)

    def _emit_fused(self, node: FusedSpatialNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        # Apply each predicate independently. Correctness is identical to chaining;
        # the single-index-build benefit is deferred to the Phase 4 expression plugin.
        for pred in node.predicates:
            lf = self._emit_node(pred, sf, lf)
        return lf

    def _filter_by_indices(self, lf: pl.LazyFrame, indices: list[int]) -> pl.LazyFrame:
        if not indices:
            return lf.filter(pl.lit(False))
        match_series = pl.Series("_match", indices, dtype=pl.UInt32)
        return lf.filter(pl.col(_ROW_IDX).is_in(match_series))

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


def _resolve_column_conflicts(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """Prefix any right-side columns that also appear in left with 'right_'."""
    overlap = set(left.columns) & set(right.columns)
    if overlap:
        return right.rename({c: f"right_{c}" for c in overlap})
    return right
