"""SpatialExecutor — walks the optimised plan and emits a Polars LazyFrame chain."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import polars as pl

from pycanopy.nodes import (
    ContainsNode,
    FusedSpatialNode,
    KnnJoinNode,
    KnnNode,
    Plan,
    PluginPath,
    PolygonKnnJoinNode,
    PolygonWithinDistanceJoinNode,
    RangeNode,
    ScalarNode,
    WithinDistanceJoinNode,
    WithinJoinNode,
)

# Internal column name used to track original row positions through scalar filters.
# Engine results are always indices into the original dataset, so we need to
# correlate them with post-filter rows using this persistent index.
_ROW_IDX = "__orig_row__"

# Join node types. All carry a `query_df` probe side, which is what gets streamed.
_JOIN_TYPES = (
    KnnJoinNode,
    WithinJoinNode,
    WithinDistanceJoinNode,
    PolygonWithinDistanceJoinNode,
    PolygonKnnJoinNode,
)

# Probe rows processed per morsel when a join's query side is streamed. A fixed,
# cache and memory friendly constant in the spirit of a vectorised execution unit
# (DuckDB-style: a tuned constant, not a per-query fanout prediction). At typical
# fanout the per-morsel transient is a few hundred MB, and the value is validated at
# SF1. Callers override per query via batch_size on collect / collect_batched. A join
# whose probe fits in one morsel skips streaming entirely.
MORSEL_ROWS = 1_048_576


def _range_plugin_expr(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    engine,
) -> pl.Expr:
    """Boolean mask expression for a bounding-box filter via map_batches.

    is_elementwise=False acts as a Polars barrier: scalar predicates upstream
    execute first, so the closure receives only the M already-filtered rows via
    their _ROW_IDX values. The pre-built global Engine is queried and the hit
    bitmap is indexed by the surviving original row positions to produce the mask.
    """
    n_total = engine.n

    def _apply(s: pl.Series) -> pl.Series:
        orig_idx = s.to_numpy()
        if len(orig_idx) == 0:
            return pl.Series("", [], dtype=pl.Boolean)
        hits = engine.range_query(min_x, min_y, max_x, max_y)
        hit_bitmap = np.zeros(n_total, dtype=bool)
        if hits:
            hit_bitmap[hits] = True
        return pl.Series("", hit_bitmap[orig_idx], dtype=pl.Boolean)

    return pl.col(_ROW_IDX).map_batches(_apply, return_dtype=pl.Boolean, is_elementwise=False)


def _contains_plugin_expr(qx: float, qy: float, engine) -> pl.Expr:
    """Boolean mask expression for a point exact-match filter via map_batches.

    Queries the pre-built global Engine and intersects hits with the surviving
    _ROW_IDX values from upstream scalar filters via a hit bitmap.
    """
    n_total = engine.n

    def _apply(s: pl.Series) -> pl.Series:
        orig_idx = s.to_numpy()
        if len(orig_idx) == 0:
            return pl.Series("", [], dtype=pl.Boolean)
        hits = engine.contains(qx, qy)
        hit_bitmap = np.zeros(n_total, dtype=bool)
        if hits:
            hit_bitmap[hits] = True
        return pl.Series("", hit_bitmap[orig_idx], dtype=pl.Boolean)

    return pl.col(_ROW_IDX).map_batches(_apply, return_dtype=pl.Boolean, is_elementwise=False)


def _fused_plugin_expr(predicates: list[RangeNode | ContainsNode], engine) -> pl.Expr:
    """Boolean mask expression applying all fused predicates via sorted merge in Rust.

    Range predicates are intersected via engine.intersect_ranges — a Rust sorted merge
    on hit arrays, O(K * |H|) rather than O(K * N) bitmap AND passes. Contains predicates
    are intersected separately on the Python side (each returns at most a handful of hits).
    """
    n_total = engine.n
    range_queries = [
        (pred.min_x, pred.min_y, pred.max_x, pred.max_y)
        for pred in predicates
        if isinstance(pred, RangeNode)
    ]
    contains_preds = [pred for pred in predicates if isinstance(pred, ContainsNode)]

    def _apply(s: pl.Series) -> pl.Series:
        orig_idx = s.to_numpy()
        if len(orig_idx) == 0:
            return pl.Series("", [], dtype=pl.Boolean)
        hits: list[int] | None = engine.intersect_ranges(range_queries) if range_queries else None
        for pred in contains_preds:
            c_hits = set(engine.contains(pred.qx, pred.qy))
            hits = sorted(c_hits) if hits is None else sorted(set(hits) & c_hits)
            if not hits:
                return pl.Series("", np.zeros(len(orig_idx), dtype=bool), dtype=pl.Boolean)
        hit_bitmap = np.zeros(n_total, dtype=bool)
        if hits:
            hit_bitmap[hits] = True
        return pl.Series("", hit_bitmap[orig_idx], dtype=pl.Boolean)

    return pl.col(_ROW_IDX).map_batches(_apply, return_dtype=pl.Boolean, is_elementwise=False)


def _knn_plugin_expr(qx: float, qy: float, k: int, engine) -> pl.Expr:
    """Boolean mask for kNN restricted to M survivors from upstream scalar filters.

    is_elementwise=False acts as a Polars barrier — scalar predicates upstream
    execute first. The closure receives M _ROW_IDX values and delegates to
    knn_from_candidates: a linear scan over the M survivors that computes squared
    distances directly from the coordinate arrays and partial-sorts to find the k
    nearest. O(M + k log k), exact, no global index query.
    """
    n_total = engine.n

    def _apply(s: pl.Series) -> pl.Series:
        orig_idx = s.to_numpy()  # uint32 — _ROW_IDX is always UInt32 from with_row_index
        if len(orig_idx) == 0:
            return pl.Series("", [], dtype=pl.Boolean)
        hits = engine.knn_from_candidates(qx, qy, k, orig_idx)
        hit_bitmap = np.zeros(n_total, dtype=bool)
        if hits:
            hit_bitmap[hits] = True
        return pl.Series("", hit_bitmap[orig_idx], dtype=pl.Boolean)

    return pl.col(_ROW_IDX).map_batches(_apply, return_dtype=pl.Boolean, is_elementwise=False)


class SpatialExecutor:
    """Translates the optimised plan into Polars operations and executes them.

    Two execution paths are available, chosen by the optimizer via PluginPath:

    EXPR path (default): spatial nodes emit map_batches(is_elementwise=False)
        expressions. Polars runs scalar filters first (barrier semantics), then the
        spatial closure receives the M surviving _ROW_IDX values, queries the global
        Engine, and intersects hits with a boolean bitmap indexed by original row
        position. No local index is built regardless of how many rows scalar filters
        retain. A persistent _ROW_IDX column tracks original positions throughout.

    IO path: the pre-built Engine (on N rows) is queried directly to get K candidate
        indices. sf.df is sliced to those K rows. Scalar filters run on the K-row
        slice. No _ROW_IDX column needed. Best when spatial selectivity is tight
        (K << N) and re-building a fresh index on M rows would be wasteful.
    """

    def execute(
        self,
        plan: Plan,
        sf,
        plugin_path: PluginPath = PluginPath.EXPR,
        batch_size: int | None = None,
        auto_index: bool | None = None,
    ) -> pl.DataFrame:
        """Execute the optimised plan against sf.

        Args:
            plan: Execution-ordered plan from SpatialOptimizer.
            sf: SpatialFrame owning the Engine and DataFrame.
            plugin_path: Whether to use the expression plugin or IO plugin path.
            batch_size: Probe rows per morsel for streamed joins. Defaults to
                MORSEL_ROWS. A join whose probe side exceeds this is streamed in
                morsels and concatenated, bounding the join intermediate.
            auto_index: True/False overrides the engine's index mode for this call
                (False scans brute-force, building no index); None inherits whatever
                the engine is already configured for. The prior mode is restored after.

        Returns:
            Filtered or joined Polars DataFrame.
        """
        if auto_index is None:
            return self._execute(plan, sf, plugin_path, batch_size)
        prev_auto_index = sf.engine.set_auto_index(auto_index)
        try:
            return self._execute(plan, sf, plugin_path, batch_size)
        finally:
            sf.engine.set_auto_index(prev_auto_index)

    def _execute(
        self,
        plan: Plan,
        sf,
        plugin_path: PluginPath,
        batch_size: int | None,
    ) -> pl.DataFrame:
        """Run the plan with the engine's auto_index flag already set by execute."""
        # EXPR path requires x_col/y_col as real DataFrame columns (point datasets).
        # Polygon SpatialFrames use synthetic coordinate column names that don't
        # exist in df; degrade gracefully to the IO path in that case.
        # Exception: join nodes are handled directly by the executor and do not
        # use the plugin expression machinery, so they stay on EXPR path.
        # Scalar-only fast path: no spatial ops means no index machinery needed.
        # Feed filters directly to Polars as a zero-copy lazy chain — no _ROW_IDX
        # column, no executor dispatch overhead, no bitmap allocation.
        if all(isinstance(n, ScalarNode) for n in plan):
            lf = sf.df.lazy()
            for node in plan:
                lf = lf.filter(node.expr)
            return lf.collect()

        has_joins = any(isinstance(n, _JOIN_TYPES) for n in plan)

        # Large-probe joins stream the probe in morsels and concatenate, so the join
        # intermediate is bounded by one morsel rather than the full result. Small
        # probes fall through to the single-shot path below (no slicing overhead).
        if has_joins:
            morsel = batch_size if batch_size is not None else MORSEL_ROWS
            join_node = next(n for n in plan if isinstance(n, _JOIN_TYPES))
            if join_node.query_df.height > morsel:
                frames = self._stream_join_frames(plan, sf, morsel)
                return pl.concat(frames, how="vertical", rechunk=True)

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

    def stream(
        self,
        plan: Plan,
        sf,
        batch_size: int | None = None,
        auto_index: bool | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Yield the plan's result one morsel-frame at a time.

        For a plan containing a spatial join the probe side is sliced into morsels
        of batch_size rows (default MORSEL_ROWS); each morsel is joined and yielded
        as its own DataFrame, so a caller can reduce it (group_by, count) before the
        next morsel is computed and the full join result never materialises at once.

        Plans without a join yield a single frame (the whole result) so the iterator
        form is always usable.

        Args:
            plan: Execution-ordered plan from SpatialOptimizer.
            sf: SpatialFrame owning the Engine and DataFrame.
            batch_size: Probe rows per morsel. Defaults to MORSEL_ROWS.
            auto_index: True/False overrides the engine's index mode for the whole
                stream (False scans brute-force); None inherits the engine's setting.

        Yields:
            One DataFrame per morsel (one total for non-join plans).
        """
        if not any(isinstance(n, _JOIN_TYPES) for n in plan):
            yield self.execute(plan, sf, auto_index=auto_index)
            return
        morsel = batch_size if batch_size is not None else MORSEL_ROWS
        if auto_index is None:
            yield from self._stream_join_frames(plan, sf, morsel)
            return
        # The morsels are consumed lazily by the caller, so an explicit override must
        # persist across the whole iterator, not just the call that builds it.
        prev_auto_index = sf.engine.set_auto_index(auto_index)
        try:
            yield from self._stream_join_frames(plan, sf, morsel)
        finally:
            sf.engine.set_auto_index(prev_auto_index)

    def _stream_join_frames(self, plan: Plan, sf, morsel_rows: int) -> Iterator[pl.DataFrame]:
        """Slice the join's probe into morsels, emit each joined frame in turn.

        The first join node is the streaming axis; its query_df is sliced with
        iter_slices (a zero-copy view per slice). Each slice is run through the
        existing join emitter with query_df replaced by the slice, then any nodes
        after the join are applied to that morsel's result. Streaming is exact
        because a join is row-independent: a probe row's matches do not depend on
        which morsel its neighbours fall in, so the morsels partition the result.

        Args:
            plan: Execution-ordered plan containing at least one join node.
            sf: SpatialFrame owning the Engine and DataFrame.
            morsel_rows: Probe rows per morsel.

        Yields:
            One joined DataFrame per probe morsel.
        """
        join_pos = next(i for i, n in enumerate(plan) if isinstance(n, _JOIN_TYPES))
        join_node = plan[join_pos]
        post_nodes = plan[join_pos + 1 :]
        # Join emitters ignore the incoming lf (they gather from sf.df directly), so a
        # placeholder is fine; post-join nodes receive the real joined lf.
        placeholder = sf.df.lazy()
        for chunk in join_node.query_df.iter_slices(morsel_rows):
            sub = dataclasses.replace(join_node, query_df=chunk)
            lf = self._emit_node(sub, sf, placeholder, PluginPath.EXPR)
            for node in post_nodes:
                lf = self._emit_node(node, sf, lf, PluginPath.EXPR)
            yield lf.collect()

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
                    if not candidate_indices:
                        return sf.df.clear()
                continue
            elif isinstance(node, ScalarNode):
                scalar_nodes.append(node)
                continue

            if hits is not None:
                hits_set = set(hits)
                candidate_indices = (
                    hits_set if candidate_indices is None else candidate_indices & hits_set
                )
                if not candidate_indices:
                    return sf.df.clear()

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
        has_seen_scalar = False
        for node in plan:
            if isinstance(node, KnnNode):
                lf = self._emit_knn(node, sf, lf, has_seen_scalar)
            else:
                lf = self._emit_node(node, sf, lf, plugin_path)
            if isinstance(node, ScalarNode):
                has_seen_scalar = True
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
        if isinstance(node, PolygonWithinDistanceJoinNode):
            return self._emit_polygon_within_distance_join(node, sf, lf)
        if isinstance(node, PolygonKnnJoinNode):
            return self._emit_polygon_knn_join(node, sf, lf)
        raise TypeError(f"Unknown plan node type: {type(node)}")

    # filter nodes

    def _emit_range(
        self, node: RangeNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        if plugin_path == PluginPath.EXPR:
            return lf.filter(
                _range_plugin_expr(node.min_x, node.min_y, node.max_x, node.max_y, sf.engine)
            )
        indices = sf.engine.range_query(node.min_x, node.min_y, node.max_x, node.max_y)
        return self._filter_by_indices(lf, indices)

    def _emit_contains(
        self, node: ContainsNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        if plugin_path == PluginPath.EXPR:
            return lf.filter(_contains_plugin_expr(node.qx, node.qy, sf.engine))
        indices = sf.engine.contains(node.qx, node.qy)
        return self._filter_by_indices(lf, indices)

    def _emit_knn(
        self, node: KnnNode, sf, lf: pl.LazyFrame, has_prior_scalar: bool = False
    ) -> pl.LazyFrame:
        if has_prior_scalar:
            # Scalars ran first — M survivors arrive via _ROW_IDX. Linear scan over
            # those M rows to find k nearest; no global index query needed.
            return lf.filter(_knn_plugin_expr(node.qx, node.qy, node.k, sf.engine))
        indices = sf.engine.knn(node.qx, node.qy, node.k, node.approximate)
        return self._filter_by_indices(lf, indices)

    def _emit_fused(
        self, node: FusedSpatialNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        if plugin_path == PluginPath.EXPR:
            return lf.filter(_fused_plugin_expr(node.predicates, sf.engine))
        for pred in node.predicates:
            lf = self._emit_node(pred, sf, lf, plugin_path)
        return lf

    def _filter_by_indices(self, lf: pl.LazyFrame, indices: list[int]) -> pl.LazyFrame:
        if not indices:
            return lf.filter(pl.lit(False))
        return lf.filter(pl.col(_ROW_IDX).is_in(indices))

    # join nodes

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
        q_idx = pl.Series("", np.repeat(np.arange(n_queries, dtype=np.uint32), node.k))
        t_idx = pl.Series("", match_indices.astype(np.uint32))

        query_part = node.query_df.gather(q_idx)
        target_part = sf.df.gather(t_idx)

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
        q_idx = pl.Series("", pairs[:, 0].astype(np.uint32))
        t_idx = pl.Series("", pairs[:, 1].astype(np.uint32))

        query_part = node.query_df.gather(q_idx)
        target_part = sf.df.gather(t_idx)

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
        q_idx = pl.Series("", pairs[:, 0].astype(np.uint32))
        t_idx = pl.Series("", pairs[:, 1].astype(np.uint32))

        query_part = node.query_df.gather(q_idx)
        target_part = sf.df.gather(t_idx)

        target_part = _resolve_column_conflicts(query_part, target_part)
        return pl.concat([query_part, target_part], how="horizontal").lazy()

    def _emit_polygon_within_distance_join(
        self, node: PolygonWithinDistanceJoinNode, sf, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """For each query point find Engine polygons within node.distance.

        Engine must be a polygon dataset. One row per (query, polygon) match.
        """
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()

        pairs_flat = sf.engine.batch_within_distance_to_polygons(query_xs, query_ys, node.distance)

        if len(pairs_flat) == 0:
            empty_q = node.query_df.clear()
            empty_t = _resolve_column_conflicts(empty_q, sf.df.clear())
            return pl.concat([empty_q, empty_t], how="horizontal").lazy()

        pairs = pairs_flat.reshape(-1, 2)
        q_idx = pl.Series("", pairs[:, 0].astype(np.uint32))
        t_idx = pl.Series("", pairs[:, 1].astype(np.uint32))

        query_part = node.query_df.gather(q_idx)
        target_part = sf.df.gather(t_idx)

        target_part = _resolve_column_conflicts(query_part, target_part)
        return pl.concat([query_part, target_part], how="horizontal").lazy()

    def _emit_polygon_knn_join(
        self, node: PolygonKnnJoinNode, sf, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """For each query point find its k nearest Engine polygons.

        Engine must be a polygon dataset. Appends a 'distance_to_polygon' column.
        Padding slots (queries with fewer than k polygons available) are dropped.
        """
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()
        n_queries = len(node.query_df)

        indices, dists = sf.engine.batch_knn_to_polygons(query_xs, query_ys, node.k)

        q_idx_full = np.repeat(np.arange(n_queries, dtype=np.uint64), node.k)
        # Drop padding slots (no polygon for that rank).
        keep = indices != np.iinfo(np.uint64).max
        q_idx = pl.Series("", q_idx_full[keep].astype(np.uint32))
        t_idx = pl.Series("", indices[keep].astype(np.uint32))
        kept_dists = dists[keep]

        if len(t_idx) == 0:
            empty_q = node.query_df.clear()
            empty_t = _resolve_column_conflicts(empty_q, sf.df.clear())
            out = pl.concat([empty_q, empty_t], how="horizontal")
            return out.with_columns(pl.Series("distance_to_polygon", [], dtype=pl.Float64)).lazy()

        query_part = node.query_df.gather(q_idx)
        target_part = sf.df.gather(t_idx)
        target_part = _resolve_column_conflicts(query_part, target_part)
        joined = pl.concat([query_part, target_part], how="horizontal")
        return joined.with_columns(pl.Series("distance_to_polygon", kept_dists)).lazy()


def _resolve_column_conflicts(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """Prefix any right-side columns that also appear in left with 'right_'."""
    overlap = set(left.columns) & set(right.columns)
    if overlap:
        return right.rename({c: f"right_{c}" for c in overlap})
    return right
