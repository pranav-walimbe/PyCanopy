"""
Define SpatialExecutor which walks the optimised plan and emits a Polars LazyFrame chain.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import polars as pl

from pycanopy.nodes import (
    ContainsNode,
    FusedSpatialNode,
    IntersectsSelfJoinNode,
    KnnJoinNode,
    KnnNode,
    Plan,
    PluginPath,
    PointsWithinDistanceOfPolygonNode,
    PolygonKnnJoinNode,
    PolygonWithinDistanceJoinNode,
    RangeNode,
    ScalarNode,
    SelectNode,
    WithinDistanceJoinNode,
    WithinDistanceOfPointNode,
    WithinJoinNode,
)

# Internal column tracking original row positions through scalar filters. Engine results
# are indices into the original dataset and must correlate with post-filter rows.
_ROW_IDX = "__orig_row__"

# Join node types. All carry a `query_df` probe side, which is what gets streamed
_JOIN_TYPES = (
    KnnJoinNode,
    WithinJoinNode,
    WithinDistanceJoinNode,
    PolygonWithinDistanceJoinNode,
    PolygonKnnJoinNode,
)

# Probe rows per morsel for a streamed join
MORSEL_ROWS = 262_144


def _range_plugin_expr(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    engine,
) -> pl.Expr:
    # Bounding-box filter via map_batches. is_elementwise=False is a barrier, so the closure
    # sees only post-scalar-filter rows, masked against the global index in Rust.
    def _apply(s: pl.Series) -> pl.Series:
        orig_idx = s.to_numpy()
        if len(orig_idx) == 0:
            return pl.Series("", [], dtype=pl.Boolean)
        return pl.Series("", engine.range_mask(min_x, min_y, max_x, max_y, orig_idx))

    return pl.col(_ROW_IDX).map_batches(_apply, return_dtype=pl.Boolean, is_elementwise=False)


def _contains_plugin_expr(qx: float, qy: float, engine) -> pl.Expr:
    # Point-in-polygon filter masking the surviving _ROW_IDX values against the global index in Rust
    def _apply(s: pl.Series) -> pl.Series:
        orig_idx = s.to_numpy()
        if len(orig_idx) == 0:
            return pl.Series("", [], dtype=pl.Boolean)
        return pl.Series("", engine.contains_mask(qx, qy, orig_idx))

    return pl.col(_ROW_IDX).map_batches(_apply, return_dtype=pl.Boolean, is_elementwise=False)


def _fused_plugin_expr(predicates: list[RangeNode | ContainsNode], engine) -> pl.Expr:
    # Mask applying all fused predicates, queried and intersected by sorted merge in Rust
    range_queries = [
        (pred.min_x, pred.min_y, pred.max_x, pred.max_y)
        for pred in predicates
        if isinstance(pred, RangeNode)
    ]
    contains_points = [(pred.qx, pred.qy) for pred in predicates if isinstance(pred, ContainsNode)]

    def _apply(s: pl.Series) -> pl.Series:
        orig_idx = s.to_numpy()
        if len(orig_idx) == 0:
            return pl.Series("", [], dtype=pl.Boolean)
        return pl.Series("", engine.fused_mask(range_queries, contains_points, orig_idx))

    return pl.col(_ROW_IDX).map_batches(_apply, return_dtype=pl.Boolean, is_elementwise=False)


def _knn_plugin_expr(qx: float, qy: float, k: int, engine) -> pl.Expr:
    # kNN over the M survivors from upstream scalar filters, masked in Rust by an exact
    # O(M + k log k) linear scan with no global index query.
    def _apply(s: pl.Series) -> pl.Series:
        orig_idx = s.to_numpy()  # uint32, _ROW_IDX is always UInt32 from with_row_index
        if len(orig_idx) == 0:
            return pl.Series("", [], dtype=pl.Boolean)
        return pl.Series("", engine.knn_mask_from_candidates(qx, qy, k, orig_idx))

    return pl.col(_ROW_IDX).map_batches(_apply, return_dtype=pl.Boolean, is_elementwise=False)


class SpatialExecutor:
    """Translates the optimised plan into Polars operations and executes them.

    Two execution paths are available, chosen by the optimizer via PluginPath:

    EXPR path (default): scalar filters run first (map_batches barrier), then the spatial
        closure passes the M surviving _ROW_IDX values to the Engine, which returns a Rust
        boolean mask over them. No local index is built. Best when filters retain many rows.

    IO path: the pre-built Engine is queried directly for the K candidate indices and sf.df
        is sliced to them, with scalar filters run on that slice. Best when spatial
        selectivity is tight (K << N), where rebuilding an index on M rows would be wasteful.
    """

    def execute(
        self,
        plan: Plan,
        sf,
        plugin_path: PluginPath = PluginPath.EXPR,
        batch_size: int | None = None,
    ) -> pl.DataFrame:
        """Execute the optimised plan against sf.

        The engine's index mode (eager/none/auto, fixed at frame construction)
        governs whether indexes are built, the executor does not change it.

        Args:
            plan: Execution-ordered plan from SpatialOptimizer.
            sf: SpatialFrame owning the Engine and DataFrame.
            plugin_path: Whether to use the expression plugin or IO plugin path.
            batch_size: Probe rows per morsel for streamed joins. Defaults to
                MORSEL_ROWS. A join whose probe side exceeds this is streamed in
                morsels and concatenated, bounding the join intermediate.

        Returns:
            Filtered or joined Polars DataFrame.
        """
        return self._execute(plan, sf, plugin_path, batch_size)

    def _execute(
        self,
        plan: Plan,
        sf,
        plugin_path: PluginPath,
        batch_size: int | None,
    ) -> pl.DataFrame:
        # Strip a trailing projection, run the body, then apply the final select
        projection, body = self._extract_projection(plan)
        df = self._execute_body(body, sf, plugin_path, batch_size)
        if projection is not None:
            df = df.select(list(projection))
        return df

    def _extract_projection(self, plan: Plan) -> tuple[tuple[str, ...] | None, Plan]:
        # Split a trailing SelectNode off the plan and push its keep-set onto join nodes
        if not plan or not isinstance(plan[-1], SelectNode):
            return None, plan
        output_columns = plan[-1].columns
        body = plan[:-1]
        join_positions = [i for i, n in enumerate(body) if isinstance(n, _JOIN_TYPES)]
        if not join_positions:
            return output_columns, body
        # Keep the projected columns plus any columns post-join filters read, as output names
        keep = set(output_columns)
        for node in body[join_positions[-1] + 1 :]:
            if isinstance(node, ScalarNode):
                keep |= set(node.expr.meta.root_names())
        keep_tuple = tuple(keep)
        body = [
            dataclasses.replace(n, keep_columns=keep_tuple) if isinstance(n, _JOIN_TYPES) else n
            for n in body
        ]
        return output_columns, body

    def _execute_body(
        self,
        plan: Plan,
        sf,
        plugin_path: PluginPath,
        batch_size: int | None,
    ) -> pl.DataFrame:
        # Run the optimised plan, dispatching scalar / IO / EXPR / streamed-join paths

        # Scalar-only fast path with no spatial ops
        if all(isinstance(n, ScalarNode) for n in plan):
            lf = sf.df.lazy()
            for node in plan:
                lf = lf.filter(node.expr)
            return lf.collect()

        # Intersects self-join is terminal and produces a pair frame
        if any(isinstance(n, IntersectsSelfJoinNode) for n in plan):
            return self._execute_intersects(plan, sf)

        has_joins = any(isinstance(n, _JOIN_TYPES) for n in plan)

        # Large-probe joins stream the probe in morsels and concatenate, so the join
        # intermediate is bounded by one morsel rather than the full result.
        if has_joins:
            morsel = batch_size if batch_size is not None else MORSEL_ROWS
            join_node = next(n for n in plan if isinstance(n, _JOIN_TYPES))
            if isinstance(join_node, PolygonKnnJoinNode) and join_node.sorted_output:
                return self._execute_polygon_knn_sorted(plan, sf)
            if join_node.query_df.height > morsel:
                frames = self._stream_join_frames(plan, sf, morsel)
                return pl.concat(frames, how="vertical", rechunk=False)

        # EXPR needs x_col/y_col as real columns (point datasets). Polygon frames use
        # synthetic coord names absent from df and degrade to IO.
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
    ) -> Iterator[pl.DataFrame]:
        """Yield the plan's result one morsel-frame at a time.

        For a join plan the probe is sliced into batch_size-row morsels (default MORSEL_ROWS)
        and each joined morsel is yielded on its own, so the full result never materialises.

        Args:
            plan: Execution-ordered plan from SpatialOptimizer.
            sf: SpatialFrame owning the Engine and DataFrame.
            batch_size: Probe rows per morsel. Defaults to MORSEL_ROWS.

        Yields:
            One DataFrame per morsel (one total for non-join plans).
        """
        if not any(isinstance(n, _JOIN_TYPES) for n in plan):
            yield self.execute(plan, sf)
            return
        morsel = batch_size if batch_size is not None else MORSEL_ROWS
        projection, body = self._extract_projection(plan)
        yield from self._stream_join_frames(body, sf, morsel, projection)

    def _stream_join_frames(
        self, plan: Plan, sf, morsel_rows: int, projection: tuple[str, ...] | None = None
    ) -> Iterator[pl.DataFrame]:
        # Slice the join's probe into morsels and emit each joined frame, applying
        # post-join nodes per morsel so the full result never materialises.
        join_pos = next(i for i, n in enumerate(plan) if isinstance(n, _JOIN_TYPES))
        join_node = plan[join_pos]
        post_nodes = plan[join_pos + 1 :]

        # Join emitters ignore the incoming lf (they gather from sf.df directly)
        placeholder = sf.df.lazy()
        total_probe_rows = join_node.query_df.height
        for chunk in join_node.query_df.iter_slices(morsel_rows):
            sub = dataclasses.replace(join_node, query_df=chunk, total_probe_rows=total_probe_rows)
            lf = self._emit_node(sub, sf, placeholder, PluginPath.EXPR)
            for node in post_nodes:
                lf = self._emit_node(node, sf, lf, PluginPath.EXPR)
            out = lf.collect()
            if projection is not None:
                out = out.select(list(projection))
            yield out

    def _execute_io(self, plan: Plan, sf) -> pl.DataFrame:
        # IO path: resolve every spatial node against the global Engine (no index rebuild),
        # AND-intersect the hits, slice df to candidates, then run scalars on that small slice.
        hit_lists: list[list[int]] = []
        scalar_nodes: list[ScalarNode] = []

        for node in plan:
            hits: list[int] | None = None
            if isinstance(node, RangeNode):
                hits = sf.engine.range_query(node.min_x, node.min_y, node.max_x, node.max_y)
            elif isinstance(node, ContainsNode):
                hits = sf.engine.contains(node.qx, node.qy)
            elif isinstance(node, PointsWithinDistanceOfPolygonNode):
                hits = sf.engine.points_within_distance_of_polygon(
                    node.polygon, node.distance
                ).tolist()
            elif isinstance(node, FusedSpatialNode):
                for pred in node.predicates:
                    if isinstance(pred, RangeNode):
                        hit_lists.append(
                            sf.engine.range_query(pred.min_x, pred.min_y, pred.max_x, pred.max_y)
                        )
                    elif isinstance(pred, ContainsNode):
                        hit_lists.append(sf.engine.contains(pred.qx, pred.qy))
                continue
            elif isinstance(node, ScalarNode):
                scalar_nodes.append(node)
                continue

            if hits is not None:
                hit_lists.append(hits)

        if not hit_lists:
            lf = sf.df.lazy()
        else:
            candidates = sf.engine.intersect_hits(hit_lists)
            if not candidates:
                return sf.df.clear()
            lf = sf.df[candidates].lazy()

        for node in scalar_nodes:
            lf = lf.filter(node.expr)

        return lf.collect()

    def _execute_intersects(self, plan: Plan, sf) -> pl.DataFrame:
        # Build the polygon intersects pair frame, then apply any trailing scalar filters.
        pos = next(i for i, n in enumerate(plan) if isinstance(n, IntersectsSelfJoinNode))
        lf = sf.intersects_pairs().lazy()
        for node in plan[pos + 1 :]:
            if isinstance(node, ScalarNode):
                lf = lf.filter(node.expr)
        return lf.collect()

    def _emit_chain(
        self, plan: Plan, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        # Emit each plan node in order, routing KNN through the scalar-aware emitter
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
        # Dispatch one plan node to its emitter by type
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
        if isinstance(node, PointsWithinDistanceOfPolygonNode):
            return self._emit_points_within_distance_of_polygon(node, sf, lf)
        if isinstance(node, WithinDistanceOfPointNode):
            return self._emit_within_distance_of_point(node, sf, lf)
        raise TypeError(f"Unknown plan node type: {type(node)}")

    # filter nodes

    def _emit_range(
        self, node: RangeNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        # Emit a range filter as an EXPR mask or an IO index slice
        if plugin_path == PluginPath.EXPR:
            return lf.filter(
                _range_plugin_expr(node.min_x, node.min_y, node.max_x, node.max_y, sf.engine)
            )
        indices = sf.engine.range_query(node.min_x, node.min_y, node.max_x, node.max_y)
        return self._filter_by_indices(lf, indices)

    def _emit_contains(
        self, node: ContainsNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        # Emit a point-in-polygon filter as an EXPR mask or an IO index slice
        if plugin_path == PluginPath.EXPR:
            return lf.filter(_contains_plugin_expr(node.qx, node.qy, sf.engine))
        indices = sf.engine.contains(node.qx, node.qy)
        return self._filter_by_indices(lf, indices)

    def _emit_knn(
        self, node: KnnNode, sf, lf: pl.LazyFrame, has_prior_scalar: bool = False
    ) -> pl.LazyFrame:
        # Emit kNN over post-scalar survivors via EXPR, otherwise a global index query
        if has_prior_scalar:
            # Scalars ran first. M survivors arrive via _ROW_IDX. A linear scan over those
            # M rows finds the k nearest without a global index query.
            return lf.filter(_knn_plugin_expr(node.qx, node.qy, node.k, sf.engine))
        indices = sf.engine.knn(node.qx, node.qy, node.k)
        return self._filter_by_indices(lf, indices)

    def _emit_fused(
        self, node: FusedSpatialNode, sf, lf: pl.LazyFrame, plugin_path: PluginPath
    ) -> pl.LazyFrame:
        # Emit fused predicates as one EXPR mask, or as separate IO filters
        if plugin_path == PluginPath.EXPR:
            return lf.filter(_fused_plugin_expr(node.predicates, sf.engine))
        for pred in node.predicates:
            lf = self._emit_node(pred, sf, lf, plugin_path)
        return lf

    def _emit_points_within_distance_of_polygon(
        self, node: PointsWithinDistanceOfPolygonNode, sf, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        # Keep points within node.distance of the query polygon. The polygon is queried
        # against all points, so indices resolve once and the lf filters by original row.
        indices = sf.engine.points_within_distance_of_polygon(node.polygon, node.distance)
        return self._filter_by_indices(lf, indices)

    def _emit_within_distance_of_point(
        self, node: WithinDistanceOfPointNode, sf, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        # One radius query resolves the indices, then the lf filters by original row
        indices = sf.engine.radius_query(node.cx, node.cy, node.distance)
        return self._filter_by_indices(lf, indices)

    def _filter_by_indices(self, lf: pl.LazyFrame, indices) -> pl.LazyFrame:
        # Filter the lf to the given original row positions (list or numpy array), empty to nothing
        if len(indices) == 0:
            return lf.filter(pl.lit(False))
        return lf.filter(pl.col(_ROW_IDX).is_in(indices))

    # join nodes

    def _assemble_join(self, node, sf, q_idx: pl.Series, t_idx: pl.Series) -> pl.DataFrame:
        # Gather both join sides at the paired indices, narrowed to node.keep_columns, with
        # 'right_' prefixing right-side name collisions and unkept sides dropped before gather.
        query_cols = node.query_df.columns
        target_cols = sf.df.columns
        overlap = set(query_cols) & set(target_cols)
        keep = None if node.keep_columns is None else set(node.keep_columns)
        if keep is None:
            query_keep, target_keep = query_cols, target_cols
        else:
            query_keep = [c for c in query_cols if c in keep]
            target_keep = [c for c in target_cols if (f"right_{c}" if c in overlap else c) in keep]

        parts: list[pl.DataFrame] = []
        if query_keep:
            parts.append(node.query_df.select(query_keep).gather(q_idx))
        if target_keep:
            target_part = sf.df.select(target_keep).gather(t_idx)
            rename = {c: f"right_{c}" for c in target_keep if c in overlap}
            if rename:
                target_part = target_part.rename(rename)
            parts.append(target_part)

        if not parts:
            return pl.DataFrame()
        if len(parts) == 1:
            return parts[0]
        return pl.concat(parts, how="horizontal")

    def _emit_knn_join(self, node: KnnJoinNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        # For each row in query_df, find the k nearest in the Engine's dataset
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()
        n_queries = len(node.query_df)

        total_q = n_queries if node.total_probe_rows is None else node.total_probe_rows
        # batch_knn_join returns a flat (n_queries * k,) array, each query row repeats k times
        match_indices = sf.engine.batch_knn_join(query_xs, query_ys, node.k, total_q)
        q_idx = pl.Series("", np.repeat(np.arange(n_queries, dtype=np.uint32), node.k))
        t_idx = pl.Series("", match_indices.astype(np.uint32))
        return self._assemble_join(node, sf, q_idx, t_idx).lazy()

    def _emit_within_join(self, node: WithinJoinNode, sf, lf: pl.LazyFrame) -> pl.LazyFrame:
        # For each point in query_df, find the Engine polygons that contain it
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()
        total_q = len(node.query_df) if node.total_probe_rows is None else node.total_probe_rows

        # batch_contains returns flat (M * 2,) array: [q0, e0, q1, e1, ...].
        pairs = sf.engine.batch_contains(query_xs, query_ys, total_q).reshape(-1, 2)
        q_idx = pl.Series("", pairs[:, 0].astype(np.uint32))
        t_idx = pl.Series("", pairs[:, 1].astype(np.uint32))
        return self._assemble_join(node, sf, q_idx, t_idx).lazy()

    def _emit_within_distance_join(
        self, node: WithinDistanceJoinNode, sf, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        # For each query point, find Engine points within node.distance
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()
        total_q = len(node.query_df) if node.total_probe_rows is None else node.total_probe_rows

        pairs = sf.engine.batch_within_distance(
            query_xs, query_ys, node.distance, node.flip, total_q
        ).reshape(-1, 2)
        q_idx = pl.Series("", pairs[:, 0].astype(np.uint32))
        t_idx = pl.Series("", pairs[:, 1].astype(np.uint32))
        return self._assemble_join(node, sf, q_idx, t_idx).lazy()

    def _emit_polygon_within_distance_join(
        self, node: PolygonWithinDistanceJoinNode, sf, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        # For each query point find Engine polygons within node.distance
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()
        total_q = len(node.query_df) if node.total_probe_rows is None else node.total_probe_rows

        pairs = sf.engine.batch_within_distance_to_polygons(
            query_xs, query_ys, node.distance, total_q
        ).reshape(-1, 2)
        q_idx = pl.Series("", pairs[:, 0].astype(np.uint32))
        t_idx = pl.Series("", pairs[:, 1].astype(np.uint32))
        return self._assemble_join(node, sf, q_idx, t_idx).lazy()

    def _emit_polygon_knn_join(
        self, node: PolygonKnnJoinNode, sf, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        # For each query point find its k nearest Engine polygons, appending distance_to_polygon
        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()
        n_queries = len(node.query_df)
        total_q = n_queries if node.total_probe_rows is None else node.total_probe_rows

        indices, dists = sf.engine.batch_knn_to_polygons(query_xs, query_ys, node.k, total_q)

        q_idx_full = np.repeat(np.arange(n_queries, dtype=np.uint64), node.k)
        # Drop padding slots (no polygon for that rank)
        keep = indices != np.iinfo(np.uint64).max
        q_idx = pl.Series("", q_idx_full[keep].astype(np.uint32))
        t_idx = pl.Series("", indices[keep].astype(np.uint32))

        joined = self._assemble_join(node, sf, q_idx, t_idx)
        dist_series = pl.Series("distance_to_polygon", dists[keep])
        # joined has no width only when the projection kept just distance_to_polygon
        if joined.width == 0:
            return pl.DataFrame([dist_series]).lazy()
        return joined.with_columns(dist_series).lazy()

    def _execute_polygon_knn_sorted(self, plan: Plan, sf) -> pl.DataFrame:
        # Full-probe path for sorted_output=True: runs all queries in one Rust call,
        # gets back globally sorted pairs, assembles join, applies post-join nodes.
        projection, body = self._extract_projection(plan)
        join_pos = next(i for i, n in enumerate(body) if isinstance(n, _JOIN_TYPES))
        node = body[join_pos]
        post_nodes = body[join_pos + 1 :]

        query_xs = node.query_df[node.x_col].to_numpy()
        query_ys = node.query_df[node.y_col].to_numpy()

        q_indices, t_indices, dists = sf.engine.batch_knn_to_polygons_sorted(
            query_xs, query_ys, node.k
        )
        q_idx = pl.Series("", q_indices.astype(np.uint32))
        t_idx = pl.Series("", t_indices.astype(np.uint32))

        joined = self._assemble_join(node, sf, q_idx, t_idx)
        dist_series = pl.Series("distance_to_polygon", dists)
        if joined.width == 0:
            df = pl.DataFrame([dist_series])
        else:
            df = joined.with_columns(dist_series)

        lf = df.lazy()
        for post in post_nodes:
            lf = self._emit_node(post, sf, lf, PluginPath.EXPR)
        df = lf.collect()
        if projection is not None:
            df = df.select(list(projection))
        return df
