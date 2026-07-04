"""
Define SpatialLazyFrame, an immutable plan builder that does not execute until .collect().
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from polars.io.plugins import register_io_source

from pycanopy.agg import AggSpec, _partial_agg, _reduce_partials
from pycanopy.executor import _ROW_IDX, SpatialExecutor
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
from pycanopy.optimizer import SpatialOptimizer


def _fmt_expr(expr: pl.Expr) -> str:
    # Format a Polars expression as a compact one-line string
    s = str(expr)
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return s.strip()


def _fmt_node(node) -> str:
    # Format one plan node as its explain() line or lines
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
        return f"KNN [k={node.k}, ({node.qx:.4g}, {node.qy:.4g})]"
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
        return f"KNN_JOIN [k={node.k}, query_rows={len(node.query_df):,}, barrier]"
    if isinstance(node, WithinJoinNode):
        flip = ", flip" if node.flip else ""
        return f"WITHIN_JOIN [query_rows={len(node.query_df):,}, barrier{flip}]"
    if isinstance(node, WithinDistanceJoinNode):
        flip = ", flip" if node.flip else ""
        return f"WITHIN_DIST_JOIN [dist={node.distance:.4g}, query_rows={len(node.query_df):,}, barrier{flip}]"
    if isinstance(node, PolygonWithinDistanceJoinNode):
        return (
            f"POLY_WITHIN_DIST_JOIN [dist={node.distance:.4g}, "
            f"query_rows={len(node.query_df):,}, barrier]"
        )
    if isinstance(node, PolygonKnnJoinNode):
        return f"POLY_KNN_JOIN [k={node.k}, query_rows={len(node.query_df):,}, barrier]"
    if isinstance(node, PointsWithinDistanceOfPolygonNode):
        return f"POINTS_WITHIN_DIST_OF_POLY [dist={node.distance:.4g}]"
    if isinstance(node, WithinDistanceOfPointNode):
        return (
            f"WITHIN_DIST_OF_POINT [center=({node.cx:.4g}, {node.cy:.4g}), "
            f"dist={node.distance:.4g}, sel={node.selectivity:.3g}]"
        )
    if isinstance(node, IntersectsSelfJoinNode):
        return "INTERSECTS_SELF_JOIN [pairs, barrier]"
    return f"UNKNOWN [{type(node).__name__}]"


def _fmt_plan(plan: Plan, path: PluginPath | None, n: int) -> str:
    # Format the full plan as Polars-style indented explain() text
    path_suffix = ""
    if path is not None:
        path_label = "EXPR" if path == PluginPath.EXPR else "IO"
        path_suffix = f"; path: {path_label}"
    df_line = f"DF [N={n:,}{path_suffix}]"

    if not plan:
        return df_line

    # Polars convention: outermost (last executed) op at top, source at bottom
    # Each op is followed by FROM, then its source indented one level deeper.
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

    All methods return a new SpatialLazyFrame with the node appended without mutation.
    Join and kNN nodes act as barriers and are never reordered by the cost sort.

    Args:
        sf: The SpatialFrame that owns the Engine and DataFrame.
        plan: Current list of plan nodes (do not mutate directly).
    """

    def __init__(self, sf: SpatialFrame, plan: Plan) -> None:  # noqa: F821
        self._sf = sf
        self._plan = plan

    def filter(self, expr: pl.Expr) -> SpatialLazyFrame:
        """Add a scalar Polars expression filter.

        Args:
            expr: Any Polars expression that evaluates to a boolean column.

        Returns:
            New SpatialLazyFrame with the scalar node appended.
        """
        return SpatialLazyFrame(self._sf, [*self._plan, ScalarNode(expr)])

    def select(self, *columns: str | list[str] | tuple[str, ...]) -> SpatialLazyFrame:
        """Restrict the collected output to these columns, pushed into a join gather when present.

        Args:
            columns: Output column names to keep, as varargs or a single list/tuple.

        Returns:
            New SpatialLazyFrame with the terminal select node appended.
        """
        if len(columns) == 1 and isinstance(columns[0], (list, tuple)):
            cols = tuple(columns[0])
        else:
            cols = tuple(columns)
        return SpatialLazyFrame(self._sf, [*self._plan, SelectNode(cols)])

    def group_by(self, *keys: str | list[str] | tuple[str, ...]) -> SpatialGroupBy:
        """Begin a grouped aggregation, reduced over the streamed join.

        Args:
            keys: Group-by key columns, as varargs or a single list/tuple.

        Returns:
            A SpatialGroupBy builder. Call .agg() to run the aggregation.
        """
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            key_cols = list(keys[0])
        else:
            key_cols = list(keys)
        return SpatialGroupBy(self, key_cols)

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
    ) -> SpatialLazyFrame:
        """Add a k-nearest-neighbour lookup.

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            k: Number of neighbours to return.

        Returns:
            New SpatialLazyFrame with the knn node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, KnnNode(x, y, k)],
        )

    def knn_join(
        self,
        query_df: pl.DataFrame,
        x_col: str,
        y_col: str,
        k: int,
    ) -> SpatialLazyFrame:
        """Spatial join: for each row in query_df find its k nearest in this Engine's dataset.

        Result columns are query_df's followed by the Engine df's (conflicting right-side
        columns are prefixed with 'right_').

        Args:
            query_df: DataFrame of query points.
            x_col: Column in query_df holding x coordinates.
            y_col: Column in query_df holding y coordinates.
            k: Number of neighbours per query row.

        Returns:
            New SpatialLazyFrame with the knn join node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, KnnJoinNode(query_df, x_col, y_col, k)],
        )

    def within_distance_join(
        self,
        query_df: pl.DataFrame,
        x_col: str,
        y_col: str,
        distance: float,
    ) -> SpatialLazyFrame:
        """Spatial join: for each point in query_df find Engine points within `distance`.

        Result columns are query_df's followed by the Engine df's (conflicting right-side
        columns are prefixed with 'right_').

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
        """Spatial join: for each point in query_df find the Engine polygons that contain it.

        Engine must be a polygon dataset. Result columns are query_df's then the Engine df's
        (conflicting right-side columns are prefixed with 'right_').

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

    def polygon_within_distance_join(
        self,
        query_df: pl.DataFrame,
        x_col: str,
        y_col: str,
        distance: float,
    ) -> SpatialLazyFrame:
        """Spatial join: for each point in query_df find Engine polygons within `distance`.

        Distance is to the polygon boundary (zero when the point is inside). Result columns
        are query_df's then the Engine df's (conflicting right-side columns prefixed 'right_').

        Args:
            query_df: DataFrame of query points.
            x_col: Column in query_df holding x coordinates.
            y_col: Column in query_df holding y coordinates.
            distance: Maximum point-to-polygon distance for a match.

        Returns:
            New SpatialLazyFrame with the polygon within-distance join node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, PolygonWithinDistanceJoinNode(query_df, x_col, y_col, distance)],
        )

    def polygon_knn_join(
        self,
        query_df: pl.DataFrame,
        x_col: str,
        y_col: str,
        k: int,
        sorted_output: bool = False,
    ) -> SpatialLazyFrame:
        """Spatial join: for each point in query_df find its k nearest Engine polygons.

        Ranking is by exact point-to-polygon distance and a 'distance_to_polygon' column
        is appended.

        Args:
            query_df: DataFrame of query points.
            x_col: Column in query_df holding x coordinates.
            y_col: Column in query_df holding y coordinates.
            k: Number of nearest polygons per query point.
            sorted_output: If True, all pairs are sorted by (distance_to_polygon ASC,
                target_idx ASC) inside Rust via rayon before returning. The full result
                materialises in RAM, so morsel streaming is bypassed. Matches
                ORDER BY distance_to_building, b_buildingkey without a Polars sort step.

        Returns:
            New SpatialLazyFrame with the polygon kNN join node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [
                *self._plan,
                PolygonKnnJoinNode(query_df, x_col, y_col, k, sorted_output=sorted_output),
            ],
        )

    def points_within_distance_of_polygon(self, polygon, distance: float) -> SpatialLazyFrame:
        """Keep points within `distance` of a single query polygon (point dataset).

        Distance is measured to the polygon boundary (zero when the point is inside). The
        result is a subset of this frame's rows like a spatial filter.

        Args:
            polygon: A single shapely Polygon (interior holes supported).
            distance: Maximum point-to-polygon distance for a row to be kept.

        Returns:
            New SpatialLazyFrame with the points-within-distance node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, PointsWithinDistanceOfPolygonNode(polygon, distance)],
        )

    def within_distance_of_point(self, cx: float, cy: float, distance: float) -> SpatialLazyFrame:
        """Keep points within `distance` of a single center (point dataset).

        Distance is Euclidean. The result is a subset of this frame's rows like a spatial
        filter.

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            distance: Maximum Euclidean distance for a row to be kept.

        Returns:
            New SpatialLazyFrame with the within-distance-of-point node appended.
        """
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, WithinDistanceOfPointNode(cx, cy, distance)],
        )

    def intersects_pairs(self) -> SpatialLazyFrame:
        """Find all intersecting polygon pairs with overlap area and IoU (polygon dataset).

        Returns:
            New SpatialLazyFrame with the intersects self-join node appended.
        """
        return SpatialLazyFrame(self._sf, [*self._plan, IntersectsSelfJoinNode()])

    def explain(self) -> str:
        """Return a human-readable description of the computed query plan.

        Shows the optimised plan that collect() will execute (reordered operations, fused
        predicates, chosen EXPR or IO path) rather than the declaration order.

        Returns:
            Multi-line string describing the plan. Print it for readable output.
        """
        engine = self._sf.engine
        opt = SpatialOptimizer()
        plan = opt.optimize(self._plan, engine)
        path = opt._select_plugin_path(plan, engine)
        return _fmt_plan(plan, path, engine.n)

    def collect(self, batch_size: int | None = None) -> pl.DataFrame:
        """Optimise (SpatialOptimizer) and execute (SpatialExecutor) the plan.

        A plan ending in a large-probe spatial join streams the probe in morsels and
        concatenates, bounding the intermediate. Indexing follows the frame's mode.

        Args:
            batch_size: Probe rows per morsel for streamed joins. Defaults to
                MORSEL_ROWS. Ignored for plans without a join.

        Returns:
            The executed result as a Polars DataFrame.
        """
        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        optimized = optimizer.optimize(self._plan, self._sf.engine)
        plugin_path = optimizer._select_plugin_path(optimized, self._sf.engine)
        return executor.execute(optimized, self._sf, plugin_path, batch_size)

    def collect_batched(self, batch_size: int | None = None) -> Iterator[pl.DataFrame]:
        """Execute the plan and yield the result one morsel-frame at a time.

        A join plan yields the result one joined morsel at a time so the full result never
        materialises. Plans without a join yield one frame.

        Args:
            batch_size: Probe rows per morsel. Defaults to MORSEL_ROWS.

        Returns:
            An iterator of DataFrames, one per probe morsel.
        """
        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        optimized = optimizer.optimize(self._plan, self._sf.engine)
        return executor.stream(optimized, self._sf, batch_size)

    def sink_parquet(self, path: str | Path, batch_size: int | None = None) -> None:
        """Execute the plan and stream its result to a Parquet file in bounded memory.

        Args:
            path: Destination Parquet file path.
            batch_size: Probe rows per morsel. Defaults to MORSEL_ROWS.
        """
        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        optimized = optimizer.optimize(self._plan, self._sf.engine)
        writer: pq.ParquetWriter | None = None
        try:
            for morsel in executor.stream(optimized, self._sf, batch_size):
                table = morsel.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(str(path), table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    def lazy_source(self, batch_size: int | None = None) -> pl.LazyFrame:
        """Expose the plan's streamed output as a native Polars LazyFrame source.

        The plan runs morsel by morsel as a Polars IO source, so downstream ops (sort,
        sink_parquet) fuse with the join into one out-of-core pipeline. A one-row probe runs first.

        Args:
            batch_size: Probe rows per morsel. Defaults to MORSEL_ROWS.

        Returns:
            A Polars LazyFrame that streams this plan's output.
        """
        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        optimized = optimizer.optimize(self._plan, self._sf.engine)

        sample = next(executor.stream(optimized, self._sf, batch_size=1), None)
        schema = sample.schema if sample is not None else pl.Schema({})

        def source(with_columns, predicate, n_rows, batch_size_hint):
            # Stream plan morsels, applying Polars predicate, projection, and row-count pushdown
            produced = 0
            for morsel in executor.stream(optimized, self._sf, batch_size):
                if predicate is not None:
                    morsel = morsel.filter(predicate)
                if with_columns is not None:
                    morsel = morsel.select(with_columns)
                if n_rows is not None and produced + morsel.height > n_rows:
                    morsel = morsel.head(n_rows - produced)
                produced += morsel.height
                yield morsel
                if n_rows is not None and produced >= n_rows:
                    break

        return register_io_source(source, schema=schema)

    @staticmethod
    def collect_all(frames: list[SpatialLazyFrame]) -> list[pl.DataFrame]:
        """Collect multiple SpatialLazyFrames, caching any shared plan prefix.

        Caches the plan prefix shared by frames branched from the same base, emitting it
        once and building each branch's suffix from it.

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

        # Optimize the shared prefix as a standalone plan and cache its Polars chain
        prefix_plan = plans[0][:prefix_len]
        optimized_prefix = optimizer.optimize(prefix_plan, sf.engine)
        base_lf = sf.df.with_row_index(_ROW_IDX).lazy()
        cached_lf = executor._emit_chain(optimized_prefix, sf, base_lf, PluginPath.EXPR).cache()

        # Build each branch's suffix chain starting from the cached result
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


class SpatialGroupBy:
    """Pending grouped aggregation over a SpatialLazyFrame. Created by .group_by().

    Args:
        slf: The SpatialLazyFrame to aggregate.
        keys: Group-by key columns.
    """

    def __init__(self, slf: SpatialLazyFrame, keys: list[str]) -> None:
        self._slf = slf
        self._keys = keys

    def agg(self, **named_aggs: AggSpec) -> pl.DataFrame:
        """Run the grouped aggregation, reducing each join morsel into per-group partials.

        Args:
            named_aggs: Output column name to aggregation spec (pycanopy.agg.count, sum, etc).

        Returns:
            One row per group with the named aggregate columns.

        Raises:
            ValueError: If no aggregations are given.
        """
        if not named_aggs:
            raise ValueError("agg requires at least one aggregation")
        keep = list(
            dict.fromkeys([*self._keys, *(c for spec in named_aggs.values() for c in spec.inputs)])
        )
        projected = self._slf.select(keep)
        partials = [_partial_agg(m, self._keys, named_aggs) for m in projected.collect_batched()]
        if not partials:
            partials = [_partial_agg(projected.collect(), self._keys, named_aggs)]
        return _reduce_partials(partials, self._keys, named_aggs)
