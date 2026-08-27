"""
Define SpatialLazyFrame, an immutable plan builder that does not execute until .collect().
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from polars.io.plugins import register_io_source

from pycanopy.agg import AggSpec, _partial_agg, _reduce_partials, _try_fused_join_agg
from pycanopy.engine import (
    _extract_query_polygon_rings,
    _points_within_distance_of_polygon_scan,
    distance_to_point,
    wkb_points_to_xy,
)
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
        rows = "?" if isinstance(node.query_df, pl.LazyFrame) else f"{len(node.query_df):,}"
        return f"WITHIN_JOIN [query_rows={rows}, barrier{flip}]"
    if isinstance(node, WithinDistanceJoinNode):
        flip = ", flip" if node.flip else ""
        return f"WITHIN_DIST_JOIN [dist={node.distance:.4g}, query_rows={len(node.query_df):,}, barrier{flip}]"
    if isinstance(node, PolygonWithinDistanceJoinNode):
        rows = "?" if isinstance(node.query_df, pl.LazyFrame) else f"{len(node.query_df):,}"
        return f"POLY_WITHIN_DIST_JOIN [dist={node.distance:.4g}, query_rows={rows}, barrier]"
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


def _fmt_plan(
    plan: Plan,
    path: PluginPath | None,
    n: int | None,
    source_name: str = "DF",
) -> str:
    # Format the full plan as Polars-style indented explain() text
    path_suffix = ""
    if path is not None:
        path_label = "EXPR" if path == PluginPath.EXPR else "IO"
        path_suffix = f"; path: {path_label}"
    row_count = "?" if n is None else f"{n:,}"
    df_line = f"{source_name} [N={row_count}{path_suffix}]"

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


_SOURCE_JOIN_TYPES = (
    KnnJoinNode,
    WithinJoinNode,
    WithinDistanceJoinNode,
    PolygonWithinDistanceJoinNode,
    PolygonKnnJoinNode,
)

_PUSHDOWN_BINARY_OPS = {
    "And",
    "Eq",
    "EqValidity",
    "Gt",
    "GtEq",
    "Lt",
    "LtEq",
    "NotEq",
    "NotEqValidity",
    "Or",
}
_PUSHDOWN_BOOLEAN_FUNCTIONS = {"IsBetween", "IsIn", "IsNotNull", "IsNull", "Not"}


def _is_pushdown_ast(node: object) -> bool:
    # Accept a small, deterministic subset of the serialized Polars expression tree
    if not isinstance(node, dict) or len(node) != 1:
        return False
    kind, value = next(iter(node.items()))
    if kind in {"Column", "Literal"}:
        return True
    if kind == "BinaryExpr" and isinstance(value, dict):
        return (
            value.get("op") in _PUSHDOWN_BINARY_OPS
            and _is_pushdown_ast(value.get("left"))
            and _is_pushdown_ast(value.get("right"))
        )
    if kind == "Cast" and isinstance(value, dict):
        return _is_pushdown_ast(value.get("expr"))
    if kind != "Function" or not isinstance(value, dict):
        return False
    function = value.get("function")
    if not isinstance(function, dict) or "Boolean" not in function:
        return False
    boolean = function["Boolean"]
    if isinstance(boolean, str):
        name = boolean
    elif isinstance(boolean, dict):
        name = next(iter(boolean), "")
    else:
        return False
    return name in _PUSHDOWN_BOOLEAN_FUNCTIONS and all(
        _is_pushdown_ast(item) for item in value.get("input", [])
    )


def _is_pushdown_expr(expr: pl.Expr, source_columns: set[str], geometry_col: str) -> bool:
    # Reject derived, geometry, and unsupported expressions; failure simply disables pushdown
    try:
        roots = set(expr.meta.root_names())
        if geometry_col in roots or not roots.issubset(source_columns):
            return False
        tree = json.loads(expr.meta.serialize(format="json"))
    except Exception:
        return False
    return _is_pushdown_ast(tree)


def _source_filter_prefix(
    plan: Plan,
    schema: pl.Schema,
    geometry_col: str,
) -> tuple[list[pl.Expr], Plan]:
    # Remove only the contiguous safe-filter prefix before any spatial or row-changing node
    source_columns = set(schema.names())
    count = 0
    filters = []
    for node in plan:
        if not isinstance(node, ScalarNode) or not _is_pushdown_expr(
            node.expr, source_columns, geometry_col
        ):
            break
        filters.append(node.expr)
        count += 1
    return filters, plan[count:]


def _source_columns_for_plan(plan: Plan, schema: pl.Schema) -> set[str] | None:
    # Return source columns needed during execution, or None when all columns are output
    if not plan or not isinstance(plan[-1], SelectNode):
        if plan and isinstance(plan[-1], IntersectsSelfJoinNode):
            return set()
        return None

    body = plan[:-1]
    join_position = next(
        (position for position, node in enumerate(body) if isinstance(node, _SOURCE_JOIN_TYPES)),
        None,
    )
    if join_position is None:
        query_columns = set()
    else:
        query = body[join_position].query_df
        query_schema = query.collect_schema() if isinstance(query, pl.LazyFrame) else query.schema
        query_columns = set(query_schema.names())
    source_columns = set(schema.names())

    def source_name(output_name: str) -> str | None:
        # Map a post-join output name back to its source-side column
        if output_name.startswith("right_"):
            candidate = output_name.removeprefix("right_")
            if candidate in query_columns and candidate in source_columns:
                return candidate
            return None
        if output_name in source_columns and output_name not in query_columns:
            return output_name
        return None

    required = set()
    for output in plan[-1].columns:
        source = source_name(output) if join_position is not None else output
        if source in source_columns:
            required.add(source)
    for position, node in enumerate(body):
        if not isinstance(node, ScalarNode):
            continue
        roots = node.expr.meta.root_names()
        if join_position is None or position < join_position:
            required.update(root for root in roots if root in source_columns)
        else:
            required.update(source for root in roots if (source := source_name(root)) is not None)
    return required


@dataclass(frozen=True)
class _StreamingPointFilter:
    filters: tuple[pl.Expr, ...]
    spatial: RangeNode | WithinDistanceOfPointNode | PointsWithinDistanceOfPolygonNode
    scan_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    schema: pl.Schema


def _streaming_point_filter_plan(sf, plan: Plan, schema: pl.Schema) -> _StreamingPointFilter | None:
    # Recognize one-shot point filters that are independent across source batches
    source = sf._lazy_source
    if (
        source is None
        or source.geometry_kind != "point"
        or source.index_mode not in ("auto", "none")
        or source.geometry_col not in schema
        or schema[source.geometry_col] != pl.Binary
    ):
        return None
    projection = plan[-1] if plan and isinstance(plan[-1], SelectNode) else None
    body = plan[:-1] if projection is not None else plan
    spatial_types = (RangeNode, WithinDistanceOfPointNode, PointsWithinDistanceOfPolygonNode)
    spatials = [node for node in body if isinstance(node, spatial_types)]
    if len(spatials) != 1 or any(
        not isinstance(node, (ScalarNode, *spatial_types)) for node in body
    ):
        return None
    if (
        isinstance(spatials[0], (WithinDistanceOfPointNode, PointsWithinDistanceOfPolygonNode))
        and spatials[0].distance < 0
    ):
        return None
    source_columns = set(schema.names())
    scalars = [node for node in body if isinstance(node, ScalarNode)]
    if not all(
        _is_pushdown_expr(node.expr, source_columns, source.geometry_col) for node in scalars
    ):
        return None

    if projection is None:
        output_columns = tuple(schema.names())
    else:
        output_columns = projection.columns
        if any(column not in source_columns for column in output_columns):
            return None
    required = _source_columns_for_plan(plan, schema)
    retained = source_columns if required is None else required
    scan_columns = tuple(
        name for name in schema.names() if name in retained or name == source.geometry_col
    )
    return _StreamingPointFilter(
        tuple(node.expr for node in scalars),
        spatials[0],
        scan_columns,
        output_columns,
        schema,
    )


def _streaming_point_mask(
    node: RangeNode | WithinDistanceOfPointNode,
    xs: np.ndarray,
    ys: np.ndarray,
    coordinate_system: str,
) -> np.ndarray:
    # Evaluate one supported point predicate while preserving inclusive engine boundaries
    if isinstance(node, RangeNode):
        return (xs >= node.min_x) & (xs <= node.max_x) & (ys >= node.min_y) & (ys <= node.max_y)
    if coordinate_system == "geographic":
        return distance_to_point(xs, ys, node.cx, node.cy, "geographic") <= node.distance
    dx = xs - node.cx
    dy = ys - node.cy
    return dx * dx + dy * dy <= node.distance * node.distance


def _stream_deferred_point_filter(sf, strategy: _StreamingPointFilter) -> Iterator[pl.DataFrame]:
    # Decode and filter one source batch at a time without retaining reusable engine state
    source = sf._lazy_source
    source_frame = source.frame
    polygon_rings = (
        _extract_query_polygon_rings(strategy.spatial.polygon)
        if isinstance(strategy.spatial, PointsWithinDistanceOfPolygonNode)
        else None
    )
    for expr in strategy.filters:
        source_frame = source_frame.filter(expr)
    yielded = False
    for batch in source_frame.select(strategy.scan_columns).collect_batches(
        chunk_size=source.ingest_batch_size,
        maintain_order=True,
    ):
        xs, ys = wkb_points_to_xy(batch[source.geometry_col])
        if polygon_rings is not None:
            indices = _points_within_distance_of_polygon_scan(
                xs,
                ys,
                polygon_rings,
                strategy.spatial.distance,
            )
            matched = batch[indices].select(strategy.output_columns)
        else:
            mask = _streaming_point_mask(strategy.spatial, xs, ys, source.coordinate_system)
            matched = batch.filter(pl.Series(mask)).select(strategy.output_columns)
        if matched.height:
            yielded = True
            yield matched
    if not yielded:
        yield pl.DataFrame(schema=strategy.schema).select(strategy.output_columns)


def _deferred_point_lazy_source(sf, batch_size: int | None) -> pl.LazyFrame:
    # Expose decoded point batches without materializing reusable engine state
    source = sf._lazy_source
    chunk_size = source.ingest_batch_size if batch_size is None else batch_size
    source_schema = sf._lazy_schema()
    output_schema = pl.Schema(
        {
            **dict(source_schema),
            sf._x_col: pl.Float64,
            sf._y_col: pl.Float64,
        }
    )

    def batches(with_columns, predicate, n_rows, batch_size_hint):
        # Decode only when projected coordinates or predicates need them
        requested = output_schema.names() if with_columns is None else list(with_columns)
        predicate_columns = set() if predicate is None else set(predicate.meta.root_names())
        coordinate_columns = {sf._x_col, sf._y_col}
        decode = bool(coordinate_columns & (set(requested) | predicate_columns))
        required = set(requested) | predicate_columns
        scan_columns = [column for column in source_schema.names() if column in required]
        if decode and source.geometry_col not in scan_columns:
            scan_columns.append(source.geometry_col)
        produced = 0
        yielded = False
        for batch in source.frame.select(scan_columns).collect_batches(
            chunk_size=chunk_size,
            maintain_order=True,
        ):
            if decode:
                xs, ys = wkb_points_to_xy(batch[source.geometry_col])
                batch = batch.with_columns(pl.Series(sf._x_col, xs), pl.Series(sf._y_col, ys))
            if predicate is not None:
                batch = batch.filter(predicate)
            if with_columns is not None:
                batch = batch.select(with_columns)
            if n_rows is not None and produced + batch.height > n_rows:
                batch = batch.head(n_rows - produced)
            if batch.height:
                yielded = True
                produced += batch.height
                yield batch
            if n_rows is not None and produced >= n_rows:
                break
        if not yielded:
            schema = (
                output_schema
                if with_columns is None
                else pl.Schema({column: output_schema[column] for column in with_columns})
            )
            yield pl.DataFrame(schema=schema)

    return register_io_source(batches, schema=output_schema)


class SpatialLazyFrame:
    """Builds a spatial query plan declaratively. Declaration order is not execution order.

    All methods return a new SpatialLazyFrame with the node appended without mutation.
    Join and kNN nodes act as barriers and are never reordered by the cost sort.

    Args:
        sf: SpatialFrame owning materialized data or a deferred source.
        plan: Current list of plan nodes (do not mutate directly).
    """

    def __init__(self, sf: SpatialFrame, plan: Plan) -> None:  # noqa: F821
        self._sf = sf
        self._plan = plan

    def _prepare_plan(self) -> tuple[SpatialFrame, Plan]:  # noqa: F821
        # Fold a safe leading filter prefix into a deferred source before materialization
        if not self._sf._is_deferred:
            return self._sf, self._plan
        schema = self._sf._lazy_schema()
        source = self._sf._lazy_source
        filters, plan = _source_filter_prefix(self._plan, schema, source.geometry_col)
        required = _source_columns_for_plan(plan, schema)
        return self._sf._materialize_lazy(required, schema, filters), plan

    def _prepare(self) -> SpatialFrame:  # noqa: F821
        # Compatibility helper for callers that need only the prepared frame
        return self._prepare_plan()[0]

    def _streaming_filter_plan(self) -> _StreamingPointFilter | None:
        # Resolve a deferred streaming strategy without materializing its geometry
        if not self._sf._is_deferred:
            return None
        schema = self._sf._lazy_schema()
        return _streaming_point_filter_plan(self._sf, self._plan, schema)

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
        query_df: pl.DataFrame | pl.LazyFrame,
        x_col: str,
        y_col: str,
    ) -> SpatialLazyFrame:
        """Spatial join: for each point in query_df find the Engine polygons that contain it.

        Engine must be a polygon dataset. Result columns are query_df's then the Engine df's
        (conflicting right-side columns are prefixed with 'right_').

        Args:
            query_df: Eager or lazy frame of query points. Lazy input is consumed in batches.
            x_col: Column in query_df holding x coordinates.
            y_col: Column in query_df holding y coordinates.

        Returns:
            New SpatialLazyFrame with the within join node appended.
        """
        if not isinstance(query_df, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError("query_df must be a polars DataFrame or LazyFrame")
        schema = (
            query_df.collect_schema() if isinstance(query_df, pl.LazyFrame) else query_df.schema
        )
        if x_col not in schema:
            raise ValueError(f"x_col {x_col!r} not found in query_df")
        if y_col not in schema:
            raise ValueError(f"y_col {y_col!r} not found in query_df")
        return SpatialLazyFrame(
            self._sf,
            [*self._plan, WithinJoinNode(query_df, x_col, y_col)],
        )

    def polygon_within_distance_join(
        self,
        query_df: pl.DataFrame | pl.LazyFrame,
        x_col: str,
        y_col: str,
        distance: float,
    ) -> SpatialLazyFrame:
        """Spatial join: for each point in query_df find Engine polygons within `distance`.

        Distance is to the polygon boundary (zero when the point is inside). Result columns
        are query_df's then the Engine df's (conflicting right-side columns prefixed 'right_').

        Args:
            query_df: Eager or lazy frame of query points. Lazy input is consumed in batches.
            x_col: Column in query_df holding x coordinates.
            y_col: Column in query_df holding y coordinates.
            distance: Maximum Euclidean point-to-polygon distance for a match.

        Returns:
            New SpatialLazyFrame with the polygon within-distance join node appended.
        """
        if not isinstance(query_df, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError("query_df must be a polars DataFrame or LazyFrame")
        schema = (
            query_df.collect_schema() if isinstance(query_df, pl.LazyFrame) else query_df.schema
        )
        if x_col not in schema:
            raise ValueError(f"x_col {x_col!r} not found in query_df")
        if y_col not in schema:
            raise ValueError(f"y_col {y_col!r} not found in query_df")
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
            distance: Maximum Euclidean point-to-polygon distance for a row to be kept.

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
        """Return the physical plan, or the logical plan for a deferred source.

        Returns:
            A human-readable plan description.
        """
        if self._sf._is_deferred:
            return _fmt_plan(self._plan, None, None, "LAZY SOURCE")
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
        if strategy := self._streaming_filter_plan():
            frames = list(_stream_deferred_point_filter(self._sf, strategy))
            return pl.concat(frames, how="vertical", rechunk=False)
        sf, plan = self._prepare_plan()
        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        optimized = optimizer.optimize(plan, sf.engine)
        plugin_path = optimizer._select_plugin_path(optimized, sf.engine)
        return executor.execute(optimized, sf, plugin_path, batch_size)

    def collect_batched(self, batch_size: int | None = None) -> Iterator[pl.DataFrame]:
        """Execute the plan and yield the result one morsel-frame at a time.

        Join plans yield probe morsels. Supported deferred point filters yield source-aligned
        batches without materializing the full spatial frame. Other plans yield one frame.

        Args:
            batch_size: Probe rows per morsel. Defaults to MORSEL_ROWS.

        Returns:
            An iterator of DataFrames, one per probe morsel.
        """
        if strategy := self._streaming_filter_plan():
            return _stream_deferred_point_filter(self._sf, strategy)
        sf, plan = self._prepare_plan()
        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        optimized = optimizer.optimize(plan, sf.engine)
        return executor.stream(optimized, sf, batch_size)

    def sink_parquet(self, path: str | Path, batch_size: int | None = None) -> None:
        """Execute the plan and stream result morsels to a Parquet file.

        Args:
            path: Destination Parquet file path.
            batch_size: Probe rows per morsel. Defaults to MORSEL_ROWS.
        """
        writer: pq.ParquetWriter | None = None
        try:
            for morsel in self.collect_batched(batch_size):
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
        A base deferred point source decodes WKB per batch and exposes its coordinate columns.

        Args:
            batch_size: Probe rows per morsel. Defaults to MORSEL_ROWS.

        Returns:
            A Polars LazyFrame that streams this plan's output.
        """
        source = self._sf._lazy_source
        if not self._plan and source is not None and source.geometry_kind == "point":
            return _deferred_point_lazy_source(self._sf, batch_size)
        sample = next(self.collect_batched(batch_size=1), None)
        schema = sample.schema if sample is not None else pl.Schema({})

        def source(with_columns, predicate, n_rows, batch_size_hint):
            # Stream plan morsels, applying Polars predicate, projection, and row-count pushdown
            produced = 0
            for morsel in self.collect_batched(batch_size):
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
        """Collect frames while caching a shared materialized plan prefix.

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
        if sf._is_deferred:
            return [frame.collect() for frame in frames]

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
        """Run a grouped aggregation over the spatial plan.

        Supported target-side polygon join aggregations accumulate directly in Rust.
        Other plans reduce each join morsel into per-group partials.

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
        prepared, prepared_plan = self._slf.select(keep)._prepare_plan()
        body = prepared_plan[:-1]
        fused = _try_fused_join_agg(prepared, body, self._keys, named_aggs)
        if fused is not None:
            return fused
        projected = SpatialLazyFrame(prepared, body).select(keep)
        partials = [_partial_agg(m, self._keys, named_aggs) for m in projected.collect_batched()]
        if not partials:
            partials = [_partial_agg(projected.collect(), self._keys, named_aggs)]
        return _reduce_partials(partials, self._keys, named_aggs)
