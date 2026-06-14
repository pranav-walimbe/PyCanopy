"""Plan node types for the SpatialOptimizer.

Declaration order is not execution order. The optimizer reorders nodes by cost
using selectivity estimates before emitting the final Polars chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import polars as pl


class PluginPath(Enum):
    EXPR = auto()  # expression plugin — default path
    IO = auto()  # IO plugin — wide DataFrame + high selectivity


@dataclass
class ScalarNode:
    """A Polars expression filter. Evaluated by Polars, not the spatial engine."""

    expr: pl.Expr
    selectivity: float = 1.0
    cost: int = 0


@dataclass
class RangeNode:
    """Bounding-box spatial filter."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    selectivity: float = 1.0


@dataclass
class ContainsNode:
    """Point-in-polygon filter (polygon dataset only)."""

    qx: float
    qy: float
    selectivity: float = 1.0


@dataclass
class KnnNode:
    """k-nearest-neighbour lookup. Returns matching row indices, not a boolean mask."""

    qx: float
    qy: float
    k: int
    approximate: bool = False
    selectivity: float = 1.0


@dataclass
class FusedSpatialNode:
    """Two or more consecutive spatial predicates merged into a single index build.

    All predicates must select the same index type and none may have selectivity
    below the fusion floor (see SpatialOptimizer._FUSION_SELECTIVITY_FLOOR).
    """

    predicates: list[RangeNode | ContainsNode]


@dataclass
class KnnJoinNode:
    """Spatial join: for each row in query_df find k nearest in Engine's dataset.

    Acts as a barrier in the plan — no nodes are reordered past it.
    Result columns: all query_df columns followed by all Engine df columns
    (conflicting names in the right side are prefixed with 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    k: int
    approximate: bool = False


@dataclass
class WithinJoinNode:
    """Spatial join: for each point in query_df find which Engine polygons contain it.

    Acts as a barrier in the plan. Engine must be a polygon dataset.
    Result columns: all query_df columns followed by all Engine df columns
    (conflicting names in the right side are prefixed with 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    flip: bool = False


@dataclass
class WithinDistanceJoinNode:
    """Spatial join: for each point in query_df find Engine points within `distance`.

    Acts as a barrier in the plan. Engine must be a point dataset.
    Result columns: all query_df columns followed by all Engine df columns
    (conflicting names in the right side are prefixed with 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    distance: float
    flip: bool = False


@dataclass
class PolygonWithinDistanceJoinNode:
    """Spatial join: for each point in query_df find Engine polygons within `distance`.

    Acts as a barrier in the plan. Engine must be a polygon dataset. Distance is
    measured to the polygon boundary (zero when the point is inside).
    Result columns: all query_df columns followed by all Engine df columns
    (conflicting names in the right side are prefixed with 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    distance: float


@dataclass
class PointsWithinDistanceOfPolygonNode:
    """Spatial filter: keep points within `distance` of a single query polygon.

    Point datasets only. Distance is to the polygon boundary (zero inside). Returns
    a subset of the frame's rows, so it behaves like range / contains.
    """

    polygon: object  # shapely Polygon (interior holes supported)
    distance: float
    selectivity: float = 1.0


@dataclass
class IntersectsSelfJoinNode:
    """Spatial self-join: all intersecting polygon pairs with overlap area and IoU.

    Polygon datasets only. Acts as a terminal barrier — it produces a pair frame
    (left, right, area_left, area_right, overlap_area, iou), not a row subset.
    """


@dataclass
class PolygonKnnJoinNode:
    """Spatial join: for each point in query_df find its k nearest Engine polygons.

    Acts as a barrier in the plan. Engine must be a polygon dataset. Ranking is by
    exact point-to-polygon distance. A 'distance_to_polygon' column is appended.
    Result columns: all query_df columns followed by all Engine df columns
    (conflicting names in the right side are prefixed with 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    k: int


# Type alias for a complete plan
Plan = list[
    ScalarNode
    | RangeNode
    | ContainsNode
    | KnnNode
    | FusedSpatialNode
    | KnnJoinNode
    | WithinJoinNode
    | WithinDistanceJoinNode
    | PolygonWithinDistanceJoinNode
    | PolygonKnnJoinNode
    | PointsWithinDistanceOfPolygonNode
    | IntersectsSelfJoinNode
]
