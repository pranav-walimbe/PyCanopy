"""Plan node types for the SpatialOptimizer.

Declaration order is not execution order. The optimizer reorders nodes by cost
using selectivity estimates before emitting the final Polars chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import polars as pl


class PluginPath(Enum):
    EXPR = auto()  # expression plugin, default path
    IO = auto()  # IO plugin, wide DataFrame plus high selectivity


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
    selectivity: float = 1.0


@dataclass
class FusedSpatialNode:
    """Two or more consecutive spatial predicates merged into a single index build.

    All predicates must select the same index type and none may have selectivity
    below the fusion floor (see SpatialOptimizer._FUSION_SELECTIVITY_FLOOR).
    """

    predicates: list[RangeNode | ContainsNode]


@dataclass
class SelectNode:
    """Terminal projection restricting the collected output to these columns.

    Pushed into a preceding join's gather (as keep_columns) so unused columns are
    never materialized. Must be the last node in a plan.
    """

    columns: tuple[str, ...]


@dataclass
class KnnJoinNode:
    """Spatial join: for each row in query_df find k nearest in Engine's dataset.

    Acts as a barrier in the plan. Output is query_df columns then Engine df columns
    (conflicting right-side names prefixed 'right_').

    keep_columns, when set by the optimizer from a trailing SelectNode, are the output
    column names the gather must retain (projection plus any post-join filter inputs).
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    k: int
    keep_columns: tuple[str, ...] | None = None


@dataclass
class WithinJoinNode:
    """Spatial join: for each point in query_df find which Engine polygons contain it.

    Acts as a barrier on a polygon dataset. Output is query_df columns then Engine df
    columns (conflicting right-side names prefixed 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    flip: bool = False
    keep_columns: tuple[str, ...] | None = None


@dataclass
class WithinDistanceJoinNode:
    """Spatial join: for each point in query_df find Engine points within `distance`.

    Acts as a barrier on a point dataset. Output is query_df columns then Engine df
    columns (conflicting right-side names prefixed 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    distance: float
    flip: bool = False
    keep_columns: tuple[str, ...] | None = None


@dataclass
class PolygonWithinDistanceJoinNode:
    """Spatial join: for each point in query_df find Engine polygons within `distance`.

    Acts as a barrier on a polygon dataset. Distance is to the polygon boundary (zero
    inside). Output is query_df columns then Engine df columns (conflicts prefixed 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    distance: float
    keep_columns: tuple[str, ...] | None = None


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

    Polygon datasets only. A terminal barrier that produces a pair frame (left, right,
    area_left, area_right, overlap_area, iou) rather than a row subset.
    """


@dataclass
class PolygonKnnJoinNode:
    """Spatial join: for each point in query_df find its k nearest Engine polygons.

    Acts as a barrier on a polygon dataset. Ranks by exact point-to-polygon distance and
    appends a 'distance_to_polygon' column. Output is query_df then Engine df columns
    (conflicts prefixed 'right_').
    """

    query_df: pl.DataFrame
    x_col: str
    y_col: str
    k: int
    keep_columns: tuple[str, ...] | None = None


# Type alias for a complete plan. Union (not X | Y) because this alias is evaluated at
# runtime, and the | operator on types needs Python 3.10 while the floor is 3.9.
Plan = list[
    Union[
        ScalarNode,
        RangeNode,
        ContainsNode,
        KnnNode,
        FusedSpatialNode,
        KnnJoinNode,
        WithinJoinNode,
        WithinDistanceJoinNode,
        PolygonWithinDistanceJoinNode,
        PolygonKnnJoinNode,
        PointsWithinDistanceOfPolygonNode,
        IntersectsSelfJoinNode,
        SelectNode,
    ]
]
