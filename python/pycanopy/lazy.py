"""SpatialLazyFrame — immutable plan builder. No execution until .collect()."""

from __future__ import annotations

import polars as pl

from pycanopy.executor import SpatialExecutor
from pycanopy.nodes import (
    ContainsNode,
    KnnJoinNode,
    KnnNode,
    Plan,
    RangeNode,
    ScalarNode,
    WithinJoinNode,
)
from pycanopy.optimizer import SpatialOptimizer


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

    def collect(self) -> pl.DataFrame:
        """Optimise and execute the plan. Returns a Polars DataFrame.

        Triggers:
          1. SpatialOptimizer: selectivity estimation, cost-based sort, fusion pass.
          2. SpatialExecutor: emits the optimised plan as a Polars LazyFrame chain.
        """
        optimizer = SpatialOptimizer()
        executor = SpatialExecutor()
        optimized = optimizer.optimize(self._plan, self._sf.engine)
        return executor.execute(optimized, self._sf)
