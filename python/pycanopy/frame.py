"""SpatialFrame — entry point for spatial query planning."""

from __future__ import annotations

import polars as pl

from pycanopy.engine import Engine
from pycanopy.lazy import SpatialLazyFrame


class SpatialFrame:
    """Owns a materialized DataFrame, a spatial index Engine, and cached column stats.

    All spatial query planning begins with .lazy(). The DataFrame must be fully
    materialized before construction — the Engine is built from the coordinate
    columns at this point, making dataset statistics available to the optimizer.

    Args:
        df: Materialized Polars DataFrame.
        x_col: Name of the column holding x (longitude/easting) coordinates.
        y_col: Name of the column holding y (latitude/northing) coordinates.
    """

    def __init__(self, df: pl.DataFrame, x_col: str, y_col: str) -> None:
        if x_col not in df.columns:
            raise ValueError(f"x_col {x_col!r} not found in DataFrame")
        if y_col not in df.columns:
            raise ValueError(f"y_col {y_col!r} not found in DataFrame")
        self._df = df
        self._x_col = x_col
        self._y_col = y_col
        self._engine = Engine.from_coords(
            df[x_col].to_numpy(),
            df[y_col].to_numpy(),
        )

    @classmethod
    def from_polygons(
        cls,
        df: pl.DataFrame,
        geometry_col: str,
        x_col: str = "_x",
        y_col: str = "_y",
    ) -> SpatialFrame:
        """Construct from a DataFrame containing a shapely/GeoArrow geometry column.

        Args:
            df: Materialized Polars DataFrame with a geometry column.
            geometry_col: Name of the column holding shapely Polygon geometries.
            x_col: Internal column name for extracted x coordinates.
            y_col: Internal column name for extracted y coordinates.

        Returns:
            SpatialFrame backed by a polygon index.
        """
        if geometry_col not in df.columns:
            raise ValueError(f"geometry_col {geometry_col!r} not found in DataFrame")
        geometries = df[geometry_col].to_list()
        engine = Engine.from_polygons(geometries)
        sf = object.__new__(cls)
        sf._df = df
        sf._x_col = x_col
        sf._y_col = y_col
        sf._engine = engine
        return sf

    def lazy(self) -> SpatialLazyFrame:
        """Return a SpatialLazyFrame for declarative plan construction."""
        return SpatialLazyFrame(self, [])

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def x_col(self) -> str:
        return self._x_col

    @property
    def y_col(self) -> str:
        return self._y_col
