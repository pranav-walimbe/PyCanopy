"""
Define SpatialFrame which is the entry point for spatial query planning.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pycanopy.engine import Engine, wkb_points_to_xy
from pycanopy.lazy import SpatialLazyFrame


class SpatialFrame:
    """Owns a materialized DataFrame, a spatial index Engine, and cached column stats.

    All spatial query planning begins with .lazy(). The DataFrame must be materialized
    before construction since the Engine and its dataset statistics are built here.

    Args:
        df: Materialized Polars DataFrame.
        x_col: Name of the column holding x (longitude/easting) coordinates.
        y_col: Name of the column holding y (latitude/northing) coordinates.
        index_mode: Index build policy fixed for this frame's engine. "auto"
            (default) builds only when the cost model beats a scan, "eager" always
            builds an index, "none" always scans brute-force.
    """

    def __init__(self, df: pl.DataFrame, x_col: str, y_col: str, index_mode: str = "auto") -> None:
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
        self._engine.set_index_mode(index_mode)
        self._wkb_col: str | None = None
        self._wkb_series: pl.Series | None = None

    @classmethod
    def from_wkb_points(
        cls,
        df: pl.DataFrame,
        wkb_col: str,
        x_col: str = "_x",
        y_col: str = "_y",
        index_mode: str = "auto",
    ) -> SpatialFrame:
        """Construct a point SpatialFrame from a WKB point column of ``df``.

        The WKB points are decoded (vectorised for standard 2D LE points) and appended as
        ``x_col`` / ``y_col`` before the index is built.

        Args:
            df: Materialized Polars DataFrame with a WKB point column.
            wkb_col: Name of the Binary column holding WKB point geometries.
            x_col: Internal column name for the extracted x coordinates.
            y_col: Internal column name for the extracted y coordinates.
            index_mode: Index build policy ("eager" / "none" / "auto").

        Returns:
            SpatialFrame backed by a point index.
        """
        if wkb_col not in df.columns:
            raise ValueError(f"wkb_col {wkb_col!r} not found in DataFrame")
        xs, ys = wkb_points_to_xy(df[wkb_col])
        enriched = df.with_columns(pl.Series(x_col, xs), pl.Series(y_col, ys))
        return cls(enriched, x_col=x_col, y_col=y_col, index_mode=index_mode)

    @classmethod
    def from_polygons(
        cls,
        df: pl.DataFrame,
        geometry_col: str,
        x_col: str = "_x",
        y_col: str = "_y",
        index_mode: str = "auto",
    ) -> SpatialFrame:
        """Construct from a DataFrame containing a shapely/GeoArrow geometry column.

        Args:
            df: Materialized Polars DataFrame with a geometry column.
            geometry_col: Name of the column holding shapely Polygon geometries.
            x_col: Internal column name for extracted x coordinates.
            y_col: Internal column name for extracted y coordinates.
            index_mode: Index build policy ("eager" / "none" / "auto").

        Returns:
            SpatialFrame backed by a polygon index.
        """
        if geometry_col not in df.columns:
            raise ValueError(f"geometry_col {geometry_col!r} not found in DataFrame")
        geometries = df[geometry_col].to_list()
        engine = Engine.from_polygons(geometries)
        engine.set_index_mode(index_mode)
        sf = object.__new__(cls)
        sf._df = df
        sf._x_col = x_col
        sf._y_col = y_col
        sf._engine = engine
        return sf

    @classmethod
    def from_wkb_polygons(
        cls,
        df: pl.DataFrame,
        wkb_col: str,
        x_col: str = "_x",
        y_col: str = "_y",
        index_mode: str = "auto",
    ) -> SpatialFrame:
        """Construct a polygon SpatialFrame from a WKB polygon column of ``df``.

        The WKB Polygon / MultiPolygon bytes are decoded directly in Rust, and the raw
        WKB column is dropped from the retained DataFrame once the index is built.

        Args:
            df: Materialized Polars DataFrame with a WKB polygon column.
            wkb_col: Name of the Binary column holding WKB polygon geometries.
            x_col: Internal column name placeholder (unused for polygon frames).
            y_col: Internal column name placeholder (unused for polygon frames).
            index_mode: Index build policy ("eager" / "none" / "auto").

        Returns:
            SpatialFrame backed by a polygon index.
        """
        if wkb_col not in df.columns:
            raise ValueError(f"wkb_col {wkb_col!r} not found in DataFrame")
        engine = Engine.from_wkb_polygons(df[wkb_col])
        engine.set_index_mode(index_mode)
        sf = object.__new__(cls)
        sf._df = df.drop(wkb_col)
        sf._x_col = x_col
        sf._y_col = y_col
        sf._engine = engine
        sf._wkb_col = wkb_col
        sf._wkb_series = df[wkb_col]
        return sf

    def lazy(self) -> SpatialLazyFrame:
        """Start a declarative spatial query plan over this frame.

        Returns:
            A SpatialLazyFrame for declarative plan construction.
        """
        return SpatialLazyFrame(self, [])

    def range_filter(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> SpatialFrame:
        """Return a new SpatialFrame containing only geometries that intersect the bounding box.

        Args:
            min_x: Left edge of the query rectangle.
            min_y: Bottom edge of the query rectangle.
            max_x: Right edge of the query rectangle.
            max_y: Top edge of the query rectangle.

        Returns:
            New SpatialFrame with the matching rows and a freshly built index.
        """
        indices = self._engine.range_query(min_x, min_y, max_x, max_y)
        if not indices:
            if self._wkb_col is not None:
                empty = self._df.clear().with_columns(pl.Series(self._wkb_col, [], dtype=pl.Binary))
                return SpatialFrame.from_wkb_polygons(
                    empty, self._wkb_col, self._x_col, self._y_col
                )
            return SpatialFrame(self._df.clear(), self._x_col, self._y_col)
        idx_s = pl.Series(np.asarray(indices, dtype=np.uint32))
        if self._wkb_col is not None:
            filtered = self._df[idx_s].with_columns(self._wkb_series[idx_s].alias(self._wkb_col))
            return SpatialFrame.from_wkb_polygons(filtered, self._wkb_col, self._x_col, self._y_col)
        return SpatialFrame(self._df[idx_s], self._x_col, self._y_col)

    # Geometry aggregations and transforms (polygon datasets)

    def polygon_areas(self) -> pl.DataFrame:
        """Append an unsigned 'area' column to this frame's DataFrame (polygon datasets).

        Returns:
            The frame's DataFrame with an appended unsigned 'area' column.
        """
        areas = self._engine.polygon_areas()
        return self._df.with_columns(pl.Series("area", areas))

    def intersects_pairs(self, key_col: str | None = None) -> pl.DataFrame:
        """Return intersecting polygon pairs (i < j) with overlap area and IoU (polygon datasets).

        Args:
            key_col: Optional column name whose values replace the positional left/right indices.
                When provided, output columns are named ``{key_col}_1`` and ``{key_col}_2``,
                and each pair is canonicalized so the smaller key value appears in ``_1``.

        Returns:
            DataFrame with columns left/right (or key_1/key_2 if key_col given),
            area_left, area_right, overlap_area, iou. Empty with correct schema when none intersect.
        """
        flat = self._engine.polygon_intersects_self_join()
        if len(flat) == 0:
            if key_col is not None:
                dtype = self._df[key_col].dtype
                return pl.DataFrame(
                    schema={
                        f"{key_col}_1": dtype,
                        f"{key_col}_2": dtype,
                        "area_left": pl.Float64,
                        "area_right": pl.Float64,
                        "overlap_area": pl.Float64,
                        "iou": pl.Float64,
                    }
                )
            return pl.DataFrame(
                schema={
                    "left": pl.UInt32,
                    "right": pl.UInt32,
                    "area_left": pl.Float64,
                    "area_right": pl.Float64,
                    "overlap_area": pl.Float64,
                    "iou": pl.Float64,
                }
            )

        pairs = flat.reshape(-1, 2)
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        areas = self._engine.polygon_areas()
        overlap = self._engine.polygon_pairs_intersection_area(i_idx, j_idx)
        area_i = areas[i_idx]
        area_j = areas[j_idx]
        union = area_i + area_j - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0.0)

        if key_col is not None:
            keys = self._df[key_col].to_numpy()
            k1 = keys[i_idx]
            k2 = keys[j_idx]
            swap = k1 > k2
            return pl.DataFrame(
                {
                    f"{key_col}_1": np.where(swap, k2, k1),
                    f"{key_col}_2": np.where(swap, k1, k2),
                    "area_left": area_i,
                    "area_right": area_j,
                    "overlap_area": overlap,
                    "iou": iou,
                }
            )

        return pl.DataFrame(
            {
                "left": i_idx.astype(np.uint32),
                "right": j_idx.astype(np.uint32),
                "area_left": area_i,
                "area_right": area_j,
                "overlap_area": overlap,
                "iou": iou,
            },
            schema={
                "left": pl.UInt32,
                "right": pl.UInt32,
                "area_left": pl.Float64,
                "area_right": pl.Float64,
                "overlap_area": pl.Float64,
                "iou": pl.Float64,
            },
        )

    def radius_query(self, cx: float, cy: float, distance: float) -> pl.DataFrame:
        """Return the rows whose point lies within `distance` of the center (cx, cy).

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            distance: Maximum Euclidean distance for a match.

        Returns:
            The subset of this frame's DataFrame within the radius.
        """
        idx = self._engine.radius_query(cx, cy, distance)
        return self._df[pl.Series(idx.astype(np.uint32))]

    def points_within_distance_of_polygon(self, polygon, distance: float) -> pl.DataFrame:
        """Return the rows whose point lies within `distance` of a polygon boundary (zero inside).

        Args:
            polygon: A single shapely Polygon (interior holes supported).
            distance: Maximum Euclidean point-to-polygon distance for a match.

        Returns:
            The subset of this frame's DataFrame matching the distance predicate.
        """
        idx = self._engine.points_within_distance_of_polygon(polygon, distance)
        return self._df[pl.Series(idx.astype(np.uint32))]

    @staticmethod
    def convex_hull_area(xs, ys) -> float:
        """Compute the area of the convex hull of a standalone point set.

        Args:
            xs: Sequence of x coordinates.
            ys: Sequence of y coordinates.

        Returns:
            The area of the convex hull of the point set.
        """
        return Engine.convex_hull_area(xs, ys)

    @property
    def df(self) -> pl.DataFrame:
        """Expose the materialized DataFrame backing this frame.

        Returns:
            The underlying Polars DataFrame.
        """
        return self._df

    @property
    def engine(self) -> Engine:
        """Expose the spatial index engine backing this frame.

        Returns:
            The underlying Engine.
        """
        return self._engine

    @property
    def x_col(self) -> str:
        """Expose the x-coordinate column name.

        Returns:
            The name of the x-coordinate column.
        """
        return self._x_col

    @property
    def y_col(self) -> str:
        """Expose the y-coordinate column name.

        Returns:
            The name of the y-coordinate column.
        """
        return self._y_col
