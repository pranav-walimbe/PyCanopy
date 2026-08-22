"""
Define SpatialFrame which is the entry point for spatial query planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from pycanopy.coordinates import resolve_coordinate_system
from pycanopy.engine import Engine, wkb_points_to_xy
from pycanopy.lazy import SpatialLazyFrame

_DEFAULT_INGEST_BATCH_SIZE = 32_768


@dataclass(frozen=True)
class _LazySpatialSource:
    frame: pl.LazyFrame
    geometry_col: str
    geometry_kind: Literal["point", "polygon"]
    index_mode: str
    coordinate_system: Literal["planar", "geographic"]
    ingest_batch_size: int


class SpatialFrame:
    """Own materialized or deferred spatial data for query planning.

    Args:
        df: Materialized Polars DataFrame.
        x_col: Name of the column holding x (longitude/easting) coordinates.
        y_col: Name of the column holding y (latitude/northing) coordinates.
        index_mode: Index build policy fixed for this frame's engine. "auto"
            (default) builds only when the cost model beats a scan, "eager" always
            builds an index, "none" always scans brute-force.
        coordinate_system: How distances are measured. "planar" (default) uses the
            coordinates' own units, "geographic" reads lon/lat degrees as meters.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        index_mode: str = "auto",
        coordinate_system: Literal["planar", "geographic"] | None = None,
    ) -> None:
        if x_col not in df.columns:
            raise ValueError(f"x_col {x_col!r} not found in DataFrame")
        if y_col not in df.columns:
            raise ValueError(f"y_col {y_col!r} not found in DataFrame")
        self._df = df
        self._x_col = x_col
        self._y_col = y_col
        xs = df[x_col].to_numpy()
        ys = df[y_col].to_numpy()
        self._engine = Engine.from_coords(xs, ys)
        self._engine.set_index_mode(index_mode)
        self._engine.set_coordinate_system(resolve_coordinate_system(coordinate_system, xs, ys))
        self._geometry_kind: Literal["point", "polygon"] = "point"
        self._lazy_source: _LazySpatialSource | None = None

    @classmethod
    def _from_engine(
        cls,
        df: pl.DataFrame,
        engine: Engine,
        x_col: str,
        y_col: str,
        geometry_kind: Literal["point", "polygon"],
    ) -> SpatialFrame:
        # Construct a derived frame from already-aligned attributes and native geometry
        sf = object.__new__(cls)
        sf._df = df
        sf._x_col = x_col
        sf._y_col = y_col
        sf._engine = engine
        sf._geometry_kind = geometry_kind
        sf._lazy_source = None
        return sf

    @classmethod
    def from_lazy(
        cls,
        frame: pl.LazyFrame,
        geometry_col: str,
        geometry_kind: Literal["point", "polygon"],
        index_mode: str = "auto",
        coordinate_system: Literal["planar", "geographic"] | None = None,
        ingest_batch_size: int = _DEFAULT_INGEST_BATCH_SIZE,
    ) -> SpatialFrame:
        """Construct a deferred SpatialFrame from a Polars LazyFrame.

        Args:
            frame: Lazy source of spatial rows.
            geometry_col: Name of the Binary WKB geometry column.
            geometry_kind: WKB geometry kind, ``"point"`` or ``"polygon"``.
            index_mode: Index build policy ("eager" / "none" / "auto").
            coordinate_system: Point distance system ("planar" / "geographic").
            ingest_batch_size: Source rows decoded per batch.

        Returns:
            A SpatialFrame materialized when its query is collected.
        """
        if not isinstance(frame, pl.LazyFrame):
            raise TypeError("frame must be a polars LazyFrame")
        if geometry_kind not in ("point", "polygon"):
            raise ValueError("geometry_kind must be 'point' or 'polygon'")
        if isinstance(ingest_batch_size, bool) or not isinstance(ingest_batch_size, int):
            raise TypeError("ingest_batch_size must be an integer")
        if ingest_batch_size <= 0:
            raise ValueError("ingest_batch_size must be positive")
        if geometry_kind == "polygon" and coordinate_system not in (None, "planar"):
            raise ValueError("polygon sources support only planar coordinates")
        resolved_system = resolve_coordinate_system(
            coordinate_system,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        sf = object.__new__(cls)
        sf._df = None
        sf._x_col = "_x"
        sf._y_col = "_y"
        sf._engine = None
        sf._geometry_kind = geometry_kind
        sf._lazy_source = _LazySpatialSource(
            frame,
            geometry_col,
            geometry_kind,
            index_mode,
            resolved_system,
            ingest_batch_size,
        )
        return sf

    @classmethod
    def scan_parquet(
        cls,
        source: str | Path | list[str] | list[Path],
        geometry_col: str,
        geometry_kind: Literal["point", "polygon"],
        index_mode: str = "auto",
        storage_options: dict[str, str] | None = None,
        coordinate_system: Literal["planar", "geographic"] | None = None,
        ingest_batch_size: int = _DEFAULT_INGEST_BATCH_SIZE,
        **scan_options: object,
    ) -> SpatialFrame:
        """Construct a deferred SpatialFrame from Parquet.

        Args:
            source: Parquet path, glob, cloud URI, or list of paths.
            geometry_col: Name of the Binary WKB geometry column.
            geometry_kind: WKB geometry kind, ``"point"`` or ``"polygon"``.
            index_mode: Index build policy ("eager" / "none" / "auto").
            storage_options: Cloud connection options for Polars.
            coordinate_system: Point distance system ("planar" / "geographic").
            ingest_batch_size: Source rows decoded per batch.
            **scan_options: Options forwarded to ``polars.scan_parquet``.

        Returns:
            A SpatialFrame materialized when its query is collected.
        """
        frame = pl.scan_parquet(source, storage_options=storage_options, **scan_options)
        return cls.from_lazy(
            frame,
            geometry_col,
            geometry_kind,
            index_mode,
            coordinate_system,
            ingest_batch_size,
        )

    @property
    def _is_deferred(self) -> bool:
        # Report whether this frame still owns a lazy source instead of materialized data
        return self._lazy_source is not None

    def _lazy_schema(self) -> pl.Schema:
        # Resolve source schema only during execution preparation
        if self._lazy_source is None:
            raise RuntimeError("materialized SpatialFrame has no lazy source schema")
        return self._lazy_source.frame.collect_schema()

    def _materialize_lazy(
        self,
        required_columns: set[str] | None,
        schema: pl.Schema,
    ) -> SpatialFrame:
        # Stream projected input into native geometry and retain only required columns
        source = self._lazy_source
        if source is None:
            return self
        if source.geometry_col not in schema:
            raise ValueError(f"geometry_col {source.geometry_col!r} not found in LazyFrame")
        if schema[source.geometry_col] != pl.Binary:
            raise TypeError(
                f"geometry_col {source.geometry_col!r} must have Binary dtype, "
                f"got {schema[source.geometry_col]}"
            )

        schema_columns = list(schema.names())
        if required_columns is None:
            retained_columns = schema_columns
        else:
            retained_columns = [name for name in schema_columns if name in required_columns]
        scan_columns = list(dict.fromkeys([*retained_columns, source.geometry_col]))
        retained_batches: list[pl.DataFrame] = []

        def geometry_batches():
            # Yield geometry for native decoding after saving the requested attributes
            for batch in source.frame.select(scan_columns).collect_batches(
                chunk_size=source.ingest_batch_size,
                maintain_order=True,
            ):
                if retained_columns:
                    retained_batches.append(batch.select(retained_columns))
                geometry = batch[source.geometry_col]
                del batch
                yield geometry
                del geometry

        if source.geometry_kind == "point":
            engine = Engine._from_wkb_point_batches(geometry_batches())
            extent = engine.extent
            if extent is None:
                xs = ys = np.empty(0, dtype=np.float64)
            else:
                xs = np.asarray([extent[0], extent[2]])
                ys = np.asarray([extent[1], extent[3]])
            engine.set_coordinate_system(
                resolve_coordinate_system(source.coordinate_system, xs, ys)
            )
        else:
            engine = Engine._from_wkb_polygon_batches(geometry_batches())
        engine.set_index_mode(source.index_mode)
        if retained_batches:
            collected = pl.concat(retained_batches, how="vertical", rechunk=False)
        else:
            collected = pl.DataFrame(schema={name: schema[name] for name in retained_columns})
        return self._from_engine(
            collected,
            engine,
            self._x_col,
            self._y_col,
            source.geometry_kind,
        )

    @classmethod
    def from_wkb_points(
        cls,
        df: pl.DataFrame,
        wkb_col: str,
        x_col: str = "_x",
        y_col: str = "_y",
        index_mode: str = "auto",
        coordinate_system: Literal["planar", "geographic"] | None = None,
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
            coordinate_system: How distances are measured ("planar" / "geographic").

        Returns:
            SpatialFrame backed by a point index.
        """
        if wkb_col not in df.columns:
            raise ValueError(f"wkb_col {wkb_col!r} not found in DataFrame")
        xs, ys = wkb_points_to_xy(df[wkb_col])
        enriched = df.with_columns(pl.Series(x_col, xs), pl.Series(y_col, ys))
        return cls(
            enriched,
            x_col=x_col,
            y_col=y_col,
            index_mode=index_mode,
            coordinate_system=coordinate_system,
        )

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
        return cls._from_engine(df, engine, x_col, y_col, "polygon")

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
        WKB column is dropped from the retained DataFrame once native geometry is built.

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
        return cls._from_engine(df.drop(wkb_col), engine, x_col, y_col, "polygon")

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
            New SpatialFrame with compact matching geometry. Its index builds on demand
            according to the inherited index policy.
        """
        engine = self.engine
        indices = engine.range_query(min_x, min_y, max_x, max_y)
        idx_s = pl.Series(np.asarray(indices, dtype=np.uint32))
        filtered = self._df[idx_s] if indices else self._df.clear()
        if self._geometry_kind == "polygon":
            return self._from_engine(
                filtered,
                engine.subset(indices),
                self._x_col,
                self._y_col,
                "polygon",
            )
        return SpatialFrame(
            filtered,
            self._x_col,
            self._y_col,
            coordinate_system=self.coordinate_system,
        )

    # Geometry aggregations and transforms (polygon datasets)

    def polygon_areas(self) -> pl.DataFrame:
        """Append an unsigned 'area' column to this frame's DataFrame (polygon datasets).

        Returns:
            The frame's DataFrame with an appended unsigned 'area' column.
        """
        areas = self.engine.polygon_areas()
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
        flat = self.engine.polygon_intersects_self_join()
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
        areas = self.engine.polygon_areas()
        overlap = self.engine.polygon_pairs_intersection_area(i_idx, j_idx)
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
                "left": i_idx,
                "right": j_idx,
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
        idx = self.engine.radius_query(cx, cy, distance)
        return self._df[pl.Series(idx.astype(np.uint32))]

    def points_within_distance_of_polygon(self, polygon, distance: float) -> pl.DataFrame:
        """Return the rows whose point lies within `distance` of a polygon boundary (zero inside).

        Args:
            polygon: A single shapely Polygon (interior holes supported).
            distance: Maximum Euclidean point-to-polygon distance for a match.

        Returns:
            The subset of this frame's DataFrame matching the distance predicate.
        """
        idx = self.engine.points_within_distance_of_polygon(polygon, distance)
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
        if self._is_deferred:
            raise RuntimeError("deferred SpatialFrame data is available only through lazy queries")
        return self._df

    @property
    def engine(self) -> Engine:
        """Expose the spatial index engine backing this frame.

        Returns:
            The underlying Engine.
        """
        if self._is_deferred:
            raise RuntimeError("deferred SpatialFrame Engine is built when a lazy query collects")
        return self._engine

    @property
    def coordinate_system(self) -> str:
        """Expose how this frame measures threshold distances.

        Returns:
            Either "planar" or "geographic".
        """
        return self.engine.coordinate_system

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
