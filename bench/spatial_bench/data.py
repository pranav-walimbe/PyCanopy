"""Loaders for official Apache SpatialBench parquet data.

The benchmark tables are stored as parquet with geometry columns encoded as
Well-Known Binary (WKB). This module reads those tables (from a local directory
or an ``s3://`` URI) and converts geometries into the coordinate forms PyCanopy
consumes: x/y arrays for point columns, shapely Polygons for polygon columns.

Table/geometry reference (SpatialBench star schema):
  * trip      fact   : t_pickuploc, t_dropoffloc (WKB Point)
  * zone      dim    : z_boundary (WKB Polygon / MultiPolygon)
  * building  dim    : b_boundary (WKB Polygon / MultiPolygon)
  * customer / driver / vehicle : non-spatial dimensions

MultiPolygon handling: PyCanopy indexes single Polygons, so a MultiPolygon is
reduced to its largest-area constituent Polygon. This is a documented fidelity
caveat; it affects a small fraction of administrative zones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import polars as pl
import shapely

from pycanopy import SpatialFrame

# Geometry column names per table, for convenience.
POINT_GEOM = {
    "trip_pickup": ("trip", "t_pickuploc"),
    "trip_dropoff": ("trip", "t_dropoffloc"),
}
POLYGON_GEOM = {
    "zone": ("zone", "z_boundary"),
    "building": ("building", "b_boundary"),
}

_SHAPELY_POLYGON = 3
_SHAPELY_MULTIPOLYGON = 6


def _resolve_table(data_dir: str, table: str) -> tuple[str, bool]:
    """Locate ``table`` under ``data_dir``, returning (path, is_directory).

    Handles the single-file layout (``trip.parquet`` from ``spatialbench-cli``) and
    the directory-of-files layout (``trip/``) of the public S3 datasets. ``s3://``
    URIs are assumed to be directory-of-files. The single source of truth for path
    resolution shared by the polars (glob) and pandas (directory) readers below.
    """
    base = data_dir.rstrip("/")
    if base.startswith("s3://"):
        return f"{base}/{table}", True
    single = f"{base}/{table}.parquet"
    if os.path.exists(single):
        return single, False
    if os.path.isdir(f"{base}/{table}"):
        return f"{base}/{table}", True
    return single, False


def _parquet_glob(data_dir: str, table: str) -> str:
    """Return a parquet path/glob for ``table``, for the polars reader.

    Directories become a ``**/*.parquet`` glob, which polars (and pyarrow glob)
    expand. Works for local paths and ``s3://`` URIs.
    """
    path, is_dir = _resolve_table(data_dir, table)
    return f"{path}/**/*.parquet" if is_dir else path


def table_path(data_dir: str, table: str) -> str:
    """Return a parquet path for ``table`` that pandas/pyarrow reads directly.

    The GeoPandas reference loads tables with ``pd.read_parquet``, whose pyarrow
    engine reads a directory as a dataset but does not expand the ``**/*.parquet``
    glob that ``_parquet_glob`` hands polars. So a directory-of-files table is
    returned as the bare directory (pyarrow dataset discovery skips ``_`` and ``.``
    prefixed files such as ``_SUCCESS``). Works for local paths and ``s3://`` URIs.
    """
    return _resolve_table(data_dir, table)[0]


def read_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read one SpatialBench table as a Polars DataFrame.

    Args:
        data_dir: Local directory or ``s3://`` URI containing the table.
        table: Table name (e.g. "trip", "zone", "building", "customer").
        columns: Optional column subset to read (projection pushdown).

    Returns:
        Materialized Polars DataFrame. Geometry columns remain WKB (pl.Binary).
    """
    path = _parquet_glob(data_dir, table)
    return pl.read_parquet(path, columns=columns)


def wkb_to_polygons(series: pl.Series) -> list:
    """Convert a WKB-encoded polygon column to a list of shapely Polygons.

    MultiPolygons are reduced to their largest-area constituent Polygon so the
    result is a flat list of single Polygons suitable for Engine.from_polygons.

    Args:
        series: Polars Binary series of WKB polygon/multipolygon geometries.

    Returns:
        List of shapely Polygon objects, one per input row.
    """
    geoms = shapely.from_wkb(series.to_numpy())
    type_ids = shapely.get_type_id(geoms)
    out = []
    for geom, tid in zip(geoms, type_ids):
        if tid == _SHAPELY_MULTIPOLYGON:
            parts = list(geom.geoms)
            out.append(max(parts, key=lambda p: p.area))
        else:
            out.append(geom)
    return out


@dataclass
class SpatialBenchTables:
    """Lazily-built handles to the SpatialBench tables for a given scale factor.

    Tables are read on first access and cached. Only the tables a query touches
    are loaded, keeping memory in check for large scale factors.

    Args:
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        scale_factor: SpatialBench scale factor (used only for reporting metadata).
        index_mode: Index build policy ("eager" / "none" / "auto") for every
            SpatialFrame built here. "none" is the --no-index benchmark mode.
    """

    data_dir: str
    scale_factor: float
    index_mode: str = "eager"
    _cache: dict[str, pl.DataFrame] | None = None

    def table(self, name: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Return table ``name``, reading and caching it on first access."""
        if self._cache is None:
            self._cache = {}
        key = name if columns is None else f"{name}:{','.join(columns)}"
        if key not in self._cache:
            self._cache[key] = read_table(self.data_dir, name, columns)
        return self._cache[key]

    def point_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a point SpatialFrame from a WKB point column of ``df``.

        Adds internal ``_x`` / ``_y`` columns and indexes on them. The returned
        frame's DataFrame retains all original columns plus ``_x`` / ``_y``.
        """
        return SpatialFrame.from_wkb_points(df, wkb_col, index_mode=self.index_mode)

    def polygon_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a polygon SpatialFrame from a WKB polygon column of ``df``.

        The DataFrame is given a ``_geom`` Object column of shapely Polygons that
        the polygon Engine indexes. The source WKB column is dropped once ``_geom``
        is built, so spatial joins do not replicate the raw geometry bytes across
        their matched rows.
        """
        polys = wkb_to_polygons(df[wkb_col])
        enriched = df.drop(wkb_col).with_columns(pl.Series("_geom", polys, dtype=pl.Object))
        return SpatialFrame.from_polygons(
            enriched, geometry_col="_geom", index_mode=self.index_mode
        )
