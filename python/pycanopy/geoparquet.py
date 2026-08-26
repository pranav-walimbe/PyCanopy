"""Resolve PyCanopy geometry configuration from GeoParquet metadata."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Literal

import polars as pl
from polars.io.cloud import CredentialProviderFunction

GeometryKind = Literal["point", "polygon"]
ParquetSource = str | Path | list[str] | list[Path]


def _local_metadata_source(source: str | Path) -> str | Path | None:
    # Choose one representative footer without touching the dataset rows
    if isinstance(source, str) and "://" in source:
        if glob.has_magic(source):
            raise ValueError(
                "cannot infer geometry from a cloud glob; specify geometry_col and "
                "geometry_kind explicitly"
            )
        return source
    path = Path(source)
    if path.is_dir():
        return next(path.rglob("*.parquet"), None)
    if glob.has_magic(str(source)):
        return next(glob.iglob(str(source), recursive=True), None)
    return source


def _metadata_source(source: ParquetSource) -> str | Path:
    # Select the first resolvable source from explicit dataset paths
    sources = source if isinstance(source, list) else [source]
    for item in sources:
        metadata_source = _local_metadata_source(item)
        if metadata_source is not None:
            return metadata_source
    raise ValueError("cannot infer geometry from an empty Parquet source")


def _geometry_kind(metadata: dict[str, object], column: str) -> GeometryKind:
    # Map the declared GeoParquet geometry family to a supported engine kind
    geometry_types = metadata.get("geometry_types")
    if not isinstance(geometry_types, list) or not geometry_types:
        raise ValueError(
            f"GeoParquet column {column!r} has no geometry_types; specify geometry_kind explicitly"
        )
    if not all(isinstance(value, str) for value in geometry_types):
        raise ValueError(f"GeoParquet column {column!r} has invalid geometry_types")
    base_types = {value.partition(" ")[0] for value in geometry_types}
    if base_types == {"Point"}:
        return "point"
    if base_types <= {"Polygon", "MultiPolygon"}:
        return "polygon"
    declared = ", ".join(sorted(geometry_types))
    raise ValueError(f"GeoParquet column {column!r} has unsupported geometry_types: {declared}")


def _parse_geometry_metadata(
    raw_metadata: dict[str, str],
    geometry_col: str | None,
    geometry_kind: GeometryKind | None,
) -> tuple[str, GeometryKind]:
    # Resolve one footer while validating the selected column's encoding
    raw_geo = raw_metadata.get("geo")
    if raw_geo is None:
        raise ValueError(
            "Parquet source has no GeoParquet metadata; specify geometry_col and "
            "geometry_kind explicitly"
        )
    try:
        geo = json.loads(raw_geo)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Parquet source has invalid GeoParquet metadata") from error
    if not isinstance(geo, dict):
        raise ValueError("Parquet source has invalid GeoParquet metadata")

    column = geometry_col if geometry_col is not None else geo.get("primary_column")
    if not isinstance(column, str) or not column:
        raise ValueError("GeoParquet metadata has no valid primary_column")
    columns = geo.get("columns")
    if not isinstance(columns, dict) or not isinstance(columns.get(column), dict):
        raise ValueError(f"GeoParquet metadata has no definition for column {column!r}")
    column_metadata = columns[column]
    if column_metadata.get("encoding") != "WKB":
        encoding = column_metadata.get("encoding")
        raise ValueError(
            f"GeoParquet column {column!r} uses unsupported encoding {encoding!r}; "
            "only 'WKB' is supported"
        )
    kind = geometry_kind if geometry_kind is not None else _geometry_kind(column_metadata, column)
    return column, kind


def infer_geoparquet_geometry(
    source: ParquetSource,
    geometry_col: str | None = None,
    geometry_kind: GeometryKind | None = None,
    storage_options: dict[str, str] | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    retries: int | None = None,
) -> tuple[str, GeometryKind]:
    """Resolve a WKB geometry column and kind from GeoParquet metadata.

    Explicit geometry values override their metadata counterparts. Dataset sources
    use the first matching file as their representative footer.

    Args:
        source: Parquet file, local dataset, cloud URI, or explicit list of paths.
        geometry_col: Geometry column override.
        geometry_kind: Geometry kind override, ``"point"`` or ``"polygon"``.
        storage_options: Cloud connection options for Polars.
        credential_provider: Cloud credential provider forwarded to Polars.
        retries: Maximum cloud metadata read retries.

    Returns:
        The resolved geometry column and kind.
    """
    if geometry_kind not in (None, "point", "polygon"):
        raise ValueError("geometry_kind must be 'point' or 'polygon'")
    metadata = pl.read_parquet_metadata(
        _metadata_source(source),
        storage_options=storage_options,
        credential_provider=credential_provider,
        retries=retries,
    )
    return _parse_geometry_metadata(metadata, geometry_col, geometry_kind)
