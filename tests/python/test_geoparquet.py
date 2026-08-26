"""GeoParquet metadata inference for lazy spatial sources."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import shapely

from pycanopy import SpatialFrame, infer_geoparquet_geometry


def _write_geoparquet(
    path: Path,
    geometry_types: list[str],
    *,
    encoding: str = "WKB",
    geometry_col: str = "geometry",
) -> None:
    # Write a minimal GeoParquet fixture with real file-level metadata
    if geometry_types and geometry_types[0].startswith("Point"):
        geometry = shapely.Point(0, 0).wkb
    else:
        geometry = shapely.box(0, 0, 1, 1).wkb
    geo = {
        "version": "1.1.0",
        "primary_column": geometry_col,
        "columns": {
            geometry_col: {
                "encoding": encoding,
                "geometry_types": geometry_types,
            }
        },
    }
    pl.DataFrame({"id": [1], geometry_col: [geometry]}).write_parquet(
        path,
        metadata={"geo": json.dumps(geo)},
    )


@pytest.mark.parametrize(
    ("geometry_types", "expected_kind"),
    [
        (["Point"], "point"),
        (["Point Z"], "point"),
        (["Polygon", "MultiPolygon"], "polygon"),
    ],
)
def test_infer_geoparquet_geometry_kinds(tmp_path, geometry_types, expected_kind):
    path = tmp_path / "geometry.parquet"
    _write_geoparquet(path, geometry_types)

    assert infer_geoparquet_geometry(path) == ("geometry", expected_kind)


def test_scan_parquet_infers_geometry_configuration(tmp_path):
    path = tmp_path / "points.parquet"
    _write_geoparquet(path, ["Point"])

    result = (
        SpatialFrame.scan_parquet(path, parallel="none")
        .lazy()
        .range_query(-1, -1, 1, 1)
        .select("id")
        .collect()
    )

    assert result["id"].to_list() == [1]


def test_inference_uses_explicit_overrides(tmp_path):
    path = tmp_path / "geometry.parquet"
    _write_geoparquet(path, [])

    assert infer_geoparquet_geometry(path, geometry_kind="point") == (
        "geometry",
        "point",
    )
    assert infer_geoparquet_geometry(path, geometry_col="geometry", geometry_kind="polygon") == (
        "geometry",
        "polygon",
    )


def test_inference_rejects_missing_metadata(tmp_path):
    path = tmp_path / "ordinary.parquet"
    pl.DataFrame({"geometry": [shapely.Point(0, 0).wkb]}).write_parquet(path)

    with pytest.raises(ValueError, match="no GeoParquet metadata"):
        infer_geoparquet_geometry(path)

    sf = SpatialFrame.scan_parquet(path, "geometry", "point")
    assert sf.lazy().collect().height == 1


@pytest.mark.parametrize(
    ("geometry_types", "encoding", "message"),
    [
        (["LineString"], "WKB", "unsupported geometry_types"),
        (["Point"], "point", "unsupported encoding"),
        ([], "WKB", "no geometry_types"),
    ],
)
def test_inference_rejects_unsupported_metadata(
    tmp_path,
    geometry_types,
    encoding,
    message,
):
    path = tmp_path / "unsupported.parquet"
    _write_geoparquet(path, geometry_types, encoding=encoding)

    with pytest.raises(ValueError, match=message):
        infer_geoparquet_geometry(path)


def test_inference_supports_local_dataset_sources(tmp_path):
    path = tmp_path / "part.parquet"
    _write_geoparquet(path, ["Point"])

    assert infer_geoparquet_geometry(tmp_path) == ("geometry", "point")
    assert infer_geoparquet_geometry(str(tmp_path / "*.parquet")) == (
        "geometry",
        "point",
    )
