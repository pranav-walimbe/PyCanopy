"""Deferred Polars sources and automatic WKB projection behavior."""

from __future__ import annotations

import polars as pl
import pytest
import shapely
from polars.io.plugins import register_io_source

import pycanopy as pc
from pycanopy import SpatialFrame, SpatialLazyFrame
from pycanopy.engine import Engine


def _polygon_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "unused": ["a", "b", "c"],
            "geometry": [
                shapely.box(0, 0, 1, 1).wkb,
                shapely.box(10, 10, 11, 11).wkb,
                shapely.box(20, 20, 21, 21).wkb,
            ],
        }
    )


def _point_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
            "unused": ["a", "b", "c", "d", "e"],
            "geometry": [
                shapely.Point(0, 0).wkb,
                shapely.Point(1, 0).wkb,
                shapely.Point(2, 0).wkb,
                shapely.Point(3, 0).wkb,
                shapely.Point(4, 0).wkb,
            ],
        }
    )


def _tracked_source(df: pl.DataFrame) -> tuple[pl.LazyFrame, list[list[str] | None]]:
    # Return an IO source that records the projection Polars pushes into it
    projections: list[list[str] | None] = []

    def source(with_columns, predicate, n_rows, batch_size):
        projections.append(None if with_columns is None else list(with_columns))
        output = df
        if predicate is not None:
            output = output.filter(predicate)
        if with_columns is not None:
            output = output.select(with_columns)
        if n_rows is not None:
            output = output.head(n_rows)
        yield output

    return register_io_source(source, schema=df.schema), projections


def test_from_lazy_defers_source_execution_and_explain_io():
    lazy_frame, projections = _tracked_source(_polygon_data())

    sf = SpatialFrame.from_lazy(lazy_frame, "geometry", "polygon")
    explanation = sf.lazy().range_query(0, 0, 2, 2).select("id").explain()

    assert projections == []
    assert "LAZY SOURCE [N=?]" in explanation
    assert sf.lazy().range_query(0, 0, 2, 2).select("id").collect()["id"].to_list() == [1]
    assert len(projections) == 1


def test_lazy_projection_reads_geometry_but_omits_it_from_output():
    lazy_frame, projections = _tracked_source(_polygon_data())
    sf = SpatialFrame.from_lazy(lazy_frame, "geometry", "polygon")

    result = sf.lazy().range_query(0, 0, 2, 2).select("id").collect()

    assert result.columns == ["id"]
    assert set(projections[0]) == {"id", "geometry"}
    assert "unused" not in projections[0]


def test_lazy_projection_preserves_exact_requested_wkb():
    source = _polygon_data()
    sf = SpatialFrame.from_lazy(source.lazy(), "geometry", "polygon")

    result = sf.lazy().range_query(0, 0, 2, 2).select("id", "geometry").collect()

    assert result["id"].to_list() == [1]
    assert result["geometry"].to_list() == [source["geometry"][0]]


def test_lazy_query_without_select_returns_all_source_columns():
    source = _polygon_data()
    sf = SpatialFrame.from_lazy(source.lazy(), "geometry", "polygon")

    result = sf.lazy().range_query(0, 0, 2, 2).collect()

    assert result.columns == source.columns
    assert result["geometry"].to_list() == [source["geometry"][0]]


def test_lazy_scalar_dependency_is_read_but_not_returned():
    lazy_frame, projections = _tracked_source(_polygon_data())
    sf = SpatialFrame.from_lazy(lazy_frame, "geometry", "polygon")

    result = sf.lazy().filter(pl.col("value") > 15).range_query(0, 0, 12, 12).select("id").collect()

    assert result["id"].to_list() == [2]
    assert set(projections[0]) == {"id", "value", "geometry"}


def test_repeated_lazy_collections_choose_geometry_independently():
    source = _polygon_data()
    lazy_frame, projections = _tracked_source(source)
    sf = SpatialFrame.from_lazy(lazy_frame, "geometry", "polygon")

    ids = sf.lazy().range_query(0, 0, 2, 2).select("id").collect()
    geometries = sf.lazy().range_query(0, 0, 2, 2).select("geometry").collect()

    assert ids["id"].to_list() == [1]
    assert geometries["geometry"].to_list() == [source["geometry"][0]]
    assert len(projections) == 2
    assert set(projections[0]) == {"id", "geometry"}
    assert set(projections[1]) == {"geometry"}


def test_scan_parquet_wraps_polars_lazy_scan(tmp_path):
    source = _polygon_data()
    path = tmp_path / "polygons.parquet"
    source.write_parquet(path)

    sf = SpatialFrame.scan_parquet(
        path,
        "geometry",
        "polygon",
        ingest_batch_size=3,
        parallel="none",
    )
    result = sf.lazy().range_query(0, 0, 2, 2).select("id", "geometry").collect()

    assert sf._lazy_source.ingest_batch_size == 3
    assert result["id"].to_list() == [1]
    assert result["geometry"].to_list() == [source["geometry"][0]]


def test_lazy_transformation_runs_before_spatial_ingestion():
    source = _polygon_data()
    transformed = source.lazy().filter(pl.col("id") != 1)
    sf = SpatialFrame.from_lazy(transformed, "geometry", "polygon")

    result = sf.lazy().range_query(0, 0, 12, 12).select("id").collect()

    assert result["id"].to_list() == [2]


def test_lazy_grouped_join_uses_projected_target_columns():
    source = _polygon_data()
    sf = SpatialFrame.from_lazy(source.lazy(), "geometry", "polygon")
    points = pl.DataFrame({"point_id": [1, 2, 3], "x": [0.5, 0.6, 10.5], "y": [0.5, 0.6, 10.5]})

    result = sf.lazy().within_join(points, "x", "y").group_by("id").agg(count=pc.agg.count())

    assert result.sort("id").to_dict(as_series=False) == {"id": [1, 2], "count": [2, 1]}


def test_lazy_join_preserves_requested_target_wkb_with_name_collision():
    source = _polygon_data()
    sf = SpatialFrame.from_lazy(source.lazy(), "geometry", "polygon")
    points = pl.DataFrame(
        {"point_id": [5], "x": [0.5], "y": [0.5], "geometry": [b"probe-geometry"]}
    )

    result = sf.lazy().within_join(points, "x", "y").select("point_id", "right_geometry").collect()

    assert result["point_id"].to_list() == [5]
    assert result["right_geometry"].to_list() == [source["geometry"][0]]

    query_geometry = sf.lazy().within_join(points, "x", "y").select("geometry")
    assert "geometry" not in query_geometry._prepare().df.columns
    assert query_geometry.collect()["geometry"].to_list() == [b"probe-geometry"]


def test_lazy_source_validation_is_deferred_until_collection():
    missing = SpatialFrame.from_lazy(pl.DataFrame({"id": [1]}).lazy(), "geometry", "polygon")
    wrong_type = SpatialFrame.from_lazy(
        pl.DataFrame({"geometry": ["not-wkb"]}).lazy(), "geometry", "polygon"
    )

    with pytest.raises(ValueError, match="not found"):
        missing.lazy().collect()
    with pytest.raises(TypeError, match="Binary dtype"):
        wrong_type.lazy().collect()


def test_lazy_source_rejects_unsupported_inputs():
    with pytest.raises(TypeError, match="polars LazyFrame"):
        SpatialFrame.from_lazy(_polygon_data(), "geometry", "polygon")
    with pytest.raises(ValueError, match=r"point.*polygon"):
        SpatialFrame.from_lazy(_polygon_data().lazy(), "geometry", "line")
    with pytest.raises(ValueError, match="only planar"):
        SpatialFrame.from_lazy(
            _polygon_data().lazy(),
            "geometry",
            "polygon",
            coordinate_system="geographic",
        )


def test_lazy_source_validates_ingestion_batch_size():
    source = _polygon_data().lazy()

    assert (
        SpatialFrame.from_lazy(source, "geometry", "polygon")._lazy_source.ingest_batch_size
        == 32_768
    )
    for value in (0, -1):
        with pytest.raises(ValueError, match="must be positive"):
            SpatialFrame.from_lazy(source, "geometry", "polygon", ingest_batch_size=value)
    for value in (True, 1.5, None):
        with pytest.raises(TypeError, match="must be an integer"):
            SpatialFrame.from_lazy(source, "geometry", "polygon", ingest_batch_size=value)


def test_deferred_frame_requires_lazy_query_access():
    sf = SpatialFrame.from_lazy(_polygon_data().lazy(), "geometry", "polygon")

    with pytest.raises(RuntimeError, match="lazy queries"):
        _ = sf.df
    with pytest.raises(RuntimeError, match="built when"):
        _ = sf.engine


def test_collect_all_executes_deferred_projections_independently():
    source = _polygon_data()
    lazy_frame, projections = _tracked_source(source)
    sf = SpatialFrame.from_lazy(lazy_frame, "geometry", "polygon")
    ids = sf.lazy().range_query(0, 0, 2, 2).select("id")
    geometry = sf.lazy().range_query(0, 0, 2, 2).select("geometry")

    results = SpatialLazyFrame.collect_all([ids, geometry])

    assert results[0]["id"].to_list() == [1]
    assert results[1]["geometry"].to_list() == [source["geometry"][0]]
    assert len(projections) == 2


def test_lazy_ingestion_decodes_bounded_batches_in_order(monkeypatch):
    source = pl.DataFrame(
        {
            "id": list(range(7)),
            "geometry": [shapely.box(i, 0, i + 0.5, 0.5).wkb for i in range(7)],
        }
    )
    batch_lengths: list[int] = []
    original = Engine._from_wkb_polygon_batches.__func__

    def tracked(cls, columns):
        # Record each geometry batch before forwarding it to the native builder
        def inspected():
            for column in columns:
                batch_lengths.append(len(column))
                yield column

        return original(cls, inspected())

    monkeypatch.setattr(Engine, "_from_wkb_polygon_batches", classmethod(tracked))

    result = (
        SpatialFrame.from_lazy(source.lazy(), "geometry", "polygon", ingest_batch_size=2)
        .lazy()
        .range_query(-1, -1, 10, 10)
        .select("id")
        .collect()
    )

    assert batch_lengths == [2, 2, 2, 1]
    assert result["id"].to_list() == source["id"].to_list()


def test_lazy_batched_ingestion_preserves_multipolygons_and_holes():
    polygon = shapely.box(0, 0, 1, 1)
    multipolygon = shapely.MultiPolygon([shapely.box(10, 0, 11, 1), shapely.box(20, 0, 21, 1)])
    polygon_with_hole = shapely.Polygon(
        [(30, 0), (34, 0), (34, 4), (30, 4), (30, 0)],
        [[(31, 1), (32, 1), (32, 2), (31, 1)]],
    )
    source = pl.DataFrame(
        {
            "id": ["polygon", "multipolygon", "hole"],
            "geometry": [polygon.wkb, multipolygon.wkb, polygon_with_hole.wkb],
        }
    )
    sf = SpatialFrame.from_lazy(source.lazy(), "geometry", "polygon", ingest_batch_size=1)

    assert sf.lazy().contains(20.5, 0.5).select("id").collect()["id"].to_list() == ["multipolygon"]
    assert sf.lazy().contains(31.5, 1.25).select("id").collect().is_empty()
    retained = sf.lazy().range_query(-1, -1, 40, 10).select("id", "geometry").collect()
    assert retained.to_dict(as_series=False) == source.to_dict(as_series=False)


def test_lazy_batched_ingestion_supports_empty_sources():
    source = pl.DataFrame(
        schema={
            "id": pl.Int64,
            "geometry": pl.Binary,
        }
    )
    result = (
        SpatialFrame.from_lazy(source.lazy(), "geometry", "polygon")
        .lazy()
        .range_query(0, 0, 1, 1)
        .select("id")
        .collect()
    )

    assert result.schema == pl.Schema({"id": pl.Int64})
    assert result.is_empty()


def test_lazy_batched_ingestion_keeps_unusual_wkb_fallback():
    polygon_3d = shapely.Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 0, 1)])
    source = pl.DataFrame(
        {
            "id": [1, 2],
            "geometry": [
                shapely.to_wkb(polygon_3d, output_dimension=3),
                shapely.box(10, 10, 11, 11).wkb,
            ],
        }
    )
    result = (
        SpatialFrame.from_lazy(source.lazy(), "geometry", "polygon", ingest_batch_size=1)
        .lazy()
        .contains(0.75, 0.25)
        .select("id")
        .collect()
    )

    assert result["id"].to_list() == [1]


def test_lazy_point_ingestion_is_bounded_and_projection_aware(monkeypatch):
    source = _point_data()
    lazy_frame, projections = _tracked_source(source)
    batch_lengths: list[int] = []
    original = Engine._from_wkb_point_batches.__func__

    def tracked(cls, columns):
        # Record point batches before forwarding them to the native builder
        def inspected():
            for column in columns:
                batch_lengths.append(len(column))
                yield column

        return original(cls, inspected())

    monkeypatch.setattr(Engine, "_from_wkb_point_batches", classmethod(tracked))
    result = (
        SpatialFrame.from_lazy(lazy_frame, "geometry", "point", ingest_batch_size=2)
        .lazy()
        .range_query(0.5, -1, 3.5, 1)
        .select("id")
        .collect()
    )

    assert batch_lengths == [2, 2, 1]
    assert result["id"].to_list() == [2, 3, 4]
    assert set(projections[0]) == {"id", "geometry"}
    assert "unused" not in projections[0]


def test_lazy_point_ingestion_preserves_requested_wkb_and_order():
    source = _point_data()

    result = (
        SpatialFrame.from_lazy(source.lazy(), "geometry", "point", ingest_batch_size=2)
        .lazy()
        .knn(2.1, 0, 3)
        .select("id", "geometry")
        .collect()
    )

    assert result["id"].to_list() == [2, 3, 4]
    assert result["geometry"].to_list() == [
        source["geometry"][1],
        source["geometry"][2],
        source["geometry"][3],
    ]


def test_lazy_point_scalar_filter_uses_native_row_alignment():
    source = _point_data()
    result = (
        SpatialFrame.from_lazy(source.lazy(), "geometry", "point")
        .lazy()
        .filter(pl.col("value") >= 30)
        .range_query(1.5, -1, 3.5, 1)
        .select("id")
        .collect()
    )

    assert result["id"].to_list() == [3, 4]


def test_lazy_point_geographic_distance_system():
    source = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "geometry": [
                shapely.Point(0, 0).wkb,
                shapely.Point(0, 1).wkb,
                shapely.Point(0, 3).wkb,
            ],
        }
    )
    sf = SpatialFrame.from_lazy(
        source.lazy(),
        "geometry",
        "point",
        coordinate_system="geographic",
    )

    result = sf.lazy().within_distance_of_point(0, 0, 120_000).select("id").collect()

    assert result["id"].to_list() == [1, 2]


def test_scan_parquet_supports_lazy_points(tmp_path):
    source = _point_data()
    path = tmp_path / "points.parquet"
    source.write_parquet(path)

    result = (
        SpatialFrame.scan_parquet(path, "geometry", "point", parallel="none")
        .lazy()
        .range_query(-1, -1, 1.5, 1)
        .select("id")
        .collect()
    )

    assert result["id"].to_list() == [1, 2]
