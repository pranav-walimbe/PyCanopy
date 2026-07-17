"""
Integration tests for SpatialFrame / SpatialLazyFrame.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest
import shapely

from pycanopy import PyCanopyCoordinateWarning, SpatialFrame
from pycanopy.nodes import PluginPath
from pycanopy.optimizer import SpatialOptimizer

# fixtures


# 5-point dataset: (0,0),(1,0),(2,0),(0,1),(1,1) with values 10-50.
# Extent (0,0)-(2,1), area=2. Moderate queries have selectivity >= 0.05 → EXPR path.
@pytest.fixture(scope="session")
def sf():
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 0.0, 1.0],
            "y": [0.0, 0.0, 0.0, 1.0, 1.0],
            "v": [10, 20, 30, 40, 50],
        }
    )
    return SpatialFrame(df, "x", "y")


# 100-point uniform grid: xs=0..9, ys=0..9 (i%10, i//10).
# contains selectivity = 1/100 = 0.01 < 0.05 → IO path.
# Tight range over extent (0,0)-(9,9) also falls below threshold.
@pytest.fixture(scope="session")
def sf_large():
    xs = [(i % 10) * 1.0 for i in range(100)]
    ys = [(i // 10) * 1.0 for i in range(100)]
    df = pl.DataFrame({"x": xs, "y": ys, "v": list(range(100))})
    return SpatialFrame(df, "x", "y")


# EXPR path correctness


def test_range_returns_matching_rows(sf):
    result = sf.lazy().range_query(0.0, 0.0, 1.5, 0.5).collect()
    assert sorted(result["v"].to_list()) == [10, 20]


def test_range_empty_bbox_returns_no_rows(sf):
    result = sf.lazy().range_query(5.0, 5.0, 10.0, 10.0).collect()
    assert result.is_empty()


def test_contains_returns_exact_match(sf):
    result = sf.lazy().contains(1.0, 0.0).collect()
    assert result["v"].to_list() == [20]


def test_contains_no_match_returns_empty(sf):
    result = sf.lazy().contains(0.5, 0.5).collect()
    assert result.is_empty()


def test_knn_returns_k_nearest(sf):
    # query (1.2, 0.1): nearest are (1,0)=v20 and (2,0)=v30
    result = sf.lazy().knn(1.2, 0.1, 2).collect()
    assert sorted(result["v"].to_list()) == [20, 30]


def test_knn_k_larger_than_n_returns_all(sf):
    result = sf.lazy().knn(0.0, 0.0, 100).collect()
    assert sorted(result["v"].to_list()) == [10, 20, 30, 40, 50]


# scalar + spatial ordering


def test_scalar_before_range_filters_correctly(sf):
    # scalar: v > 15 keeps (1,0)v20 (2,0)v30 (0,1)v40 (1,1)v50
    # range (0,0)-(1.5,0.5): keeps x<=1.5, y<=0.5 → only (1,0)v20
    result = sf.lazy().filter(pl.col("v") > 15).range_query(0.0, 0.0, 1.5, 0.5).collect()
    assert result["v"].to_list() == [20]


def test_range_declared_before_scalar_gives_same_result(sf):
    # Optimizer reorders scalar before spatial; result must match.
    result = sf.lazy().range_query(0.0, 0.0, 1.5, 0.5).filter(pl.col("v") > 15).collect()
    assert result["v"].to_list() == [20]


def test_no_predicates_returns_all_rows(sf):
    result = sf.lazy().collect()
    assert sorted(result["v"].to_list()) == [10, 20, 30, 40, 50]


# chained spatial predicates


def test_two_range_queries_intersect(sf):
    # First range: all 5 points. Second range: x in [0.5,2.5], y in [-0.1,0.5] → (1,0) and (2,0).
    result = sf.lazy().range_query(0.0, 0.0, 2.0, 1.0).range_query(0.5, -0.1, 2.5, 0.5).collect()
    assert sorted(result["v"].to_list()) == [20, 30]


# IO path correctness


def test_io_contains_returns_correct_row(sf_large):
    # selectivity = 1/100 = 0.01 < 0.05 → IO path.
    # point at (3,4): index = 3 + 4*10 = 43, v=43.
    result = sf_large.lazy().contains(3.0, 4.0).collect()
    assert result["v"].to_list() == [43]


def test_io_range_returns_correct_row(sf_large):
    # bbox (0,0)-(0.5,0.5): area 0.25 vs total (0-9)^2=81 → selectivity ~0.003 → IO path.
    # Only point (0,0) = index 0, v=0.
    result = sf_large.lazy().range_query(0.0, 0.0, 0.5, 0.5).collect()
    assert result["v"].to_list() == [0]


def test_io_scalar_applied_to_candidates(sf_large):
    # IO path slices to spatial candidates, then applies scalar filter.
    result = sf_large.lazy().contains(3.0, 4.0).filter(pl.col("v") > 50).collect()
    assert result.is_empty()


def test_io_two_contains_intersect(sf_large):
    # Two contains predicates on the same point: still returns that point.
    result = sf_large.lazy().contains(3.0, 4.0).contains(3.0, 4.0).collect()
    assert result["v"].to_list() == [43]


def test_io_disjoint_ranges_return_empty(sf_large):
    # First range has a candidate; second is disjoint → intersection is empty.
    result = (
        sf_large.lazy().range_query(0.0, 0.0, 0.5, 0.5).range_query(5.0, 5.0, 9.5, 9.5).collect()
    )
    assert result.is_empty()


# plugin path selection


def test_path_select_expr_for_moderate_selectivity(sf):
    # range selectivity = (1.5*0.5) / (2*1) = 0.375 > 0.05 → EXPR.
    plan = sf.lazy().range_query(0.0, 0.0, 1.5, 0.5)._plan
    opt = SpatialOptimizer()
    optimized = opt.optimize(plan, sf.engine)
    assert opt._select_plugin_path(optimized, sf.engine) == PluginPath.EXPR


def test_path_select_io_for_tight_selectivity(sf_large):
    # contains selectivity = 1/100 = 0.01 < 0.05 → IO.
    plan = sf_large.lazy().contains(3.0, 4.0)._plan
    opt = SpatialOptimizer()
    optimized = opt.optimize(plan, sf_large.engine)
    assert opt._select_plugin_path(optimized, sf_large.engine) == PluginPath.IO


def test_path_select_expr_when_knn_present(sf_large):
    # KNN node forces EXPR regardless of spatial selectivity.
    plan = sf_large.lazy().contains(3.0, 4.0).knn(3.0, 4.0, 1)._plan
    opt = SpatialOptimizer()
    optimized = opt.optimize(plan, sf_large.engine)
    assert opt._select_plugin_path(optimized, sf_large.engine) == PluginPath.EXPR


def test_path_select_expr_when_knn_join_present(sf_large):
    query_df = pl.DataFrame({"qx": [3.0], "qy": [4.0]})
    plan = sf_large.lazy().knn_join(query_df, "qx", "qy", k=1)._plan
    opt = SpatialOptimizer()
    optimized = opt.optimize(plan, sf_large.engine)
    assert opt._select_plugin_path(optimized, sf_large.engine) == PluginPath.EXPR


# join nodes


def test_knn_join_returns_k_rows_per_query(sf):
    query_df = pl.DataFrame({"qx": [1.2], "qy": [0.1]})
    result = sf.lazy().knn_join(query_df, "qx", "qy", k=2).collect()
    assert len(result) == 2
    assert sorted(result["v"].to_list()) == [20, 30]


def test_knn_join_multiple_queries(sf):
    query_df = pl.DataFrame({"qx": [1.2, 0.1], "qy": [0.1, 0.1]})
    result = sf.lazy().knn_join(query_df, "qx", "qy", k=1).collect()
    # Each query returns 1 nearest: (1.2,0.1)→(1,0), (0.1,0.1)→(0,0)
    assert len(result) == 2
    assert sorted(result["v"].to_list()) == [10, 20]


# SpatialFrame.from_wkb_points


def _wkb_point_frame():
    pts = [
        shapely.Point(x, y).wkb
        for x, y in zip([0.0, 1.0, 2.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0])
    ]
    return pl.DataFrame({"v": [10, 20, 30, 40, 50], "geom": pts})


def test_from_wkb_points_appends_coords_and_keeps_columns():
    sf = SpatialFrame.from_wkb_points(_wkb_point_frame(), "geom")
    assert "_x" in sf.df.columns and "_y" in sf.df.columns
    assert sf.df["v"].to_list() == [10, 20, 30, 40, 50]
    assert sf.df["_x"].to_list() == pytest.approx([0.0, 1.0, 2.0, 0.0, 1.0])


def test_from_wkb_points_frame_answers_spatial_query():
    sf = SpatialFrame.from_wkb_points(_wkb_point_frame(), "geom")
    result = sf.lazy().range_query(0.0, 0.0, 1.5, 0.5).collect()
    assert sorted(result["v"].to_list()) == [10, 20]


def test_from_wkb_points_custom_coord_names():
    sf = SpatialFrame.from_wkb_points(_wkb_point_frame(), "geom", x_col="lon", y_col="lat")
    assert sf.x_col == "lon" and sf.y_col == "lat"
    assert "lon" in sf.df.columns and "lat" in sf.df.columns


def test_from_wkb_points_missing_column_raises():
    with pytest.raises(ValueError, match="wkb_col"):
        SpatialFrame.from_wkb_points(pl.DataFrame({"v": [1]}), "geom")


# range_filter tests


def _wkb_polygon_frame():
    # Two non-overlapping unit squares: one near origin, one far away
    boxes = [shapely.box(0, 0, 1, 1), shapely.box(10, 10, 11, 11)]
    return pl.DataFrame({"id": [1, 2], "geom": [b.wkb for b in boxes]})


def test_range_filter_point_frame_returns_spatial_frame(sf):
    result = sf.range_filter(0.0, 0.0, 1.5, 0.5)
    assert isinstance(result, SpatialFrame)
    assert sorted(result.df["v"].to_list()) == [10, 20]


def test_range_filter_polygon_frame_returns_matching_polygon():
    sf = SpatialFrame.from_wkb_polygons(_wkb_polygon_frame(), "geom")
    result = sf.range_filter(0.0, 0.0, 2.0, 2.0)
    assert isinstance(result, SpatialFrame)
    assert result.df["id"].to_list() == [1]
    assert result.engine.n == 1


def test_range_filter_polygon_frame_empty_bbox():
    sf = SpatialFrame.from_wkb_polygons(_wkb_polygon_frame(), "geom")
    result = sf.range_filter(5.0, 5.0, 6.0, 6.0)
    assert result.engine.n == 0


# radius_query tests


def test_radius_query_returns_matching_rows(sf):
    # Center (0,0) radius 1.0 keeps (0,0),(1,0),(0,1); (1,1) is dropped by the circle refine
    result = sf.radius_query(0.0, 0.0, 1.0)
    assert sorted(result["v"].to_list()) == [10, 20, 40]


def test_within_distance_of_point_lazy_matches_eager(sf):
    result = sf.lazy().within_distance_of_point(0.0, 0.0, 1.0).collect()
    assert sorted(result["v"].to_list()) == [10, 20, 40]


# coordinate_system tests

# Real airports with published great-circle distances from JFK: LAX 3974 km,
# SFO 4152 km, LHR 5540 km, CDG 5834 km.
_JFK = (-73.7781, 40.6413)


@pytest.fixture(scope="session")
def sf_airports():
    df = pl.DataFrame(
        {
            "name": ["JFK", "LAX", "SFO", "LHR", "CDG"],
            "lon": [-73.7781, -118.4085, -122.3790, -0.4543, 2.5479],
            "lat": [40.6413, 33.9416, 37.6213, 51.4700, 49.0097],
        }
    )
    return SpatialFrame(df, "lon", "lat", coordinate_system="geographic")


def test_coordinate_system_defaults_to_planar(sf):
    assert sf.coordinate_system == "planar"


def test_coordinate_system_geographic_is_reported(sf_airports):
    assert sf_airports.coordinate_system == "geographic"


def test_coordinate_system_rejects_unknown_value():
    df = pl.DataFrame({"x": [0.0], "y": [0.0]})
    with pytest.raises(ValueError, match="planar"):
        SpatialFrame(df, "x", "y", coordinate_system="wgs84")


def test_geographic_radius_query_measures_meters(sf_airports):
    # 4000 km reaches LAX (3974) but not SFO (4152), so the threshold lands between them
    result = sf_airports.radius_query(*_JFK, 4_000_000)
    assert sorted(result["name"].to_list()) == ["JFK", "LAX"]


def test_geographic_radius_query_wider_threshold(sf_airports):
    result = sf_airports.radius_query(*_JFK, 5_600_000)
    assert sorted(result["name"].to_list()) == ["JFK", "LAX", "LHR", "SFO"]


def test_geographic_within_distance_of_point_lazy_matches_eager(sf_airports):
    result = sf_airports.lazy().within_distance_of_point(*_JFK, 4_000_000).collect()
    assert sorted(result["name"].to_list()) == ["JFK", "LAX"]


def test_planar_frame_reads_the_same_distance_as_degrees(sf_airports):
    # The identical call on a planar frame measures degrees, so 4e6 degrees spans the globe
    df = sf_airports.df
    planar = SpatialFrame(df, "lon", "lat")
    result = planar.radius_query(*_JFK, 4_000_000)
    assert len(result) == 5


def test_geographic_survives_range_filter(sf_airports):
    # A derived frame keeps the setting, since it is a fact about the coordinates
    derived = sf_airports.range_filter(-180.0, -90.0, 180.0, 90.0)
    assert derived.coordinate_system == "geographic"


def test_geographic_radius_query_across_the_antimeridian():
    # Suva 178.44E and Apia 171.75W are 1152 km apart across the +/-180 seam
    df = pl.DataFrame({"name": ["Apia"], "lon": [-171.7513], "lat": [-13.8333]})
    sf = SpatialFrame(df, "lon", "lat", coordinate_system="geographic")
    assert len(sf.radius_query(178.4419, -18.1416, 1_000_000)) == 0
    assert len(sf.radius_query(178.4419, -18.1416, 1_500_000)) == 1


def test_geographic_warns_on_projected_coordinates():
    # UTM easting/northing declared geographic is always a mistake, degrees cannot reach 500000
    df = pl.DataFrame({"x": [500_000.0, 501_000.0], "y": [4_649_000.0, 4_650_000.0]})
    with pytest.warns(PyCanopyCoordinateWarning, match="lon/lat"):
        SpatialFrame(df, "x", "y", coordinate_system="geographic")


def test_geographic_warns_on_0_360_longitudes():
    # The 0..360 convention puts true neighbours either side of the seam out of each other's box
    df = pl.DataFrame({"x": [350.0, 5.0], "y": [10.0, 10.0]})
    with pytest.warns(PyCanopyCoordinateWarning, match="0..360"):
        SpatialFrame(df, "x", "y", coordinate_system="geographic")


def test_geographic_does_not_warn_on_lon_lat(sf_airports):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpatialFrame(sf_airports.df, "lon", "lat", coordinate_system="geographic")


def test_planar_never_warns_whatever_the_coordinates_look_like(sf_airports):
    # A small planar grid sits inside lon/lat's range, so guessing from the data would misfire
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpatialFrame(sf_airports.df, "lon", "lat")
        SpatialFrame(pl.DataFrame({"x": [0.0, 9.0], "y": [0.0, 9.0]}), "x", "y")


def test_empty_geographic_frame_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        empty = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64})
        SpatialFrame(empty, "x", "y", coordinate_system="geographic")
