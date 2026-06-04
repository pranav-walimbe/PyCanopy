"""Tests for the Python Engine wrapper (pycanopy.Engine)."""

import numpy as np
import pyarrow as pa
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402
from shapely.geometry import Point as ShapelyPoint  # noqa: E402

from pycanopy import Engine  # noqa: E402

# Point fixture:
# Index 0=(0,0)  1=(1,0)  2=(2,0)  3=(0,1)  4=(1,1)
# Query (1.2, 0.1): distance² order → 1, 2, 4, 0, 3
XS = [0.0, 1.0, 2.0, 0.0, 1.0]
YS = [0.0, 0.0, 0.0, 1.0, 1.0]

# Polygon fixture: five non-overlapping unit squares
#
#   3=(0,2)-(1,3)   4=(2,2)-(3,3)
#   0=(0,0)-(1,1)   1=(2,0)-(3,1)   2=(4,0)-(5,1)
SQUARES = [
    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
    Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
    Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
]


@pytest.fixture(scope="session")
def engine():
    return Engine.from_coords(XS, YS)


@pytest.fixture(scope="session")
def poly_engine():
    return Engine.from_polygons(SQUARES)


@pytest.fixture(scope="session")
def large_poly_engine():
    polys = [Polygon([(i, 0), (i + 0.9, 0), (i + 0.9, 0.9), (i, 0.9)]) for i in range(600)]
    return Engine.from_polygons(polys)


# point construction


def test_from_coords_creates_engine():
    eng = Engine.from_coords(XS, YS)
    assert eng is not None


def test_from_tuple_list_creates_engine():
    eng = Engine(list(zip(XS, YS)))
    assert eng is not None


def test_from_coords_mismatched_lengths_raises():
    with pytest.raises(Exception):
        Engine.from_coords([0.0, 1.0], [0.0])


def test_repr_contains_n(engine):
    assert "n=5" in repr(engine)


def test_stats_returns_string(engine):
    s = engine.stats()
    assert isinstance(s, str)
    assert "n=5" in s


# knn


def test_knn_k1_returns_closest(engine):
    assert engine.knn(1.2, 0.1, 1) == [1]


def test_knn_k2_returns_two_closest(engine):
    assert sorted(engine.knn(1.2, 0.1, 2)) == [1, 2]


def test_knn_k3_returns_three_closest(engine):
    assert sorted(engine.knn(1.2, 0.1, 3)) == [1, 2, 4]


def test_knn_k_larger_than_n_returns_all(engine):
    assert sorted(engine.knn(0.0, 0.0, 100)) == [0, 1, 2, 3, 4]


def test_knn_at_exact_point_returns_that_point(engine):
    assert engine.knn(1.0, 0.0, 1) == [1]


def test_knn_approximate_flag_accepted(engine):
    assert len(engine.knn(1.2, 0.1, 2, approximate=True)) == 2


# range_query (points)


def test_range_returns_correct_points(engine):
    assert sorted(engine.range_query(0.0, 0.0, 1.5, 0.5)) == [0, 1]


def test_range_single_result(engine):
    assert engine.range_query(0.5, 0.5, 1.5, 1.5) == [4]


def test_range_all_points(engine):
    assert sorted(engine.range_query(0.0, 0.0, 2.0, 1.0)) == [0, 1, 2, 3, 4]


def test_range_empty_returns_empty(engine):
    assert engine.range_query(5.0, 5.0, 10.0, 10.0) == []


# contains (points)


def test_contains_exact_point_match(engine):
    assert engine.contains(1.0, 0.0) == [1]


def test_contains_no_match_returns_empty(engine):
    assert engine.contains(0.5, 0.5) == []


# alternative input formats


def test_from_numpy_array():
    arr = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    eng = Engine(arr)
    assert sorted(eng.knn(1.2, 0.0, 1)) == [1]


def test_from_pyarrow_struct_array():
    arr = pa.StructArray.from_arrays([pa.array(XS), pa.array(YS)], names=["x", "y"])
    eng = Engine(arr)
    assert sorted(eng.knn(1.2, 0.1, 1)) == [1]


def test_from_pyarrow_fixed_size_list():
    flat = [v for x, y in zip(XS, YS) for v in (x, y)]
    arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float64()), 2)
    eng = Engine(arr)
    assert sorted(eng.knn(1.2, 0.1, 1)) == [1]


# large point dataset — exercises index selection past the brute-force threshold


def test_large_dataset_knn():
    xs = [(i % 50) * 2.0 for i in range(1000)]
    ys = [(i // 50) * 2.0 for i in range(1000)]
    eng = Engine.from_coords(xs, ys)
    assert len(eng.knn(25.0, 10.0, 5)) == 5


def test_large_dataset_range():
    xs = [(i % 50) * 2.0 for i in range(1000)]
    ys = [(i // 50) * 2.0 for i in range(1000)]
    eng = Engine.from_coords(xs, ys)
    assert len(eng.range_query(0.0, 0.0, 10.0, 10.0)) > 0


# polygon construction


def test_from_polygons_creates_engine():
    eng = Engine.from_polygons(SQUARES)
    assert eng is not None


def test_from_polygons_repr_contains_n(poly_engine):
    assert "n=5" in repr(poly_engine)


def test_from_polygons_stats_contains_n(poly_engine):
    assert "n=5" in poly_engine.stats()


def test_from_polygons_geoseries():
    gpd = pytest.importorskip("geopandas")
    gs = gpd.GeoSeries(SQUARES)
    eng = Engine.from_polygons(gs)
    assert "n=5" in repr(eng)


def test_from_polygons_rejects_multipolygon():
    mp = MultiPolygon([SQUARES[0], SQUARES[1]])
    with pytest.raises(TypeError, match="MultiPolygon"):
        Engine.from_polygons([mp])


def test_from_polygons_rejects_non_polygon():
    with pytest.raises(TypeError, match="not a Polygon"):
        Engine.from_polygons([ShapelyPoint(0.0, 0.0)])


# polygon range queries


def test_polygon_range_intersecting(poly_engine):
    # bbox (0,0)-(3,3) intersects squares 0,1,3,4 — misses square 2 at (4,0)-(5,1)
    assert sorted(poly_engine.range_query(0.0, 0.0, 3.0, 3.0)) == [0, 1, 3, 4]


def test_polygon_range_single(poly_engine):
    assert sorted(poly_engine.range_query(0.0, 0.0, 1.0, 1.0)) == [0]


def test_polygon_range_all(poly_engine):
    assert sorted(poly_engine.range_query(0.0, 0.0, 5.0, 3.0)) == [0, 1, 2, 3, 4]


def test_polygon_range_empty(poly_engine):
    assert poly_engine.range_query(10.0, 10.0, 20.0, 20.0) == []


# polygon contains queries


def test_polygon_contains_first_square(poly_engine):
    assert poly_engine.contains(0.5, 0.5) == [0]


def test_polygon_contains_second_square(poly_engine):
    assert poly_engine.contains(2.5, 0.5) == [1]


def test_polygon_contains_gap_returns_empty(poly_engine):
    # (1.5, 0.5) falls in the gap between squares 0 and 1
    assert poly_engine.contains(1.5, 0.5) == []


def test_polygon_contains_outside_returns_empty(poly_engine):
    assert poly_engine.contains(10.0, 10.0) == []


# large polygon dataset — N > 500 exercises the R-tree path


def test_large_polygon_dataset_range(large_poly_engine):
    assert len(large_poly_engine.range_query(0.0, 0.0, 9.5, 1.0)) == 10


def test_large_polygon_dataset_contains(large_poly_engine):
    assert large_poly_engine.contains(5.5, 0.5) == [5]
