"""
Tests for the Python Engine wrapper (pycanopy.Engine).
"""

import numpy as np
import pyarrow as pa
import pytest
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import Point as ShapelyPoint

from pycanopy import Engine

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


# hole_engine: one outer square (0,0)-(4,4) with an inner square hole (1,1)-(3,3)
@pytest.fixture(scope="session")
def hole_engine():
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (3, 1), (3, 3), (1, 3)]
    return Engine.from_polygons([Polygon(outer, [hole])])


@pytest.fixture(scope="session")
def large_engine():
    xs = [(i % 50) * 2.0 for i in range(1000)]
    ys = [(i // 50) * 2.0 for i in range(1000)]
    return Engine.from_coords(xs, ys)


@pytest.fixture(scope="session")
def numpy_engine():
    arr = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    return Engine(arr)


@pytest.fixture(scope="session")
def pyarrow_struct_engine():
    arr = pa.StructArray.from_arrays([pa.array(XS), pa.array(YS)], names=["x", "y"])
    return Engine(arr)


@pytest.fixture(scope="session")
def pyarrow_fsl_engine():
    flat = [v for x, y in zip(XS, YS) for v in (x, y)]
    arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float64()), 2)
    return Engine(arr)


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


# Engine.from_wkb_points


def _wkb_points(xs, ys, type_=pa.binary()):
    return pa.array([ShapelyPoint(x, y).wkb for x, y in zip(xs, ys)], type=type_)


def test_from_wkb_points_builds_queryable_index():
    eng = Engine.from_wkb_points(_wkb_points(XS, YS))
    assert eng.n == 5
    assert eng.knn(1.2, 0.1, 1) == [1]
    assert sorted(eng.range_query(0.0, 0.0, 1.5, 0.5)) == [0, 1]


def test_from_wkb_points_matches_from_coords():
    via_wkb = Engine.from_wkb_points(_wkb_points(XS, YS)).knn(1.2, 0.1, 3)
    via_coords = Engine.from_coords(XS, YS).knn(1.2, 0.1, 3)
    assert via_wkb == via_coords == [1, 2, 4]


def test_from_wkb_points_accepts_large_binary():
    eng = Engine.from_wkb_points(_wkb_points(XS, YS, type_=pa.large_binary()))
    assert eng.knn(1.2, 0.1, 1) == [1]


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


# range_query (points)


def test_range_returns_correct_points(engine):
    assert sorted(engine.range_query(0.0, 0.0, 1.5, 0.5)) == [0, 1]


def test_range_single_result(engine):
    assert engine.range_query(0.5, 0.5, 1.5, 1.5) == [4]


def test_range_all_points(engine):
    assert sorted(engine.range_query(0.0, 0.0, 2.0, 1.0)) == [0, 1, 2, 3, 4]


def test_range_empty_returns_empty(engine):
    assert engine.range_query(5.0, 5.0, 10.0, 10.0) == []


# radius_query (points)


def test_radius_refines_to_circle(engine):
    # (0,0),(1,0),(0,1) are within 1.0; (1,1) is in the bbox but sqrt(2) away, so dropped
    assert sorted(engine.radius_query(0.0, 0.0, 1.0).tolist()) == [0, 1, 3]


# contains (points)


def test_contains_exact_point_match(engine):
    assert engine.contains(1.0, 0.0) == [1]


def test_contains_no_match_returns_empty(engine):
    assert engine.contains(0.5, 0.5) == []


# alternative input formats


def test_from_numpy_array(numpy_engine):
    assert sorted(numpy_engine.knn(1.2, 0.0, 1)) == [1]


def test_from_pyarrow_struct_array(pyarrow_struct_engine):
    assert sorted(pyarrow_struct_engine.knn(1.2, 0.1, 1)) == [1]


def test_from_pyarrow_fixed_size_list(pyarrow_fsl_engine):
    assert sorted(pyarrow_fsl_engine.knn(1.2, 0.1, 1)) == [1]


# large point dataset — exercises index selection past the brute-force threshold


def test_large_dataset_knn(large_engine):
    assert len(large_engine.knn(25.0, 10.0, 5)) == 5


def test_large_dataset_range(large_engine):
    assert len(large_engine.range_query(0.0, 0.0, 10.0, 10.0)) > 0


# polygon construction


def test_from_polygons_creates_engine():
    eng = Engine.from_polygons(SQUARES)
    assert eng is not None


def test_from_polygons_repr_contains_n(poly_engine):
    assert "n=5" in repr(poly_engine)


def test_from_polygons_stats_contains_n(poly_engine):
    assert "n=5" in poly_engine.stats()


def test_from_polygons_accepts_multipolygon():
    # SQUARES[1] and SQUARES[2] are disjoint; as one MultiPolygon they are one logical
    # polygon, so a point in either part returns the same index, counted once.
    mp = MultiPolygon([SQUARES[1], SQUARES[2]])
    eng = Engine.from_polygons([SQUARES[0], mp])
    assert eng.contains(0.5, 0.5) == [0]  # in the plain polygon
    assert eng.contains(2.5, 0.5) == [1]  # in the first part of the MultiPolygon
    assert eng.contains(4.5, 0.5) == [1]  # in the second part -> same logical polygon
    assert eng.contains(1.5, 0.5) == []  # between the parts -> no match


def test_multipolygon_area_sums_parts():
    # MultiPolygon area is the sum of its parts (two unit squares -> 2.0).
    mp = MultiPolygon([SQUARES[1], SQUARES[2]])
    eng = Engine.from_polygons([SQUARES[0], mp])
    areas = eng.polygon_areas()
    assert len(areas) == 2
    assert areas[0] == pytest.approx(1.0)
    assert areas[1] == pytest.approx(2.0)


def test_multipolygon_range_query_maps_to_logical():
    mp = MultiPolygon([SQUARES[1], SQUARES[2]])
    eng = Engine.from_polygons([SQUARES[0], mp])
    assert eng.range_query(4.0, 0.0, 5.0, 1.0) == [1]
    assert sorted(eng.range_query(2.0, 0.0, 5.0, 1.0)) == [1]


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


# polygon holes


def test_polygon_hole_excludes_point_in_hole(hole_engine):
    # (2.0, 2.0) is inside the hole — not contained
    assert hole_engine.contains(2.0, 2.0) == []


def test_polygon_hole_contains_point_outside_hole(hole_engine):
    # (0.5, 0.5) is inside the outer ring but outside the hole
    assert hole_engine.contains(0.5, 0.5) == [0]


def test_polygon_hole_range_finds_polygon(hole_engine):
    # range overlapping the outer MBR returns the polygon regardless of hole
    assert hole_engine.range_query(0.0, 0.0, 2.0, 2.0) == [0]
