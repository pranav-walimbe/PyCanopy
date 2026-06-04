"""Tests for Python-level input conversion helpers."""

import numpy as np
import pyarrow as pa
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402
from shapely.geometry import Point as ShapelyPoint  # noqa: E402

from pycanopy.engine import (  # noqa: E402
    _extract_polygon_rings,
    _geoarrow_to_numpy_xy,
    _to_numpy_xy,
)

XS = [0.0, 1.0, 2.0]
YS = [3.0, 4.0, 5.0]


# _to_numpy_xy: tuple list


def test_points_from_tuple_list():
    pairs = list(zip(XS, YS))
    xs, ys = _to_numpy_xy(pairs)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_points_from_single_pair():
    xs, ys = _to_numpy_xy([(7.0, 8.0)])
    assert xs.tolist() == [7.0]
    assert ys.tolist() == [8.0]


# _to_numpy_xy: numpy


def test_points_from_numpy_2d_array():
    arr = np.array([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    xs, ys = _to_numpy_xy(arr)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_points_from_numpy_wrong_shape_raises():
    with pytest.raises(ValueError):
        _to_numpy_xy(np.array([0.0, 1.0, 2.0]))


def test_points_from_numpy_returns_contiguous():
    arr = np.array([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    xs, ys = _to_numpy_xy(arr)
    assert xs.flags["C_CONTIGUOUS"]
    assert ys.flags["C_CONTIGUOUS"]


# _geoarrow_to_numpy_xy: struct encoding


def test_geoarrow_struct_named_xy():
    arr = pa.StructArray.from_arrays([pa.array(XS), pa.array(YS)], names=["x", "y"])
    xs, ys = _geoarrow_to_numpy_xy(arr)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_geoarrow_struct_positional():
    arr = pa.StructArray.from_arrays([pa.array(XS), pa.array(YS)], names=["lon", "lat"])
    xs, ys = _geoarrow_to_numpy_xy(arr)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


# _to_numpy_xy: GeoArrow dispatch


def test_points_from_geoarrow_struct():
    arr = pa.StructArray.from_arrays([pa.array(XS), pa.array(YS)], names=["x", "y"])
    xs, ys = _to_numpy_xy(arr)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_points_from_geoarrow_fixed_size_list():
    flat = [v for x, y in zip(XS, YS) for v in (x, y)]
    arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float64()), 2)
    xs, ys = _to_numpy_xy(arr)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_points_from_geoarrow_chunked_array():
    chunk1 = pa.StructArray.from_arrays(
        [pa.array([0.0, 1.0]), pa.array([3.0, 4.0])], names=["x", "y"]
    )
    chunk2 = pa.StructArray.from_arrays([pa.array([2.0]), pa.array([5.0])], names=["x", "y"])
    chunked = pa.chunked_array([chunk1, chunk2])
    xs, ys = _to_numpy_xy(chunked)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


# _extract_polygon_rings: basic extraction


def test_extract_polygon_rings_single_polygon():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    xs, ys, ring_offsets, poly_offsets = _extract_polygon_rings([poly])
    assert ring_offsets.tolist() == [0, 5]
    assert poly_offsets.tolist() == [0, 1]
    assert len(xs) == 5
    assert len(ys) == 5


def test_extract_polygon_rings_two_polygons_offsets():
    tri = Polygon([(0, 0), (1, 0), (0.5, 1)])
    sq = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    xs, _ys, ring_offsets, poly_offsets = _extract_polygon_rings([tri, sq])
    assert ring_offsets.tolist() == [0, 4, 9]
    assert poly_offsets.tolist() == [0, 1, 2]
    assert len(xs) == 9


def test_extract_polygon_rings_coordinates_correct():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    xs, ys, _ring_offsets, _poly_offsets = _extract_polygon_rings([poly])
    assert xs[0] == pytest.approx(0.0)
    assert ys[0] == pytest.approx(0.0)
    assert xs[1] == pytest.approx(1.0)
    assert ys[1] == pytest.approx(0.0)


def test_extract_polygon_rings_returns_contiguous_arrays():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    xs, ys, ring_offsets, poly_offsets = _extract_polygon_rings([poly])
    for arr in (xs, ys, ring_offsets, poly_offsets):
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.dtype == np.float64 or arr.dtype == np.int64
    assert xs.dtype == np.float64
    assert ys.dtype == np.float64
    assert ring_offsets.dtype == np.int64
    assert poly_offsets.dtype == np.int64


def test_extract_polygon_rings_geoseries():
    gpd = pytest.importorskip("geopandas")
    gs = gpd.GeoSeries([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    _xs, _ys, ring_offsets, poly_offsets = _extract_polygon_rings(gs)
    assert len(ring_offsets) == 2
    assert len(poly_offsets) == 2


def test_extract_polygon_rings_with_hole():
    # Square with a smaller square hole: 2 rings, 1 polygon
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (3, 1), (3, 3), (1, 3)]
    poly = Polygon(outer, [hole])
    xs, _ys, ring_offsets, poly_offsets = _extract_polygon_rings([poly])
    assert poly_offsets.tolist() == [0, 2]  # 1 polygon, 2 rings
    assert len(ring_offsets) == 3  # 2 rings + sentinel
    assert ring_offsets[0] == 0
    assert ring_offsets[2] == len(xs)  # all coords accounted for


def test_extract_polygon_rings_rejects_multipolygon():
    mp = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 0), (3, 0), (3, 1)])])
    with pytest.raises(TypeError, match="MultiPolygon"):
        _extract_polygon_rings([mp])


def test_extract_polygon_rings_rejects_point():
    with pytest.raises(TypeError, match="not a Polygon"):
        _extract_polygon_rings([ShapelyPoint(0.0, 0.0)])


def test_extract_polygon_rings_error_reports_index():
    valid = Polygon([(0, 0), (1, 0), (1, 1)])
    mp = MultiPolygon([Polygon([(2, 0), (3, 0), (3, 1)]), Polygon([(4, 0), (5, 0), (5, 1)])])
    with pytest.raises(TypeError, match="index 1"):
        _extract_polygon_rings([valid, mp])
