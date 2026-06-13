"""Tests for Python-level input conversion helpers."""

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import Point as ShapelyPoint

from pycanopy.engine import (
    _extract_polygon_rings,
    _geoarrow_to_numpy_xy,
    _to_numpy_xy,
    _wkb_points_fast,
    wkb_points_to_xy,
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


# wkb_points_to_xy: WKB point decoding


def _wkb_point_array(xs, ys):
    """Return a pyarrow binary array of standard little-endian WKB points."""
    return pa.array([ShapelyPoint(x, y).wkb for x, y in zip(xs, ys)], type=pa.binary())


def test_wkb_points_fast_path_decodes_xy():
    arr = _wkb_point_array(XS, YS)
    xs, ys = wkb_points_to_xy(arr)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_wkb_points_fast_path_taken_for_standard_points():
    # _wkb_points_fast returns None when it declines; standard 2D points must take it.
    arr = _wkb_point_array(XS, YS)
    assert _wkb_points_fast(arr) is not None


def test_wkb_points_returns_contiguous_float64():
    xs, ys = wkb_points_to_xy(_wkb_point_array(XS, YS))
    assert xs.flags["C_CONTIGUOUS"] and xs.dtype == np.float64
    assert ys.flags["C_CONTIGUOUS"] and ys.dtype == np.float64


def test_wkb_points_empty_column():
    xs, ys = wkb_points_to_xy(pa.array([], type=pa.binary()))
    assert xs.tolist() == [] and ys.tolist() == []


def test_wkb_points_sliced_array_honours_offset():
    # A zero-copy pyarrow slice shares the parent buffers; decoding must respect offset.
    arr = _wkb_point_array([0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0])
    xs, ys = wkb_points_to_xy(arr.slice(2, 2))
    assert xs.tolist() == pytest.approx([2.0, 3.0])
    assert ys.tolist() == pytest.approx([7.0, 8.0])


def test_wkb_points_large_binary_decodes():
    # polars emits large_binary (int64 offsets); the fast path must handle it.
    arr = pa.array([ShapelyPoint(x, y).wkb for x, y in zip(XS, YS)], type=pa.large_binary())
    xs, ys = wkb_points_to_xy(arr)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_wkb_points_fallback_matches_fast_path_for_big_endian():
    # Big-endian WKB declines the fast path and must still decode via shapely.
    geoms = [ShapelyPoint(x, y) for x, y in zip(XS, YS)]
    big_endian = pa.array([shapely.to_wkb(g, byte_order=0) for g in geoms], type=pa.binary())
    assert _wkb_points_fast(big_endian) is None
    xs, ys = wkb_points_to_xy(big_endian)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_wkb_points_fallback_for_nulls():
    arr = pa.array([ShapelyPoint(0.0, 3.0).wkb, None, ShapelyPoint(2.0, 5.0).wkb], type=pa.binary())
    assert _wkb_points_fast(arr) is None
    xs, _ys = wkb_points_to_xy(arr)
    assert xs[0] == pytest.approx(0.0)
    assert xs[2] == pytest.approx(2.0)


def test_wkb_points_chunked_array():
    c1 = _wkb_point_array([0.0, 1.0], [3.0, 4.0])
    c2 = _wkb_point_array([2.0], [5.0])
    xs, ys = wkb_points_to_xy(pa.chunked_array([c1, c2]))
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_wkb_points_from_polars_series():
    series = pl.Series("geom", [ShapelyPoint(x, y).wkb for x, y in zip(XS, YS)], dtype=pl.Binary)
    xs, ys = wkb_points_to_xy(series)
    assert xs.tolist() == pytest.approx(XS)
    assert ys.tolist() == pytest.approx(YS)


def test_to_numpy_xy_routes_wkb_binary():
    arr = _wkb_point_array(XS, YS)
    xs, ys = _to_numpy_xy(arr)
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
