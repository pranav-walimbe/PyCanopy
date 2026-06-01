"""Tests for the _extract_points and _from_geoarrow conversion helpers."""

import pytest

from pycanopy.engine import _extract_points, _from_geoarrow

XS = [0.0, 1.0, 2.0]
YS = [3.0, 4.0, 5.0]


# _extract_points: tuple list


def test_extract_tuple_list():
    pairs = list(zip(XS, YS))
    xs, ys = _extract_points(pairs)
    assert xs == XS
    assert ys == YS


def test_extract_single_pair():
    xs, ys = _extract_points([(7.0, 8.0)])
    assert xs == [7.0]
    assert ys == [8.0]


# _extract_points: numpy


def test_extract_numpy_2d_array():
    np = pytest.importorskip("numpy")
    arr = np.array([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    xs, ys = _extract_points(arr)
    assert xs == pytest.approx(XS)
    assert ys == pytest.approx(YS)


def test_extract_numpy_wrong_shape_raises():
    np = pytest.importorskip("numpy")
    arr = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        _extract_points(arr)


# _from_geoarrow: struct encoding


def test_geoarrow_struct_named_xy():
    pa = pytest.importorskip("pyarrow")
    arr = pa.StructArray.from_arrays(
        [pa.array(XS), pa.array(YS)],
        names=["x", "y"],
    )
    xs, ys = _from_geoarrow(arr)
    assert xs == pytest.approx(XS)
    assert ys == pytest.approx(YS)


def test_geoarrow_struct_positional():
    pa = pytest.importorskip("pyarrow")
    arr = pa.StructArray.from_arrays(
        [pa.array(XS), pa.array(YS)],
        names=["lon", "lat"],
    )
    xs, ys = _from_geoarrow(arr)
    assert xs == pytest.approx(XS)
    assert ys == pytest.approx(YS)


# _from_geoarrow: interleaved FixedSizeList


def test_geoarrow_fixed_size_list():
    pa = pytest.importorskip("pyarrow")
    flat = []
    for x, y in zip(XS, YS):
        flat.extend([x, y])
    arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float64()), 2)
    xs, ys = _from_geoarrow(arr)
    assert xs == pytest.approx(XS)
    assert ys == pytest.approx(YS)


# _from_geoarrow: chunked array is flattened automatically


def test_geoarrow_chunked_array():
    pa = pytest.importorskip("pyarrow")
    chunk1 = pa.StructArray.from_arrays(
        [pa.array([0.0, 1.0]), pa.array([3.0, 4.0])],
        names=["x", "y"],
    )
    chunk2 = pa.StructArray.from_arrays(
        [pa.array([2.0]), pa.array([5.0])],
        names=["x", "y"],
    )
    chunked = pa.chunked_array([chunk1, chunk2])
    xs, ys = _from_geoarrow(chunked)
    assert xs == pytest.approx(XS)
    assert ys == pytest.approx(YS)
