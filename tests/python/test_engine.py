"""Tests for the Python Engine wrapper (pycanopy.Engine)."""

import pytest

from pycanopy import Engine

# Shared five-point fixture:
# Index 0=(0,0)  1=(1,0)  2=(2,0)  3=(0,1)  4=(1,1)
# Query (1.2, 0.1): distance² order → 1, 2, 4, 0, 3
XS = [0.0, 1.0, 2.0, 0.0, 1.0]
YS = [0.0, 0.0, 0.0, 1.0, 1.0]


@pytest.fixture
def engine():
    return Engine.from_coords(XS, YS)


# construction


def test_from_coords_creates_engine():
    eng = Engine.from_coords(XS, YS)
    assert eng is not None


def test_from_tuple_list_creates_engine():
    pairs = list(zip(XS, YS))
    eng = Engine(pairs)
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
    result = engine.knn(1.2, 0.1, 1)
    assert result == [1]


def test_knn_k2_returns_two_closest(engine):
    result = engine.knn(1.2, 0.1, 2)
    assert sorted(result) == [1, 2]


def test_knn_k3_returns_three_closest(engine):
    result = engine.knn(1.2, 0.1, 3)
    assert sorted(result) == [1, 2, 4]


def test_knn_k_larger_than_n_returns_all(engine):
    result = engine.knn(0.0, 0.0, 100)
    assert sorted(result) == [0, 1, 2, 3, 4]


def test_knn_at_exact_point_returns_that_point(engine):
    result = engine.knn(1.0, 0.0, 1)
    assert result == [1]


def test_knn_approximate_flag_accepted(engine):
    result = engine.knn(1.2, 0.1, 2, approximate=True)
    assert len(result) == 2


# range_query


def test_range_returns_correct_points(engine):
    result = engine.range_query(0.0, 0.0, 1.5, 0.5)
    assert sorted(result) == [0, 1]


def test_range_single_result(engine):
    result = engine.range_query(0.5, 0.5, 1.5, 1.5)
    assert result == [4]


def test_range_all_points(engine):
    result = engine.range_query(0.0, 0.0, 2.0, 1.0)
    assert sorted(result) == [0, 1, 2, 3, 4]


def test_range_empty_returns_empty(engine):
    result = engine.range_query(5.0, 5.0, 10.0, 10.0)
    assert result == []


# contains


def test_contains_exact_point_match(engine):
    result = engine.contains(1.0, 0.0)
    assert result == [1]


def test_contains_no_match_returns_empty(engine):
    result = engine.contains(0.5, 0.5)
    assert result == []


# numpy input


def test_from_numpy_array():
    np = pytest.importorskip("numpy")
    arr = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    eng = Engine(arr)
    assert sorted(eng.knn(1.2, 0.0, 1)) == [1]


# pyarrow input


def test_from_pyarrow_struct_array():
    pa = pytest.importorskip("pyarrow")
    arr = pa.StructArray.from_arrays(
        [pa.array(XS), pa.array(YS)],
        names=["x", "y"],
    )
    eng = Engine(arr)
    assert sorted(eng.knn(1.2, 0.1, 1)) == [1]


def test_from_pyarrow_fixed_size_list():
    pa = pytest.importorskip("pyarrow")
    flat = []
    for x, y in zip(XS, YS):
        flat.extend([x, y])
    arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float64()), 2)
    eng = Engine(arr)
    assert sorted(eng.knn(1.2, 0.1, 1)) == [1]


# large dataset — exercises index selection past the brute-force threshold


def test_large_dataset_knn():
    n = 1000
    xs = [(i % 50) * 2.0 for i in range(n)]
    ys = [(i // 50) * 2.0 for i in range(n)]
    eng = Engine.from_coords(xs, ys)
    result = eng.knn(25.0, 10.0, 5)
    assert len(result) == 5


def test_large_dataset_range():
    n = 1000
    xs = [(i % 50) * 2.0 for i in range(n)]
    ys = [(i // 50) * 2.0 for i in range(n)]
    eng = Engine.from_coords(xs, ys)
    result = eng.range_query(0.0, 0.0, 10.0, 10.0)
    assert len(result) > 0
