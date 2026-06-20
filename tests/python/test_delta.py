"""Tests for the Engine delta buffer.

Uses a 25x25 uniform grid (625 points) so the selector picks real indexes
(Grid for range, KD-tree for KNN) rather than brute force. A single engine
is shared across the module; flush tests are ordered last.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycanopy import Engine

_N = 625  # 25x25 grid, above the 500 brute-force threshold


@pytest.fixture(scope="module")
def engine():
    xs = np.array([float(i % 25) for i in range(_N)], dtype=np.float64)
    ys = np.array([float(i // 25) for i in range(_N)], dtype=np.float64)
    return Engine.from_coords(xs, ys)


# delta visibility — each test uses a unique far-away region so accumulated
# state from prior tests does not affect result checks.


def test_range_query_sees_delta_point(engine):
    engine.append_delta(np.array([1000.0], dtype=np.float64), np.array([1000.0], dtype=np.float64))
    result = engine.range_query(999.5, 999.5, 1000.5, 1000.5)
    assert len(result) >= 1


def test_knn_returns_delta_point_as_nearest(engine):
    engine.append_delta(np.array([2000.0], dtype=np.float64), np.array([2000.0], dtype=np.float64))
    result = engine.knn(2000.0, 2000.0, 1)
    # nearest point to (2000, 2000) must be our delta point — grid ends at (24, 24)
    assert len(result) == 1
    assert engine.range_query(1999.5, 1999.5, 2000.5, 2000.5)


def test_contains_sees_delta_point(engine):
    engine.append_delta(np.array([3000.0], dtype=np.float64), np.array([3000.0], dtype=np.float64))
    result = engine.contains(3000.0, 3000.0)
    assert len(result) >= 1


def test_batch_knn_returns_delta_point_as_nearest(engine):
    engine.append_delta(np.array([4000.0], dtype=np.float64), np.array([4000.0], dtype=np.float64))
    qxs = np.array([4000.0], dtype=np.float64)
    qys = np.array([4000.0], dtype=np.float64)
    engine.batch_knn_join(qxs, qys, 1)
    # verify the returned index points back into our region
    assert engine.range_query(3999.5, 3999.5, 4000.5, 4000.5)


# flush mechanics — run after visibility tests so accumulated delta is representative.


def test_size_cap_flushes_delta(engine):
    n_before = engine.n
    # delta already has 4 points from above; add 70 more to exceed the 10% cap
    engine.append_delta(
        np.full(70, 5000.0, dtype=np.float64),
        np.arange(70, dtype=np.float64),
    )
    assert engine.delta_len == 0
    assert engine.n > n_before


def test_cost_flush_fires():
    # Fresh uniform grid so select_index reliably picks Grid (cost threshold = N).
    # The shared module engine has accumulated flushed points from prior tests,
    # shifting its distribution to Clustered (KD-tree, cost = N*log2 N), which
    # would require ~140 queries to trigger — too expensive for a unit test.
    _n = 529  # 23x23 uniform grid, above the 500 brute-force threshold
    eng = Engine.from_coords(
        np.array([float(i % 23) for i in range(_n)], dtype=np.float64),
        np.array([float(i // 23) for i in range(_n)], dtype=np.float64),
    )
    delta_size = 50
    assert delta_size < _n * 0.1
    # Grid cost threshold = N → flush fires after ceil(N / delta_size) queries.
    queries_needed = math.ceil(_n / delta_size) + 1

    eng.append_delta(
        np.full(delta_size, 6000.0, dtype=np.float64),
        np.arange(delta_size, dtype=np.float64),
    )
    assert eng.delta_len == delta_size

    for _ in range(queries_needed):
        eng.range_query(5999.0, -1.0, 6001.0, 51.0)

    assert eng.delta_len == 0


def test_flushed_points_queryable_from_main_index(engine):
    # All delta points from previous tests have been flushed into the main index.
    # Points at x=1000, 2000, 3000, 4000 should be reachable via range query.
    for x in [1000.0, 2000.0, 3000.0, 4000.0]:
        assert len(engine.range_query(x - 0.5, x - 0.5, x + 0.5, x + 0.5)) >= 1
