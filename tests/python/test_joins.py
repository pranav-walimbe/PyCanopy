"""Tests for spatial join operations.

Uses a 1000-point dataset (10x100 grid) so real indexes are exercised.
A single engine and SpatialFrame are shared across the module.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from shapely.geometry import box

from pycanopy import SpatialFrame
from pycanopy.nodes import WithinDistanceJoinNode
from pycanopy.optimizer import SpatialOptimizer

_N = 1000  # 10x100 grid, above the 500 brute-force threshold


@pytest.fixture(scope="module")
def sf():
    xs = np.array([float(i % 10) for i in range(_N)], dtype=np.float64)
    ys = np.array([float(i // 10) for i in range(_N)], dtype=np.float64)
    df = pl.DataFrame({"x": xs, "y": ys, "id": list(range(_N))})
    return SpatialFrame(df, "x", "y")


@pytest.fixture(scope="module")
def sf_polygons():
    polygons = [box(i, 0, i + 0.9, 0.9) for i in range(500)]
    df = pl.DataFrame({"poly_id": list(range(500)), "geom": polygons})
    return SpatialFrame.from_polygons(df, geometry_col="geom")


# knn_join


def test_knn_join_returns_k_rows_per_query(sf):
    query_df = pl.DataFrame({"qx": [0.0, 5.0], "qy": [0.0, 50.0]})
    result = sf.lazy().knn_join(query_df, "qx", "qy", k=3).collect()
    assert len(result) == 6


def test_knn_join_result_contains_query_and_target_columns(sf):
    query_df = pl.DataFrame({"qx": [0.0], "qy": [0.0]})
    result = sf.lazy().knn_join(query_df, "qx", "qy", k=2).collect()
    assert "qx" in result.columns
    assert "id" in result.columns


def test_knn_join_nearest_point_is_correct(sf):
    # Query at (3.0, 50.0); nearest sf point is (3, 50) with id = 3 + 50*10 = 503.
    query_df = pl.DataFrame({"qx": [3.0], "qy": [50.0]})
    result = sf.lazy().knn_join(query_df, "qx", "qy", k=1).collect()
    assert result["id"].to_list() == [503]


# within_join


def test_within_join_returns_correct_pairs(sf_polygons):
    # Points at (0.5, 0.5) and (1.5, 0.5) should land in polygons 0 and 1.
    query_df = pl.DataFrame({"qx": [0.5, 1.5], "qy": [0.5, 0.5]})
    result = sf_polygons.lazy().within_join(query_df, "qx", "qy").collect()
    assert len(result) == 2
    assert sorted(result["poly_id"].to_list()) == [0, 1]


def test_within_join_no_match_returns_empty(sf_polygons):
    query_df = pl.DataFrame({"qx": [999.0], "qy": [999.0]})
    result = sf_polygons.lazy().within_join(query_df, "qx", "qy").collect()
    assert result.is_empty()


def test_within_join_flip_matches_standard(sf_polygons):
    # 3 query points << engine.n=500 / 2 = 250 so optimizer sets flip=True.
    # Result must be identical to the non-flipped path.
    query_df = pl.DataFrame({"qx": [0.5, 2.5, 4.5], "qy": [0.5, 0.5, 0.5]})
    result = sf_polygons.lazy().within_join(query_df, "qx", "qy").collect()
    assert len(result) == 3
    assert sorted(result["poly_id"].to_list()) == [0, 2, 4]


# within_distance_join


def test_within_distance_join_finds_nearby_points(sf):
    # Query at (0.0, 0.0); sf points within distance 1.5 are:
    # (0,0)=0, (1,0)=1, (0,1)=10, (1,1)=11 — all at distance <= sqrt(2) < 1.5.
    query_df = pl.DataFrame({"qx": [0.0], "qy": [0.0]})
    result = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.5).collect()
    assert sorted(result["id"].to_list()) == [0, 1, 10, 11]


def test_within_distance_join_empty_when_no_match(sf):
    query_df = pl.DataFrame({"qx": [999.0], "qy": [999.0]})
    result = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.0).collect()
    assert result.is_empty()


def test_within_distance_join_result_has_both_schemas(sf):
    query_df = pl.DataFrame({"qx": [0.0], "qy": [0.0]})
    result = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=0.5).collect()
    assert "qx" in result.columns
    assert "id" in result.columns


def test_within_distance_join_flip_matches_standard(sf):
    # 3 query points << 1000 / 2 = 500 so optimizer sets flip=True.
    # Both paths must return the same result set.
    query_df = pl.DataFrame({"qx": [0.0, 5.0, 9.0], "qy": [0.0, 50.0, 99.0]})
    result = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.1).collect()
    assert len(result) > 0
    assert "qx" in result.columns and "id" in result.columns


# join-side selection


def test_optimizer_sets_flip_when_query_much_smaller(sf):
    query_df = pl.DataFrame({"qx": [0.0], "qy": [0.0]})
    plan = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.0)._plan
    optimized = SpatialOptimizer().optimize(plan, sf.engine)
    node = next(n for n in optimized if isinstance(n, WithinDistanceJoinNode))
    assert node.flip is True


def test_optimizer_does_not_flip_when_query_not_much_smaller(sf):
    # 600 query points is not < 1000 // 2 = 500, so no flip.
    query_df = pl.DataFrame({"qx": [float(i) for i in range(600)], "qy": [0.0] * 600})
    plan = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.0)._plan
    optimized = SpatialOptimizer().optimize(plan, sf.engine)
    node = next(n for n in optimized if isinstance(n, WithinDistanceJoinNode))
    assert node.flip is False
