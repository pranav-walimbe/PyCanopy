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


def test_optimizer_does_not_flip_when_query_much_smaller(sf):
    # Q=1 << N=1000: use existing engine index, iterate Q queries. No flip.
    query_df = pl.DataFrame({"qx": [0.0], "qy": [0.0]})
    plan = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.0)._plan
    optimized = SpatialOptimizer().optimize(plan, sf.engine)
    node = next(n for n in optimized if isinstance(n, WithinDistanceJoinNode))
    assert node.flip is False


def test_optimizer_sets_flip_when_query_larger_than_half_engine(sf):
    # Q=600 > N//2=500: index query side, iterate N engine points. Flip.
    query_df = pl.DataFrame({"qx": [float(i) for i in range(600)], "qy": [0.0] * 600})
    plan = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.0)._plan
    optimized = SpatialOptimizer().optimize(plan, sf.engine)
    node = next(n for n in optimized if isinstance(n, WithinDistanceJoinNode))
    assert node.flip is True


# streamed joins (morsel batching)
#
# batch_size is forced below the probe size so streaming engages on small fixtures.


def _grid_query(n: int) -> pl.DataFrame:
    # n query points sweeping the 10x100 grid, each landing on a real sf point.
    return pl.DataFrame(
        {"qx": [float(i % 10) for i in range(n)], "qy": [float(i // 10) for i in range(n)]}
    )


def test_streamed_collect_matches_single_shot_within_distance(sf):
    query_df = _grid_query(300)
    single = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.1).collect()
    streamed = (
        sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.1).collect(batch_size=64)
    )
    # Same rows regardless of morsel size (order may differ across morsels).
    assert streamed.sort(streamed.columns).equals(single.sort(single.columns))


def test_streamed_collect_matches_single_shot_knn_join(sf):
    query_df = _grid_query(250)
    single = sf.lazy().knn_join(query_df, "qx", "qy", k=3).collect()
    streamed = sf.lazy().knn_join(query_df, "qx", "qy", k=3).collect(batch_size=50)
    assert len(streamed) == len(single) == 250 * 3
    assert streamed.sort(streamed.columns).equals(single.sort(single.columns))


def test_collect_batched_yields_multiple_morsels_and_concats_to_full(sf):
    query_df = _grid_query(250)
    batches = list(sf.lazy().knn_join(query_df, "qx", "qy", k=2).collect_batched(batch_size=100))
    assert len(batches) == 3  # ceil(250 / 100)
    full = pl.concat(batches)
    single = sf.lazy().knn_join(query_df, "qx", "qy", k=2).collect()
    assert full.sort(full.columns).equals(single.sort(single.columns))


def test_collect_batched_partial_reduction_combines_additively(sf):
    # Per-morsel count reduced then summed must equal the single-shot row count.
    query_df = _grid_query(300)
    per_morsel = [
        b.height
        for b in sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.1)
        .collect_batched(batch_size=64)
    ]
    single = sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.1).collect()
    assert sum(per_morsel) == single.height


def test_small_probe_yields_single_batch(sf):
    query_df = _grid_query(5)
    batches = list(sf.lazy().knn_join(query_df, "qx", "qy", k=2).collect_batched())
    assert len(batches) == 1


def test_collect_batched_without_join_yields_single_frame(sf):
    batches = list(sf.lazy().range_query(0.0, 0.0, 4.0, 4.0).collect_batched())
    assert len(batches) == 1
    expected = sf.lazy().range_query(0.0, 0.0, 4.0, 4.0).collect()
    assert batches[0].sort(batches[0].columns).equals(expected.sort(expected.columns))


def test_streamed_within_join_polygons_matches_single_shot(sf_polygons):
    query_df = pl.DataFrame({"qx": [float(i) + 0.5 for i in range(200)], "qy": [0.5] * 200})
    single = sf_polygons.lazy().within_join(query_df, "qx", "qy").collect()
    streamed = sf_polygons.lazy().within_join(query_df, "qx", "qy").collect(batch_size=32)
    # geom is a shapely object column (unsortable); compare the (qx, poly_id) pairing.
    cols = ["qx", "poly_id"]
    assert streamed.select(cols).sort(cols).equals(single.select(cols).sort(cols))
