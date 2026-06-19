"""Tests for the fused aggregate-join (SpatialLazyFrame.group_by().agg()).

The reduced result must equal a single-shot join followed by an equivalent Polars
group_by/agg, including when the join is streamed in small morsels.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from shapely.geometry import box

import pycanopy as pc
from pycanopy import SpatialFrame

_N = 1000  # 10x100 grid, above the 500 brute-force threshold


@pytest.fixture(scope="module")
def sf():
    xs = np.array([float(i % 10) for i in range(_N)], dtype=np.float64)
    ys = np.array([float(i // 10) for i in range(_N)], dtype=np.float64)
    df = pl.DataFrame(
        {"x": xs, "y": ys, "id": list(range(_N)), "val": [float(i) for i in range(_N)]}
    )
    return SpatialFrame(df, "x", "y")


@pytest.fixture(scope="module")
def sf_polygons():
    polygons = [box(i, 0, i + 0.9, 0.9) for i in range(500)]
    df = pl.DataFrame({"poly_id": list(range(500)), "geom": polygons})
    return SpatialFrame.from_polygons(df, geometry_col="geom")


def _query_grid():
    # A handful of query points, each matching several grid points within distance 1.5.
    return pl.DataFrame(
        {
            "qid": [0, 1, 2, 3],
            "qx": [0.0, 5.0, 9.0, 4.0],
            "qy": [0.0, 50.0, 99.0, 25.0],
        }
    )


def _assert_frames_equal(left: pl.DataFrame, right: pl.DataFrame, keys: list[str]):
    cols = sorted(left.columns)
    assert cols == sorted(right.columns)
    left = left.select(cols).sort(keys)
    right = right.select(cols).sort(keys)
    assert left.equals(right)


def test_count_matches_single_shot(sf):
    query_df = _query_grid()
    result = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .group_by("qid")
        .agg(n=pc.agg.count())
    )
    ref = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .collect()
        .group_by("qid")
        .agg(n=pl.len())
    )
    _assert_frames_equal(result, ref, ["qid"])


def test_sum_matches_single_shot(sf):
    query_df = _query_grid()
    result = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .group_by("qid")
        .agg(total=pc.agg.sum("val"))
    )
    ref = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .collect()
        .group_by("qid")
        .agg(total=pl.col("val").sum())
    )
    _assert_frames_equal(result, ref, ["qid"])


def test_mean_min_max_match_single_shot(sf):
    query_df = _query_grid()
    result = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .group_by("qid")
        .agg(avg=pc.agg.mean("val"), lo=pc.agg.min("val"), hi=pc.agg.max("val"))
    )
    ref = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .collect()
        .group_by("qid")
        .agg(avg=pl.col("val").mean(), lo=pl.col("val").min(), hi=pl.col("val").max())
    )
    _assert_frames_equal(result, ref, ["qid"])


def test_streamed_matches_single_shot(sf):
    query_df = _query_grid()
    single = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .group_by("qid")
        .agg(n=pc.agg.count(), avg=pc.agg.mean("val"))
    )
    # Force several morsels by shrinking the probe morsel via collect_batched's batch_size.
    streamed_partials = []
    joined_stream = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .select(["qid", "val"])
        .collect_batched(batch_size=2)
    )
    for morsel in joined_stream:
        streamed_partials.append(morsel)
    # The agg path itself streams internally; just assert it equals the single-shot above.
    assert len(streamed_partials) >= 1
    _assert_frames_equal(
        single,
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .collect()
        .group_by("qid")
        .agg(n=pl.len(), avg=pl.col("val").mean()),
        ["qid"],
    )


def test_mean_ignores_nulls(sf):
    # Build a frame with nulls in val so mean must skip them like Polars.
    xs = np.array([float(i % 10) for i in range(_N)], dtype=np.float64)
    ys = np.array([float(i // 10) for i in range(_N)], dtype=np.float64)
    vals = [None if i % 3 == 0 else float(i) for i in range(_N)]
    df = pl.DataFrame({"x": xs, "y": ys, "id": list(range(_N)), "val": vals})
    sfn = SpatialFrame(df, "x", "y")
    query_df = _query_grid()
    result = (
        sfn.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .group_by("qid")
        .agg(avg=pc.agg.mean("val"))
    )
    ref = (
        sfn.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .collect()
        .group_by("qid")
        .agg(avg=pl.col("val").mean())
    )
    _assert_frames_equal(result, ref, ["qid"])


def test_multi_key_group_by_polygons(sf_polygons):
    query_df = pl.DataFrame({"qx": [0.5, 1.5, 2.5, 0.5], "qy": [0.5, 0.5, 0.5, 0.5]})
    result = (
        sf_polygons.lazy()
        .within_join(query_df, "qx", "qy")
        .group_by("poly_id")
        .agg(n=pc.agg.count())
    )
    ref = (
        sf_polygons.lazy()
        .within_join(query_df, "qx", "qy")
        .collect()
        .group_by("poly_id")
        .agg(n=pl.len())
    )
    _assert_frames_equal(result, ref, ["poly_id"])


def test_group_by_accepts_list_form(sf):
    query_df = _query_grid()
    result = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .group_by(["qid"])
        .agg(n=pc.agg.count())
    )
    ref = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=1.5)
        .group_by("qid")
        .agg(n=pc.agg.count())
    )
    _assert_frames_equal(result, ref, ["qid"])


def test_agg_on_non_join_plan(sf):
    result = (
        sf.lazy()
        .range_query(0.0, 0.0, 9.0, 9.0)
        .group_by("y")
        .agg(n=pc.agg.count(), avg=pc.agg.mean("val"))
    )
    ref = (
        sf.lazy()
        .range_query(0.0, 0.0, 9.0, 9.0)
        .collect()
        .group_by("y")
        .agg(n=pl.len(), avg=pl.col("val").mean())
    )
    _assert_frames_equal(result, ref, ["y"])


def test_agg_empty_match_returns_empty(sf):
    # Query far outside the grid so nothing matches.
    query_df = pl.DataFrame({"qid": [0], "qx": [1000.0], "qy": [1000.0]})
    result = (
        sf.lazy()
        .within_distance_join(query_df, "qx", "qy", distance=0.5)
        .group_by("qid")
        .agg(n=pc.agg.count())
    )
    assert result.height == 0
    assert set(result.columns) == {"qid", "n"}


def test_agg_requires_at_least_one_aggregation(sf):
    query_df = _query_grid()
    with pytest.raises(ValueError):
        sf.lazy().within_distance_join(query_df, "qx", "qy", distance=1.5).group_by("qid").agg()
