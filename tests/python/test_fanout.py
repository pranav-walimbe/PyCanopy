"""Tests for fan-out detection and cache insertion (Phase 5 step 12).

SpatialLazyFrame builds plans via [*self._plan, new_node], which reuses existing
node references rather than copying them. Two branches from the same base therefore
share the same Python objects for their common prefix nodes.

_detect_fanout exploits object identity to find the longest shared prefix.
collect_all emits the prefix once with .cache(), builds each suffix from the cached
result, and calls pl.collect_all() to execute all branches together.
"""

from __future__ import annotations

import polars as pl
import pytest

from pycanopy import SpatialFrame
from pycanopy.lazy import SpatialLazyFrame
from pycanopy.optimizer import SpatialOptimizer


# 10-point grid: (0..4, 0) and (0..4, 1), values 0-9.
# Shared across the whole session so the spatial index is built only once.
@pytest.fixture(scope="session")
def sf():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    df = pl.DataFrame({"x": xs, "y": ys, "v": list(range(10))})
    return SpatialFrame(df, "x", "y")


# _detect_fanout unit tests


def test_detect_fanout_two_branches_share_one_node(sf):
    base = sf.lazy().range_query(0.0, -0.1, 4.1, 1.1)
    r1 = base.filter(pl.col("v") < 5)
    r2 = base.filter(pl.col("v") >= 5)
    assert SpatialOptimizer()._detect_fanout([r1._plan, r2._plan]) == 1


def test_detect_fanout_longer_shared_prefix(sf):
    base = sf.lazy().filter(pl.col("v") >= 0).range_query(0.0, -0.1, 4.1, 1.1)
    r1 = base.filter(pl.col("v") < 3)
    r2 = base.filter(pl.col("v") > 7)
    # ScalarNode and RangeNode are both shared objects
    assert SpatialOptimizer()._detect_fanout([r1._plan, r2._plan]) == 2


def test_detect_fanout_no_common_prefix(sf):
    r1 = sf.lazy().range_query(0.0, -0.1, 2.1, 0.5)
    r2 = sf.lazy().range_query(2.5, 0.5, 4.1, 1.1)
    assert SpatialOptimizer()._detect_fanout([r1._plan, r2._plan]) == 0


def test_detect_fanout_single_plan_returns_zero(sf):
    plan = sf.lazy().range_query(0.0, -0.1, 4.1, 1.1)._plan
    assert SpatialOptimizer()._detect_fanout([plan]) == 0


def test_detect_fanout_same_list_twice(sf):
    base = sf.lazy().range_query(0.0, -0.1, 4.1, 1.1)
    assert SpatialOptimizer()._detect_fanout([base._plan, base._plan]) == 1


def test_detect_fanout_three_branches(sf):
    base = sf.lazy().range_query(0.0, -0.1, 4.1, 1.1)
    r1 = base.filter(pl.col("v") < 3)
    r2 = base.filter(pl.col("v").is_between(3, 6))
    r3 = base.filter(pl.col("v") > 7)
    assert SpatialOptimizer()._detect_fanout([r1._plan, r2._plan, r3._plan]) == 1


def test_detect_fanout_stops_at_first_divergence(sf):
    base = sf.lazy().filter(pl.col("v") >= 0).range_query(0.0, -0.1, 4.1, 1.1)
    r1 = base.range_query(0.0, -0.1, 2.1, 1.1)
    r2 = base.range_query(2.0, -0.1, 4.1, 1.1)
    assert SpatialOptimizer()._detect_fanout([r1._plan, r2._plan]) == 2


# collect_all correctness


def test_collect_all_single_frame(sf):
    r = sf.lazy().range_query(0.0, -0.1, 2.1, 0.5)
    results = SpatialLazyFrame.collect_all([r])
    assert sorted(results[0]["v"].to_list()) == sorted(r.collect()["v"].to_list())


def test_collect_all_no_common_prefix(sf):
    r1 = sf.lazy().range_query(0.0, -0.1, 2.1, 0.5)
    r2 = sf.lazy().range_query(2.5, 0.5, 4.1, 1.1)
    results = SpatialLazyFrame.collect_all([r1, r2])
    assert sorted(results[0]["v"].to_list()) == sorted(r1.collect()["v"].to_list())
    assert sorted(results[1]["v"].to_list()) == sorted(r2.collect()["v"].to_list())


def test_collect_all_matches_individual_collect(sf):
    base = sf.lazy().range_query(0.0, -0.1, 4.1, 1.1)
    r1 = base.filter(pl.col("v") < 5)
    r2 = base.filter(pl.col("v") >= 5)
    results = SpatialLazyFrame.collect_all([r1, r2])
    assert sorted(results[0]["v"].to_list()) == sorted(r1.collect()["v"].to_list())
    assert sorted(results[1]["v"].to_list()) == sorted(r2.collect()["v"].to_list())


def test_collect_all_three_branches(sf):
    base = sf.lazy().range_query(0.0, -0.1, 4.1, 1.1)
    r1 = base.filter(pl.col("v") < 3)
    r2 = base.filter(pl.col("v").is_between(3, 6))
    r3 = base.filter(pl.col("v") > 7)
    results = SpatialLazyFrame.collect_all([r1, r2, r3])
    assert len(results) == 3
    assert sorted(results[0]["v"].to_list()) == sorted(r1.collect()["v"].to_list())
    assert sorted(results[1]["v"].to_list()) == sorted(r2.collect()["v"].to_list())
    assert sorted(results[2]["v"].to_list()) == sorted(r3.collect()["v"].to_list())


def test_collect_all_branch_is_the_prefix_itself(sf):
    # r1 IS the shared base (empty suffix), r2 extends it
    base = sf.lazy().range_query(0.0, -0.1, 4.1, 1.1)
    r2 = base.filter(pl.col("v") > 3)
    results = SpatialLazyFrame.collect_all([base, r2])
    assert sorted(results[0]["v"].to_list()) == sorted(base.collect()["v"].to_list())
    assert sorted(results[1]["v"].to_list()) == sorted(r2.collect()["v"].to_list())


def test_collect_all_result_has_no_internal_row_index(sf):
    base = sf.lazy().range_query(0.0, -0.1, 4.1, 1.1)
    r1 = base.filter(pl.col("v") < 5)
    r2 = base.filter(pl.col("v") >= 5)
    for df in SpatialLazyFrame.collect_all([r1, r2]):
        assert "__orig_row__" not in df.columns


def test_collect_all_longer_shared_prefix(sf):
    base = sf.lazy().filter(pl.col("v") >= 2).range_query(0.0, -0.1, 4.1, 1.1)
    r1 = base.filter(pl.col("v") < 7)
    r2 = base.range_query(0.0, -0.1, 2.1, 1.1)
    results = SpatialLazyFrame.collect_all([r1, r2])
    assert sorted(results[0]["v"].to_list()) == sorted(r1.collect()["v"].to_list())
    assert sorted(results[1]["v"].to_list()) == sorted(r2.collect()["v"].to_list())


def test_collect_all_raises_for_empty_list():
    with pytest.raises(ValueError, match="collect_all requires at least one frame"):
        SpatialLazyFrame.collect_all([])


def test_collect_all_raises_for_different_spatial_frames(sf):
    df2 = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0], "v": [0, 1]})
    sf2 = SpatialFrame(df2, "x", "y")
    r1 = sf.lazy().range_query(0.0, -0.1, 2.1, 1.1)
    r2 = sf2.lazy().range_query(0.0, -0.1, 1.1, 0.5)
    with pytest.raises(ValueError, match="same SpatialFrame"):
        SpatialLazyFrame.collect_all([r1, r2])
