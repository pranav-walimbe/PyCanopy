"""Integration tests for SpatialFrame / SpatialLazyFrame.

Covers Phase 4: EXPR plugin path (map_batches barrier), IO plugin path
(pre-built Engine index), scalar+spatial ordering, path selection, and joins.
"""

from __future__ import annotations

import polars as pl
import pytest

from pycanopy import SpatialFrame
from pycanopy.nodes import ContainsNode, FusedSpatialNode, KnnNode, PluginPath, RangeNode
from pycanopy.optimizer import SpatialOptimizer

# fixtures

# 5-point dataset: (0,0),(1,0),(2,0),(0,1),(1,1) with values 10-50.
# Extent (0,0)-(2,1), area=2. Moderate queries have selectivity >= 0.05 → EXPR path.
@pytest.fixture(scope="session")
def sf():
    df = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 0.0, 1.0],
            "y": [0.0, 0.0, 0.0, 1.0, 1.0],
            "v": [10, 20, 30, 40, 50],
        }
    )
    return SpatialFrame(df, "x", "y")


# 100-point uniform grid: xs=0..9, ys=0..9 (i%10, i//10).
# contains selectivity = 1/100 = 0.01 < 0.05 → IO path.
# Tight range over extent (0,0)-(9,9) also falls below threshold.
@pytest.fixture(scope="session")
def sf_large():
    xs = [(i % 10) * 1.0 for i in range(100)]
    ys = [(i // 10) * 1.0 for i in range(100)]
    df = pl.DataFrame({"x": xs, "y": ys, "v": list(range(100))})
    return SpatialFrame(df, "x", "y")


# EXPR path correctness


def test_range_returns_matching_rows(sf):
    result = sf.lazy().range_query(0.0, 0.0, 1.5, 0.5).collect()
    assert sorted(result["v"].to_list()) == [10, 20]


def test_range_empty_bbox_returns_no_rows(sf):
    result = sf.lazy().range_query(5.0, 5.0, 10.0, 10.0).collect()
    assert result.is_empty()


def test_contains_returns_exact_match(sf):
    result = sf.lazy().contains(1.0, 0.0).collect()
    assert result["v"].to_list() == [20]


def test_contains_no_match_returns_empty(sf):
    result = sf.lazy().contains(0.5, 0.5).collect()
    assert result.is_empty()


def test_knn_returns_k_nearest(sf):
    # query (1.2, 0.1): nearest are (1,0)=v20 and (2,0)=v30
    result = sf.lazy().knn(1.2, 0.1, 2).collect()
    assert sorted(result["v"].to_list()) == [20, 30]


def test_knn_k_larger_than_n_returns_all(sf):
    result = sf.lazy().knn(0.0, 0.0, 100).collect()
    assert sorted(result["v"].to_list()) == [10, 20, 30, 40, 50]


# scalar + spatial ordering


def test_scalar_before_range_filters_correctly(sf):
    # scalar: v > 15 keeps (1,0)v20 (2,0)v30 (0,1)v40 (1,1)v50
    # range (0,0)-(1.5,0.5): keeps x<=1.5, y<=0.5 → only (1,0)v20
    result = sf.lazy().filter(pl.col("v") > 15).range_query(0.0, 0.0, 1.5, 0.5).collect()
    assert result["v"].to_list() == [20]


def test_range_declared_before_scalar_gives_same_result(sf):
    # Optimizer reorders scalar before spatial; result must match.
    result = sf.lazy().range_query(0.0, 0.0, 1.5, 0.5).filter(pl.col("v") > 15).collect()
    assert result["v"].to_list() == [20]


def test_no_predicates_returns_all_rows(sf):
    result = sf.lazy().collect()
    assert sorted(result["v"].to_list()) == [10, 20, 30, 40, 50]


# chained spatial predicates


def test_two_range_queries_intersect(sf):
    # First range: all 5 points. Second range: x in [0.5,2.5], y in [-0.1,0.5] → (1,0) and (2,0).
    result = (
        sf.lazy()
        .range_query(0.0, 0.0, 2.0, 1.0)
        .range_query(0.5, -0.1, 2.5, 0.5)
        .collect()
    )
    assert sorted(result["v"].to_list()) == [20, 30]


# IO path correctness


def test_io_contains_returns_correct_row(sf_large):
    # selectivity = 1/100 = 0.01 < 0.05 → IO path.
    # point at (3,4): index = 3 + 4*10 = 43, v=43.
    result = sf_large.lazy().contains(3.0, 4.0).collect()
    assert result["v"].to_list() == [43]


def test_io_range_returns_correct_row(sf_large):
    # bbox (0,0)-(0.5,0.5): area 0.25 vs total (0-9)^2=81 → selectivity ~0.003 → IO path.
    # Only point (0,0) = index 0, v=0.
    result = sf_large.lazy().range_query(0.0, 0.0, 0.5, 0.5).collect()
    assert result["v"].to_list() == [0]


def test_io_scalar_applied_to_candidates(sf_large):
    # IO path slices to spatial candidates, then applies scalar filter.
    result = sf_large.lazy().contains(3.0, 4.0).filter(pl.col("v") > 50).collect()
    assert result.is_empty()


def test_io_two_contains_intersect(sf_large):
    # Two contains predicates on the same point: still returns that point.
    result = (
        sf_large.lazy()
        .contains(3.0, 4.0)
        .contains(3.0, 4.0)
        .collect()
    )
    assert result["v"].to_list() == [43]


def test_io_disjoint_ranges_return_empty(sf_large):
    # First range has a candidate; second is disjoint → intersection is empty.
    result = (
        sf_large.lazy()
        .range_query(0.0, 0.0, 0.5, 0.5)
        .range_query(5.0, 5.0, 9.5, 9.5)
        .collect()
    )
    assert result.is_empty()


# plugin path selection


def test_path_select_expr_for_moderate_selectivity(sf):
    # range selectivity = (1.5*0.5) / (2*1) = 0.375 > 0.05 → EXPR.
    plan = sf.lazy().range_query(0.0, 0.0, 1.5, 0.5)._plan
    opt = SpatialOptimizer()
    optimized = opt.optimize(plan, sf.engine)
    assert opt._select_plugin_path(optimized, sf.engine) == PluginPath.EXPR


def test_path_select_io_for_tight_selectivity(sf_large):
    # contains selectivity = 1/100 = 0.01 < 0.05 → IO.
    plan = sf_large.lazy().contains(3.0, 4.0)._plan
    opt = SpatialOptimizer()
    optimized = opt.optimize(plan, sf_large.engine)
    assert opt._select_plugin_path(optimized, sf_large.engine) == PluginPath.IO


def test_path_select_expr_when_knn_present(sf_large):
    # KNN node forces EXPR regardless of spatial selectivity.
    plan = sf_large.lazy().contains(3.0, 4.0).knn(3.0, 4.0, 1)._plan
    opt = SpatialOptimizer()
    optimized = opt.optimize(plan, sf_large.engine)
    assert opt._select_plugin_path(optimized, sf_large.engine) == PluginPath.EXPR


def test_path_select_expr_when_knn_join_present(sf_large):
    query_df = pl.DataFrame({"qx": [3.0], "qy": [4.0]})
    plan = sf_large.lazy().knn_join(query_df, "qx", "qy", k=1)._plan
    opt = SpatialOptimizer()
    optimized = opt.optimize(plan, sf_large.engine)
    assert opt._select_plugin_path(optimized, sf_large.engine) == PluginPath.EXPR


# join nodes


def test_knn_join_returns_k_rows_per_query(sf):
    query_df = pl.DataFrame({"qx": [1.2], "qy": [0.1]})
    result = sf.lazy().knn_join(query_df, "qx", "qy", k=2).collect()
    assert len(result) == 2
    assert sorted(result["v"].to_list()) == [20, 30]


def test_knn_join_multiple_queries(sf):
    query_df = pl.DataFrame({"qx": [1.2, 0.1], "qy": [0.1, 0.1]})
    result = sf.lazy().knn_join(query_df, "qx", "qy", k=1).collect()
    # Each query returns 1 nearest: (1.2,0.1)→(1,0), (0.1,0.1)→(0,0)
    assert len(result) == 2
    assert sorted(result["v"].to_list()) == [10, 20]
