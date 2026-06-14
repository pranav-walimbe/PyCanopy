"""Operation benchmark: every spatial primitive at a size tuned to its cost.

Runs each operation on a fresh engine (cold, index build included) and again warm
(index cached) against a naive GeoPandas baseline (no spatial index). Single-query
ops run large; the joins run smaller because the naive competitor loops over every
query point. Prints one line per op and writes the summary table to a file in assets/.

Usage:
    python -m bench.ops.run
    python -m bench.ops.run --distribution clustered
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import shapely

from bench.ops.utils import (
    BenchmarkReport,
    MockDataset,
    geopandas_batch_contains_naive,
    geopandas_contains_naive,
    geopandas_intersects_naive,
    geopandas_intersects_self_join_naive,
    geopandas_knn_join_naive,
    geopandas_knn_naive,
    geopandas_range_naive,
    geopandas_within_distance_naive,
    make_knn_queries,
    make_range_queries,
    measure,
    measure_sf,
)

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

# Per-tier dataset sizes, set by what the naive GeoPandas competitor costs: a single
# query scans N once (cheap), but a join loops over all N query points (O(N^2)).
N_SINGLE = 100_000  # single-query ops: range, kNN, contains, range (polygons)
N_JOIN = 10_000  # point joins: knn_join, within_distance_join
N_POLY = 5_000  # polygon joins + self-join (point-to-polygon distance is heavier)

K = 10
K_JOIN = 5
SELECTIVITY = 0.01
DISTANCE = 0.05
POLYGON_SIZE = 0.005


def _query_points(n: int) -> pl.DataFrame:
    """Return n random query points as a (qx, qy) DataFrame."""
    pts = np.random.default_rng(7).uniform(0.0, 1.0, (n, 2))
    return pl.DataFrame({"qx": pts[:, 0], "qy": pts[:, 1]})


def _point_query_ops(report: BenchmarkReport, distribution: str) -> None:
    ds = MockDataset("points", n=N_SINGLE, distribution=distribution, seed=42)
    coords = ds.as_coords()
    gs = gpd.GeoSeries(shapely.points(coords[:, 0], coords[:, 1]))
    range_q = make_range_queries(coords, SELECTIVITY, n=1, seed=1)[0]
    knn_q = make_knn_queries(n=1, seed=1)[0]

    report.add(
        measure(
            "range query",
            ds,
            lambda eng: eng.range_query(*range_q),
            competitors=[("GeoPandas", lambda: geopandas_range_naive(gs)(range_q))],
        )
    )
    report.add(
        measure(
            f"kNN k={K}",
            ds,
            lambda eng: eng.knn(*knn_q, K),
            competitors=[("GeoPandas", lambda: geopandas_knn_naive(gs, K)(knn_q))],
        )
    )


def _polygon_query_ops(report: BenchmarkReport, distribution: str) -> None:
    ds = MockDataset(
        "polygons", n=N_SINGLE, distribution=distribution, seed=42, polygon_size=POLYGON_SIZE
    )
    gs = gpd.GeoSeries(ds.as_shapely_list())
    first_centroid = gs.iloc[0].centroid
    contain_q = (first_centroid.x, first_centroid.y)
    range_q = (0.45, 0.45, 0.49, 0.49)

    report.add(
        measure(
            "contains (polygons)",
            ds,
            lambda eng: eng.contains(*contain_q),
            competitors=[("GeoPandas", lambda: geopandas_contains_naive(gs)(contain_q))],
        )
    )
    report.add(
        measure(
            "range (polygons)",
            ds,
            lambda eng: eng.range_query(*range_q),
            competitors=[("GeoPandas", lambda: geopandas_intersects_naive(gs)(range_q))],
        )
    )


def _point_join_ops(report: BenchmarkReport, distribution: str) -> None:
    ds = MockDataset("points", n=N_JOIN, distribution=distribution, seed=42)
    query_df = _query_points(N_JOIN)
    coords = ds.as_coords()
    gs = gpd.GeoSeries(shapely.points(coords[:, 0], coords[:, 1]))

    report.add(
        measure_sf(
            f"knn_join k={K_JOIN}",
            ds,
            lambda sf: sf.lazy().knn_join(query_df, "qx", "qy", k=K_JOIN).collect(),
            competitors=[("GeoPandas", lambda: geopandas_knn_join_naive(gs, K_JOIN)(query_df))],
        )
    )
    report.add(
        measure_sf(
            "within_distance_join",
            ds,
            lambda sf: (
                sf.lazy().within_distance_join(query_df, "qx", "qy", distance=DISTANCE).collect()
            ),
            competitors=[
                ("GeoPandas", lambda: geopandas_within_distance_naive(gs, DISTANCE)(query_df))
            ],
        )
    )


def _polygon_join_ops(report: BenchmarkReport, distribution: str) -> None:
    ds = MockDataset(
        "polygons", n=N_POLY, distribution=distribution, seed=42, polygon_size=POLYGON_SIZE
    )
    query_df = _query_points(N_POLY)
    gs = gpd.GeoSeries(ds.as_shapely_list())

    report.add(
        measure_sf(
            "within_join",
            ds,
            lambda sf: sf.lazy().within_join(query_df, "qx", "qy").collect(),
            competitors=[("GeoPandas", lambda: geopandas_batch_contains_naive(gs)(query_df))],
        )
    )
    report.add(
        measure_sf(
            f"polygon_knn_join k={K_JOIN}",
            ds,
            lambda sf: sf.lazy().polygon_knn_join(query_df, "qx", "qy", k=K_JOIN).collect(),
            competitors=[("GeoPandas", lambda: geopandas_knn_join_naive(gs, K_JOIN)(query_df))],
        )
    )
    report.add(
        measure_sf(
            "polygon_within_distance_join",
            ds,
            lambda sf: (
                sf.lazy()
                .polygon_within_distance_join(query_df, "qx", "qy", distance=DISTANCE)
                .collect()
            ),
            competitors=[
                ("GeoPandas", lambda: geopandas_within_distance_naive(gs, DISTANCE)(query_df))
            ],
        )
    )
    report.add(
        measure_sf(
            "intersects self-join",
            ds,
            lambda sf: sf.intersects_pairs(),
            competitors=[("GeoPandas", geopandas_intersects_self_join_naive(gs))],
        )
    )


def _warm_polars_jit() -> None:
    """Fire Polars' JIT before any timed collect() calls."""
    pl.DataFrame({"x": [0.0], "y": [0.0]}).lazy().filter(pl.col("x") > 0.0).collect()
    (
        pl.DataFrame({"x": [0.0]})
        .with_row_index("__r__")
        .lazy()
        .filter(
            pl.col("__r__").map_batches(
                lambda s: pl.Series([True] * len(s)),
                return_dtype=pl.Boolean,
                is_elementwise=False,
            )
        )
        .collect()
    )


def run(distribution: str = "uniform") -> BenchmarkReport:
    """Run every operation and write the CSV summary to assets/."""
    report = BenchmarkReport()
    _point_query_ops(report, distribution)
    _polygon_query_ops(report, distribution)
    _point_join_ops(report, distribution)
    _polygon_join_ops(report, distribution)
    out = _ASSETS_DIR / f"ops_{distribution}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write_table(out)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PyCanopy operation benchmarks.")
    parser.add_argument(
        "--distribution",
        choices=["uniform", "clustered"],
        default="uniform",
        help="Spatial distribution (default: uniform).",
    )
    args = parser.parse_args(argv)

    _warm_polars_jit()
    run(args.distribution)


if __name__ == "__main__":
    main()
