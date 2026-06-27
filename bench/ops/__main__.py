"""Operation benchmark

Runs each join operation cold (SpatialFrame construction + index build + query) and warm
(index cached, query only) against the best available indexed Python baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import shapely

from bench.ops.utils import (
    BenchmarkReport,
    MockDataset,
    geopandas_batch_contains_indexed,
    geopandas_intersects_self_join_indexed,
    geopandas_knn_join_indexed,
    geopandas_polygon_knn_join_indexed,
    geopandas_within_distance_indexed,
    measure_sf,
)

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

N_JOIN = 100_000   # point join target and probe size
N_POLY = 100_000   # polygon join target and probe size

K_JOIN = 5
DISTANCE = 0.05
POLYGON_SIZE = 0.005


def _query_points(n: int) -> pl.DataFrame:
    # Return n random query points as a (qx, qy) DataFrame
    pts = np.random.default_rng(7).uniform(0.0, 1.0, (n, 2))
    return pl.DataFrame({"qx": pts[:, 0], "qy": pts[:, 1]})


def _point_join_ops(report: BenchmarkReport) -> None:
    # knn_join (cKDTree) and within_distance_join (STRtree) on N_JOIN points
    ds = MockDataset("points", n=N_JOIN, seed=42)
    query_df = _query_points(N_JOIN)
    coords = ds.as_coords()
    gs = gpd.GeoSeries(shapely.points(coords[:, 0], coords[:, 1]))

    cold_fn, warm_fn = geopandas_knn_join_indexed(gs, query_df, K_JOIN)
    report.add(
        measure_sf(
            f"knn_join k={K_JOIN}",
            ds,
            lambda sf: sf.lazy().knn_join(query_df, "qx", "qy", k=K_JOIN).collect(),
            competitors=[("cKDTree", cold_fn, warm_fn)],
        )
    )

    cold_fn, warm_fn = geopandas_within_distance_indexed(gs, query_df, DISTANCE)
    report.add(
        measure_sf(
            "within_distance_join",
            ds,
            lambda sf: (
                sf.lazy().within_distance_join(query_df, "qx", "qy", distance=DISTANCE).collect()
            ),
            competitors=[("STRtree", cold_fn, warm_fn)],
        )
    )


def _polygon_join_ops(report: BenchmarkReport) -> None:
    # polygon_knn_join (cKDTree centroids), within_join, polygon_within_distance_join, intersects self-join
    ds = MockDataset("polygons", n=N_POLY, seed=42, polygon_size=POLYGON_SIZE)
    query_df = _query_points(N_POLY)
    gs = gpd.GeoSeries(ds.as_shapely_list())

    cold_fn, warm_fn = geopandas_polygon_knn_join_indexed(gs, query_df, K_JOIN)
    report.add(
        measure_sf(
            f"polygon_knn_join k={K_JOIN}",
            ds,
            lambda sf: sf.lazy().polygon_knn_join(query_df, "qx", "qy", k=K_JOIN).collect(),
            competitors=[("cKDTree (centroids)", cold_fn, warm_fn)],
        )
    )

    cold_fn, warm_fn = geopandas_batch_contains_indexed(gs, query_df)
    report.add(
        measure_sf(
            "within_join",
            ds,
            lambda sf: sf.lazy().within_join(query_df, "qx", "qy").collect(),
            competitors=[("STRtree", cold_fn, warm_fn)],
        )
    )

    cold_fn, warm_fn = geopandas_within_distance_indexed(gs, query_df, DISTANCE)
    report.add(
        measure_sf(
            "polygon_within_distance_join",
            ds,
            lambda sf: (
                sf.lazy()
                .polygon_within_distance_join(query_df, "qx", "qy", distance=DISTANCE)
                .collect()
            ),
            competitors=[("STRtree", cold_fn, warm_fn)],
        )
    )

    cold_fn, warm_fn = geopandas_intersects_self_join_indexed(gs)
    report.add(
        measure_sf(
            "intersects self-join",
            ds,
            lambda sf: sf.intersects_pairs(),
            competitors=[("STRtree", cold_fn, warm_fn)],
        )
    )


def _warm_polars_jit() -> None:
    # Fire Polars' JIT before any timed collect() calls
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


def run() -> BenchmarkReport:
    """Run every join operation and write the summary table to assets/.

    Returns:
        The populated BenchmarkReport.
    """
    report = BenchmarkReport()
    _point_join_ops(report)
    _polygon_join_ops(report)
    out = _ASSETS_DIR / "ops.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write_table(out)
    return report


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the operation benchmark suite.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        The process exit code, always 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Run PyCanopy join benchmarks vs GeoPandas on uniform data."
    )
    parser.parse_args(argv)

    _warm_polars_jit()
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
