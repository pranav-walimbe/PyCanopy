"""Operation benchmark: every spatial primitive for one dataset size + distribution.

Runs each operation on a fresh engine (cold, index build included) and again warm
(index cached), against naive baselines, and prints one table. DuckDB / Polars
competitors are included where they have a fair path and skipped if not installed.

Usage:
    python -m bench.ops.run --size 100000
    python -m bench.ops.run --size 100000 --distribution clustered
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import polars as pl
import pyarrow as pa
import shapely

from bench.utils.core import (
    BenchmarkReport,
    geopandas_batch_contains_naive,
    geopandas_contains_naive,
    geopandas_intersects_naive,
    geopandas_knn_join_naive,
    geopandas_knn_naive,
    geopandas_range_naive,
    make_knn_queries,
    make_range_queries,
    measure,
    measure_sf,
    polars_range_naive,
    polars_within_distance_batch_naive,
)
from bench.utils.generators import MockDataset

try:
    import duckdb

    _DUCKDB = True
except ImportError:
    duckdb = None
    _DUCKDB = False

K = 10
K_JOIN = 5
SELECTIVITY = 0.01
DISTANCE = 0.05
POLYGON_SIZE = 0.005


# DuckDB single-operation helpers (RTREE-backed where fair)


def _duckdb_points(xs: np.ndarray, ys: np.ndarray):
    """Return a DuckDB connection with a pts table (x, y, geom) and RTREE index."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.register("_pts", pa.table({"x": xs, "y": ys}))
    con.execute("CREATE TABLE pts AS SELECT x, y, ST_Point(x, y) AS geom FROM _pts")
    con.execute("CREATE INDEX pts_idx ON pts USING RTREE (geom)")
    return con


def _duckdb_range_fn(con, q):
    bx0, by0, bx1, by1 = q

    def fn():
        return con.execute(
            f"SELECT x, y FROM pts WHERE ST_Within(geom, ST_MakeEnvelope({bx0},{by0},{bx1},{by1}))"
        ).fetchall()

    fn()  # warm
    return fn


def _duckdb_knn_fn(con, k: int, q):
    """O(N) distance-sort. DuckDB has no index-backed KNN."""
    qx, qy = q

    def fn():
        return con.execute(
            f"SELECT x, y FROM pts ORDER BY ST_Distance(geom, ST_Point({qx},{qy})) LIMIT {k}"
        ).fetchall()

    fn()  # warm
    return fn


# operation groups


def _point_ops(report: BenchmarkReport, ds: MockDataset, size: int) -> None:
    coords = ds.as_coords()
    xs, ys = coords[:, 0], coords[:, 1]
    gs = gpd.GeoSeries(shapely.points(xs, ys))
    df = pl.DataFrame({"x": xs, "y": ys})

    range_q = make_range_queries(coords, SELECTIVITY, n=1, seed=1)[0]
    knn_q = make_knn_queries(n=1, seed=1)[0]

    polars_range_fn = polars_range_naive(df)
    c_range = [
        ("GeoPandas", lambda: geopandas_range_naive(gs)(range_q)),
        ("Polars naive", lambda: polars_range_fn(range_q)),
    ]
    if _DUCKDB:
        c_range.append(("DuckDB", _duckdb_range_fn(_duckdb_points(xs, ys), range_q)))
    else:
        c_range.append(("DuckDB", None))
    report.add(
        measure("range query", ds, lambda eng: eng.range_query(*range_q), competitors=c_range)
    )

    c_knn = [("GeoPandas", lambda: geopandas_knn_naive(gs, K)(knn_q))]
    if _DUCKDB:
        c_knn.append(("DuckDB O(N)", _duckdb_knn_fn(_duckdb_points(xs, ys), K, knn_q)))
    else:
        c_knn.append(("DuckDB O(N)", None))
    report.add(measure(f"kNN k={K}", ds, lambda eng: eng.knn(*knn_q, K), competitors=c_knn))


def _polygon_ops(report: BenchmarkReport, size: int, distribution: str) -> None:
    ds = MockDataset(
        "polygons", n=size, distribution=distribution, seed=42, polygon_size=POLYGON_SIZE
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


def _join_ops(report: BenchmarkReport, size: int, distribution: str) -> None:
    ds_points = MockDataset("points", n=size, distribution=distribution, seed=42)
    ds_polys = MockDataset(
        "polygons", n=size, distribution=distribution, seed=42, polygon_size=POLYGON_SIZE
    )

    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, 1.0, (size, 2))
    query_df = pl.DataFrame({"qx": pts[:, 0], "qy": pts[:, 1]})

    pts_coords = ds_points.as_coords()
    gs_points = gpd.GeoSeries(shapely.points(pts_coords[:, 0], pts_coords[:, 1]))
    gs_polys = gpd.GeoSeries(ds_polys.as_shapely_list())
    df_points = ds_points.as_polars_df()

    report.add(
        measure_sf(
            f"knn_join k={K_JOIN}",
            ds_points,
            lambda sf: sf.lazy().knn_join(query_df, "qx", "qy", k=K_JOIN).collect(),
            competitors=[
                ("GeoPandas loop", lambda: geopandas_knn_join_naive(gs_points, K_JOIN)(query_df))
            ],
        )
    )
    report.add(
        measure_sf(
            "within_distance_join",
            ds_points,
            lambda sf: (
                sf.lazy().within_distance_join(query_df, "qx", "qy", distance=DISTANCE).collect()
            ),
            competitors=[
                ("Polars loop", polars_within_distance_batch_naive(df_points, query_df, DISTANCE))
            ],
        )
    )
    report.add(
        measure_sf(
            "within_join",
            ds_polys,
            lambda sf: sf.lazy().within_join(query_df, "qx", "qy").collect(),
            competitors=[
                ("GeoPandas loop", lambda: geopandas_batch_contains_naive(gs_polys)(query_df))
            ],
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


def run(size: int, distribution: str = "uniform") -> BenchmarkReport:
    """Run every operation for one size + distribution and return the report."""
    ds_points = MockDataset("points", n=size, distribution=distribution, seed=42)
    report = BenchmarkReport(
        label=f"{size:,} {distribution} | operations (cold/warm/naive)", n=size
    )
    _point_ops(report, ds_points, size)
    _polygon_ops(report, size, distribution)
    _join_ops(report, size, distribution)
    report.display()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PyCanopy operation benchmarks.")
    parser.add_argument(
        "--size", type=int, required=True, help="Dataset size (number of geometries)."
    )
    parser.add_argument(
        "--distribution",
        choices=["uniform", "clustered"],
        default="uniform",
        help="Spatial distribution (default: uniform).",
    )
    args = parser.parse_args(argv)

    _warm_polars_jit()
    run(args.size, args.distribution)


if __name__ == "__main__":
    main()
