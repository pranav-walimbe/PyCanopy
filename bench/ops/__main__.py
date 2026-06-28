"""Cost model calibration benchmark.

Measures warm probe time per (index_kind, query_kind) across N values and derives the CostFactors constants.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import polars as pl

from bench.ops.utils import (
    generate_clustered_points,
    generate_points,
    generate_polygons,
    time_ms,
)
from pycanopy import SpatialFrame

_N_SIZES = [10_000, 100_000, 500_000, 1_000_000]
_Q = 500
_K = 5
_RUNS = 3
_BRUTE_MAX_N = 100_000
# 10% x 10% of [0, 1]^2, selectivity = 0.01
_SEL = 0.01
_RANGE_BOX = (0.0, 0.0, 0.1, 0.1)


def _query_pts(q: int) -> tuple[np.ndarray, np.ndarray]:
    # Return q random query points uniform over [0, 1]^2
    rng = np.random.default_rng(99)
    return rng.uniform(0.0, 1.0, q), rng.uniform(0.0, 1.0, q)


def _build_time_ms(df: pl.DataFrame, x_col: str, y_col: str, runs: int) -> float:
    # Time build_index() on a fresh engine each run to isolate build from construction
    def _build():
        sf = SpatialFrame(df, x_col, y_col, index_mode="none")
        sf.engine.set_index_mode("eager")
        sf.engine.build_index()

    return time_ms(_build, runs)


def _measure_n(n: int, q: int, runs: int, brute_max_n: int) -> dict[str, float | None]:
    # Measure warm probe and build times for all index/query combinations at dataset size n
    pts_u = generate_points(n, seed=42)
    pts_c = generate_clustered_points(n, seed=42)
    polys = generate_polygons(n, seed=42)

    df_u = pl.DataFrame({"x": pts_u[:, 0], "y": pts_u[:, 1]})
    df_c = pl.DataFrame({"x": pts_c[:, 0], "y": pts_c[:, 1]})
    df_p = pl.DataFrame({"geom": polys.tolist()})

    # eager builds the selected index: uniform → Grid, clustered → KDTree, polygons → RTree
    sf_u = SpatialFrame(df_u, "x", "y", index_mode="eager")
    sf_c = SpatialFrame(df_c, "x", "y", index_mode="eager")
    sf_p = SpatialFrame.from_polygons(df_p, "geom", index_mode="eager")

    qxs, qys = _query_pts(q)
    min_x, min_y, max_x, max_y = _RANGE_BOX

    # Trigger index builds before warm measurements
    sf_u.engine.build_index()
    sf_c.engine.batch_knn_join(qxs, qys, _K)
    sf_p.engine.batch_knn_to_polygons(qxs, qys, _K)

    kd_knn_ms = time_ms(lambda: sf_c.engine.batch_knn_join(qxs, qys, _K), runs)
    rt_knn_ms = time_ms(lambda: sf_p.engine.batch_knn_to_polygons(qxs, qys, _K), runs)

    def _kd_range():
        for _ in range(q):
            sf_c.engine.range_query(min_x, min_y, max_x, max_y)

    def _rt_range():
        for _ in range(q):
            sf_p.engine.range_query(min_x, min_y, max_x, max_y)

    def _grid_range():
        for _ in range(q):
            sf_u.engine.range_query(min_x, min_y, max_x, max_y)

    kd_range_ms = time_ms(_kd_range, runs)
    rt_range_ms = time_ms(_rt_range, runs)
    grid_ms = time_ms(_grid_range, runs)

    brute_ms: float | None = None
    if n <= brute_max_n:
        sf_brute = SpatialFrame(df_u, "x", "y", index_mode="none")
        brute_ms = time_ms(lambda: sf_brute.engine.batch_knn_join(qxs, qys, _K), runs)

    build_tree_ms = _build_time_ms(df_c, "x", "y", runs)

    return {
        "kd_knn": kd_knn_ms,
        "kd_range": kd_range_ms,
        "rt_knn": rt_knn_ms,
        "rt_range": rt_range_ms,
        "grid": grid_ms,
        "brute": brute_ms,
        "build_tree": build_tree_ms,
    }


def _derive_ns(ms: dict[str, float | None], n: int, q: int) -> dict[str, float | None]:
    # Invert measured times through the cost model formulas to derive per_ns constants
    log2n = math.log2(n)
    sel_n = _SEL * n
    return {
        "kd_knn_ns": (ms["kd_knn"] * 1e6) / (q * (log2n + _K)),
        "kd_range_ns": (ms["kd_range"] * 1e6) / (q * (log2n + sel_n)),
        "rt_knn_ns": (ms["rt_knn"] * 1e6) / (q * (log2n + _K)),
        "rt_range_ns": (ms["rt_range"] * 1e6) / (q * (log2n + sel_n)),
        "grid_ns": (ms["grid"] * 1e6) / (q * sel_n),
        "scan_ns": ((ms["brute"] * 1e6) / (q * n)) if ms["brute"] is not None else None,
        "build_tree_ns": (ms["build_tree"] * 1e6) / (n * log2n),
    }


_KEYS = [
    "kd_knn_ns",
    "kd_range_ns",
    "rt_knn_ns",
    "rt_range_ns",
    "grid_ns",
    "scan_ns",
    "build_tree_ns",
]


def run(n_sizes: list[int], q: int, runs: int, brute_max_n: int) -> None:
    """Run the calibration sweep and print per_ns estimates with suggested CostFactors.

    Args:
        n_sizes: Dataset sizes to sweep.
        q: Number of queries per timing call.
        runs: Timing repetitions per measurement (median is taken).
        brute_max_n: Skip brute-force measurement above this N.
    """
    col_w = 12
    key_w = 16

    header = f"{'':>{key_w}}" + "".join(f"{n:>{col_w},}" for n in n_sizes) + f"{'median':>{col_w}}"
    print(header)
    print("-" * len(header))

    results_by_n: dict[int, dict[str, float | None]] = {}
    for n in n_sizes:
        print(f"  N={n:,}...", end=" ", flush=True)
        ms = _measure_n(n, q, runs, brute_max_n)
        results_by_n[n] = _derive_ns(ms, n, q)
        print("done")

    print()
    all_ns: dict[str, list[float]] = {k: [] for k in _KEYS}
    for key in _KEYS:
        vals = [results_by_n[n][key] for n in n_sizes]
        numeric = [v for v in vals if v is not None]
        med = float(np.median(numeric)) if numeric else None
        if med is not None:
            all_ns[key].extend(numeric)
        cells = "".join(f"{v:>{col_w}.1f}" if v is not None else f"{'—':>{col_w}}" for v in vals)
        med_str = f"{med:>{col_w}.1f}" if med is not None else f"{'—':>{col_w}}"
        print(f"{key:>{key_w}}{cells}{med_str}")

    medians = {k: float(np.median(v)) for k, v in all_ns.items() if v}

    def _r10(key: str) -> int | str:
        v = medians.get(key)
        return max(1, round(v / 10) * 10) if v is not None else "?"

    print()
    print("Suggested CostFactors (copy into src/planner/calibration.rs):")
    print(f"    scan_ns_per_item:  {_r10('scan_ns')}.0,")
    print(f"    build_ns_per_item: {_r10('build_tree_ns')}.0,")
    print(f"    kdtree_knn_ns:     {_r10('kd_knn_ns')}.0,")
    print(f"    kdtree_range_ns:   {_r10('kd_range_ns')}.0,")
    print(f"    rtree_knn_ns:      {_r10('rt_knn_ns')}.0,")
    print(f"    rtree_range_ns:    {_r10('rt_range_ns')}.0,")
    print(f"    grid_range_ns:     {_r10('grid_ns')}.0,")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the calibration benchmark.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        The process exit code, always 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Derive CostFactors constants from measured probe times."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=_N_SIZES, metavar="N")
    parser.add_argument("--queries", type=int, default=_Q, metavar="Q")
    parser.add_argument("--runs", type=int, default=_RUNS, metavar="R")
    parser.add_argument("--brute-max-n", type=int, default=_BRUTE_MAX_N, metavar="N")
    args = parser.parse_args(argv)
    run(args.sizes, args.queries, args.runs, args.brute_max_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
