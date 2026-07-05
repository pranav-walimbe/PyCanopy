"""
Cost model calibration benchmark: measure ns/op ratios for the planner's CostFactors.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from bench.ops.utils import (
    generate_clustered_points,
    generate_points,
    generate_polygons,
    peak_rss_mb,
    time_min,
)
from pycanopy import SpatialFrame

_POINT_SIZES = [10_000, 100_000, 500_000, 1_000_000]
_POLY_SIZES = [10_000, 40_000, 100_000]

_SCAN_SIZES = [10_000, 50_000, 100_000]

_Q = 200
_K = 5
_RUNS = 3
_BOX_SIDE = 0.1

_MIN_NS = 0.1  # floor for measured constants
_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "assets" / "ops.txt"  # written report

_SCAN_FIELDS = ["knn_scan_ns_per_item", "bbox_scan_ns_per_item"]
_POINT_FIELDS = [
    "grid_build_ns_per_item",
    "kdtree_build_ns_per_item",
    "grid_range_ns",
    "kdtree_range_ns",
    "kdtree_knn_ns",
]
_POLY_FIELDS = [
    "rtree_build_ns_per_item",
    "rtree_range_ns",
    "rtree_knn_ns",
]


def _query_pts(q: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Return q query points uniform over the unit square
    rng = np.random.default_rng(seed + 1)
    return rng.uniform(0.0, 1.0, q), rng.uniform(0.0, 1.0, q)


def _query_boxes(q: int, seed: int, side: float) -> list[tuple[float, float, float, float]]:
    # Return q axis-aligned query boxes of the given side, anchored uniformly
    rng = np.random.default_rng(seed + 2)
    anchors = rng.uniform(0.0, 1.0 - side, (q, 2))
    return [(ax, ay, ax + side, ay + side) for ax, ay in anchors]


def _point_frame(pts: np.ndarray, mode: str) -> SpatialFrame:
    # Build a point SpatialFrame from an (N, 2) array under the given index mode
    df = pl.DataFrame({"x": pts[:, 0], "y": pts[:, 1]})
    return SpatialFrame(df, "x", "y", index_mode=mode)


def _poly_frame(geoms: list, mode: str) -> SpatialFrame:
    # Build a polygon SpatialFrame from a list of shapely polygons under the given index mode
    df = pl.DataFrame({"geom": geoms})
    return SpatialFrame.from_polygons(df, "geom", index_mode=mode)


def _range_total(sf: SpatialFrame, boxes: list[tuple[float, float, float, float]]) -> int:
    # Sum the actual hit count across every query box, used as the true result term
    return sum(len(sf.engine.range_query(*b)) for b in boxes)


def _run_ranges(sf: SpatialFrame, boxes: list[tuple[float, float, float, float]]) -> None:
    # Execute every range query once, for timing
    for b in boxes:
        sf.engine.range_query(*b)


def _measure_median(measure_fn, sizes: list[int]) -> float:
    # Run measure_fn per N, fold each (time_ms, term) into an ns/op ratio, then take the median
    ratios = []
    for n in sizes:
        t, term = measure_fn(n)
        ratios.append(max(_MIN_NS, t * 1e6 / term))
        gc.collect()
    return float(np.median(ratios))


def _build_probe(n: int, runs: int, seed: int, clustered: bool) -> tuple[float, float]:
    # Time index construction alone (grid for uniform, kd-tree for clustered), fresh each run
    pts = generate_clustered_points(n, seed) if clustered else generate_points(n, seed)

    def build() -> None:
        sf = _point_frame(pts, "none")
        sf.engine.set_index_mode("eager")
        sf.engine.build_index()

    t = time_min(build, runs)
    term = n * math.log2(n) if clustered else n
    return t, term


def _rtree_build_probe(n: int, runs: int, seed: int) -> tuple[float, float]:
    # Time r-tree construction alone over polygons, fresh each run
    geoms = generate_polygons(n, seed).tolist()

    def build() -> None:
        sf = _poly_frame(geoms, "none")
        sf.engine.set_index_mode("eager")
        sf.engine.build_index()

    return time_min(build, runs), n * math.log2(n)


def _scan_probe(
    n: int, runs: int, seed: int, qxs: np.ndarray, qys: np.ndarray
) -> tuple[float, float]:
    # Time a brute-force kNN probe with no index built, over uniform points
    sf = _point_frame(generate_points(n, seed), "none")
    t = time_min(lambda: sf.engine.batch_knn_join(qxs, qys, _K), runs)
    return t, _Q * n


def _bbox_scan_probe(n: int, runs: int, seed: int) -> tuple[float, float]:
    # Time one no-index range scan over an empty box, isolating the per-item box test in a single call
    sf = _point_frame(generate_points(n, seed), "none")
    empty_box = (2.0, 2.0, 3.0, 3.0)
    t = time_min(lambda: sf.engine.range_query(*empty_box), runs)
    return t, n


def _range_probe(
    n: int,
    runs: int,
    seed: int,
    boxes: list[tuple[float, float, float, float]],
    clustered: bool,
) -> tuple[float, float]:
    # Time a range sweep over an already-built index, term is the true hit total plus traversal
    pts = generate_clustered_points(n, seed) if clustered else generate_points(n, seed)
    sf = _point_frame(pts, "eager")
    sf.engine.build_index()
    hits = _range_total(sf, boxes)
    t = time_min(lambda: _run_ranges(sf, boxes), runs)
    term = _Q * math.log2(n) + hits if clustered else hits
    return t, term


def _rtree_range_probe(
    n: int, runs: int, seed: int, boxes: list[tuple[float, float, float, float]]
) -> tuple[float, float]:
    # Time a range sweep over an already-built r-tree, term is q*log2n plus the true hit total
    sf = _poly_frame(generate_polygons(n, seed).tolist(), "eager")
    sf.engine.build_index()
    hits = _range_total(sf, boxes)
    t = time_min(lambda: _run_ranges(sf, boxes), runs)
    return t, _Q * math.log2(n) + hits


def _knn_probe(
    n: int,
    runs: int,
    seed: int,
    qxs: np.ndarray,
    qys: np.ndarray,
    polygons: bool,
) -> tuple[float, float]:
    # Time a batched kNN probe over an already-built index, term is q*(log2n + k)
    if polygons:
        sf = _poly_frame(generate_polygons(n, seed).tolist(), "eager")
        sf.engine.build_index()
        sf.engine.batch_knn_to_polygons(qxs, qys, _K)
        t = time_min(lambda: sf.engine.batch_knn_to_polygons(qxs, qys, _K), runs)
    else:
        sf = _point_frame(generate_clustered_points(n, seed), "eager")
        sf.engine.build_index()
        sf.engine.batch_knn_join(qxs, qys, _K)
        t = time_min(lambda: sf.engine.batch_knn_join(qxs, qys, _K), runs)
    return t, _Q * (math.log2(n) + _K)


def run(runs: int, seed: int) -> None:
    """Run the calibration sweep and print the suggested CostFactors constants.

    Args:
        runs: Timing repetitions per measurement, the minimum is taken.
        seed: RNG seed for data and query generation.
    """
    start = time.perf_counter()
    baseline_mb = peak_rss_mb()
    qxs, qys = _query_pts(_Q, seed)
    boxes = _query_boxes(_Q, seed, _BOX_SIDE)

    fits: dict[str, float] = {
        "knn_scan_ns_per_item": _measure_median(
            lambda n: _scan_probe(n, runs, seed, qxs, qys), _SCAN_SIZES
        ),
        "bbox_scan_ns_per_item": _measure_median(
            lambda n: _bbox_scan_probe(n, runs, seed), _SCAN_SIZES
        ),
        "grid_build_ns_per_item": _measure_median(
            lambda n: _build_probe(n, runs, seed, clustered=False), _POINT_SIZES
        ),
        "kdtree_build_ns_per_item": _measure_median(
            lambda n: _build_probe(n, runs, seed, clustered=True), _POINT_SIZES
        ),
        "rtree_build_ns_per_item": _measure_median(
            lambda n: _rtree_build_probe(n, runs, seed), _POLY_SIZES
        ),
        "kdtree_knn_ns": _measure_median(
            lambda n: _knn_probe(n, runs, seed, qxs, qys, polygons=False), _POINT_SIZES
        ),
        "kdtree_range_ns": _measure_median(
            lambda n: _range_probe(n, runs, seed, boxes, clustered=True), _POINT_SIZES
        ),
        "rtree_knn_ns": _measure_median(
            lambda n: _knn_probe(n, runs, seed, qxs, qys, polygons=True), _POLY_SIZES
        ),
        "rtree_range_ns": _measure_median(
            lambda n: _rtree_range_probe(n, runs, seed, boxes), _POLY_SIZES
        ),
        "grid_range_ns": _measure_median(
            lambda n: _range_probe(n, runs, seed, boxes, clustered=False), _POINT_SIZES
        ),
    }

    lines = [
        "Suggested CostFactors (copy into src/planner/calibration.rs):",
        "",
        *_report_section("Brute Force", _SCAN_FIELDS, fits),
        *_report_section("Points", _POINT_FIELDS, fits),
        *_report_section("Polygons", _POLY_FIELDS, fits),
        *_report_footer(start, baseline_mb),
    ]
    text = "\n".join(lines)
    print(text)
    _OUTPUT_PATH.write_text(text + "\n")


def _report_section(title: str, fields: list[str], fits: dict[str, float]) -> list[str]:
    # List each constant's suggested value as a paste-ready CostFactors line
    lines = [title]
    for name in fields:
        lines.append(f"    {name + ':':<28} {fits[name]:.2f},")
    lines.append("")
    return lines


def _report_footer(start: float, baseline_mb: float) -> list[str]:
    # Report elapsed time and peak RSS, no pass/fail gating
    elapsed = time.perf_counter() - start
    rss = peak_rss_mb() - baseline_mb
    return [f"elapsed: {elapsed:.1f} s   peak RSS: {rss:.1f} MiB"]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the calibration benchmark.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        The process exit code, always 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Measure CostFactors constants from index build and probe times."
    )
    parser.add_argument("--runs", type=int, default=_RUNS, metavar="R")
    parser.add_argument("--seed", type=int, default=42, metavar="S")
    args = parser.parse_args(argv)
    run(args.runs, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
