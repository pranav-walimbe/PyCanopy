"""Calibrate the planner's CostFactors from Engine-reported build and probe metrics."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from bench.ops.utils import (
    generate_clustered_points,
    generate_points,
    generate_polygons,
    peak_rss_mb,
)
from pycanopy import SpatialFrame
from pycanopy.engine import Engine

_POINT_SIZES = [10_000, 100_000, 1_000_000]
_POLY_SIZES = [10_000, 50_000, 100_000]
_Q = 100
_BASE_K = 5
_EXTRA_K = [1, 50]
_BASE_SELECTIVITY = 0.01
_EXTRA_SELECTIVITY = [0.001, 0.1]
_PROBE_RUNS = 5
_BUILD_RUNS = 3
_MIN_NS = 0.1

_ROOT = Path(__file__).resolve().parents[2]
_REPORT_PATH = _ROOT / "assets" / "ops.txt"
_PROFILE_PATH = _ROOT / "python" / "pycanopy" / "cost_profiles" / "default.json"

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
_ALL_FIELDS = _SCAN_FIELDS + _POINT_FIELDS + _POLY_FIELDS


@dataclass(frozen=True)
class _Observation:
    """One workload term paired with the median Engine time for that configuration."""

    term: float
    elapsed_ns: float


def _query_pts(q: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    return rng.uniform(0.0, 1.0, q), rng.uniform(0.0, 1.0, q)


def _uniform_query_boxes(
    q: int, seed: int, selectivity: float
) -> list[tuple[float, float, float, float]]:
    side = math.sqrt(selectivity)
    rng = np.random.default_rng(seed)
    anchors = rng.uniform(0.0, 1.0 - side, (q, 2))
    return [(ax, ay, ax + side, ay + side) for ax, ay in anchors]


def _populated_query_boxes(
    points: np.ndarray, q: int, seed: int, selectivity: float
) -> list[tuple[float, float, float, float]]:
    """Create boxes containing known points so histogram early-exit cannot skip the probe."""
    side = math.sqrt(selectivity)
    rng = np.random.default_rng(seed)
    chosen = points[rng.integers(0, len(points), q)]
    offsets = rng.uniform(0.0, side, (q, 2))
    anchors = np.clip(chosen - offsets, 0.0, 1.0 - side)
    return [(ax, ay, ax + side, ay + side) for ax, ay in anchors]


def _point_frame(points: np.ndarray, mode: str) -> SpatialFrame:
    df = pl.DataFrame({"x": points[:, 0], "y": points[:, 1]})
    return SpatialFrame(df, "x", "y", index_mode=mode)


def _poly_frame(geometries: list, mode: str) -> SpatialFrame:
    return SpatialFrame.from_polygons(pl.DataFrame({"geom": geometries}), "geom", index_mode=mode)


def _metric(
    metrics: dict[str, object], collection: str, *, index: str, name: str | None = None
) -> dict[str, object]:
    items = metrics[collection]
    matches = [
        item
        for item in items
        if item["index"] == index and (name is None or item.get("name") == name)
    ]
    if len(matches) != 1:
        label = f"{name}/{index}" if name else index
        raise RuntimeError(f"expected one {collection} metric for {label}, got {matches}")
    return matches[0]


def _measure_build(
    frame_factory: Callable[[], SpatialFrame], index: str, runs: int, term: float
) -> _Observation:
    samples = []
    for _ in range(runs):
        frame = frame_factory()
        frame.engine.take_metrics()
        frame.engine._build_index_for_calibration(index)
        build = _metric(frame.engine.take_metrics(), "index_builds", index=index)
        if build["build_count"] != 1:
            raise RuntimeError(f"expected one fresh {index} build, got {build['build_count']}")
        samples.append(int(build["elapsed_compute_ns"]))
        del frame
    return _Observation(term, float(np.median(samples)))


def _measure_operation(
    engine: Engine,
    operation: Callable[[], object],
    *,
    name: str,
    index: str,
    calls: int,
    runs: int,
) -> tuple[float, int]:
    """Warm once, then return median Engine time and output rows from repeated samples."""
    engine.take_metrics()
    operation()
    warm = _metric(engine.take_metrics(), "operations", name=name, index=index)
    if warm["calls"] != calls:
        raise RuntimeError(f"warm-up for {name}/{index} recorded {warm['calls']} of {calls} calls")

    elapsed = []
    output_rows = []
    for _ in range(runs):
        operation()
        sample = _metric(engine.take_metrics(), "operations", name=name, index=index)
        if sample["calls"] != calls:
            raise RuntimeError(
                f"sample for {name}/{index} recorded {sample['calls']} of {calls} calls"
            )
        elapsed.append(int(sample["elapsed_compute_ns"]))
        output_rows.append(int(sample["output_rows"]))
    if len(set(output_rows)) != 1:
        raise RuntimeError(f"non-deterministic output row counts for {name}/{index}: {output_rows}")
    return float(np.median(elapsed)), output_rows[0]


def _run_ranges(engine: Engine, boxes: list[tuple[float, float, float, float]]) -> None:
    for box in boxes:
        engine.range_query(*box)


def _case_values(n: int, middle_n: int, base: float | int, extras: list) -> list:
    return [base, *extras] if n == middle_n else [base]


def _build_observations(build_runs: int, seed: int) -> dict[str, list[_Observation]]:
    observations = {
        "grid_build_ns_per_item": [],
        "kdtree_build_ns_per_item": [],
        "rtree_build_ns_per_item": [],
    }
    for n in _POINT_SIZES:
        uniform = generate_points(n, seed)
        observations["grid_build_ns_per_item"].append(
            _measure_build(
                lambda points=uniform: _point_frame(points, "none"), "grid", build_runs, n
            )
        )
        del uniform
        gc.collect()

        clustered = generate_clustered_points(n, seed)
        observations["kdtree_build_ns_per_item"].append(
            _measure_build(
                lambda points=clustered: _point_frame(points, "none"),
                "kd_tree",
                build_runs,
                n * math.log2(n),
            )
        )
        del clustered
        gc.collect()

    for n in _POLY_SIZES:
        polygons = generate_polygons(n, seed).tolist()
        observations["rtree_build_ns_per_item"].append(
            _measure_build(
                lambda geometries=polygons: _poly_frame(geometries, "none"),
                "r_tree",
                build_runs,
                n * math.log2(n),
            )
        )
        del polygons
        gc.collect()
    return observations


def _scan_observations(
    probe_runs: int, seed: int, query_xs: np.ndarray, query_ys: np.ndarray
) -> dict[str, list[_Observation]]:
    observations = {"knn_scan_ns_per_item": [], "bbox_scan_ns_per_item": []}
    for n in _POINT_SIZES:
        points = generate_points(n, seed)
        frame = _point_frame(points, "none")
        engine = frame.engine

        elapsed, _ = _measure_operation(
            engine,
            lambda engine=engine: engine.batch_knn_join(query_xs, query_ys, _BASE_K),
            name="batch_knn_join",
            index="brute_force",
            calls=1,
            runs=probe_runs,
        )
        observations["knn_scan_ns_per_item"].append(_Observation(_Q * n, elapsed))

        x, y = points[0]
        epsilon = 1e-12
        box = (x - epsilon, y - epsilon, x + epsilon, y + epsilon)
        elapsed, _ = _measure_operation(
            engine,
            lambda engine=engine: engine.range_query(*box),
            name="range_query",
            index="brute_force",
            calls=1,
            runs=probe_runs,
        )
        observations["bbox_scan_ns_per_item"].append(_Observation(n, elapsed))
        del engine, frame, points
        gc.collect()
    return observations


def _point_range_observations(probe_runs: int, seed: int, *, clustered: bool) -> list[_Observation]:
    index = "kd_tree" if clustered else "grid"
    middle_n = _POINT_SIZES[1]
    observations = []
    for n in _POINT_SIZES:
        points = generate_clustered_points(n, seed) if clustered else generate_points(n, seed)
        frame = _point_frame(points, "eager")
        engine = frame.engine
        engine._build_index_for_calibration(index)
        for selectivity in _case_values(n, middle_n, _BASE_SELECTIVITY, _EXTRA_SELECTIVITY):
            boxes = _populated_query_boxes(points, _Q, seed + int(selectivity * 1e6), selectivity)
            elapsed, rows = _measure_operation(
                engine,
                lambda boxes=boxes, engine=engine: _run_ranges(engine, boxes),
                name="range_query",
                index=index,
                calls=_Q,
                runs=probe_runs,
            )
            term = _Q * math.log2(n) + rows if clustered else max(1, rows)
            observations.append(_Observation(term, elapsed))
        del engine, frame, points
        gc.collect()
    return observations


def _rtree_range_observations(probe_runs: int, seed: int) -> list[_Observation]:
    middle_n = _POLY_SIZES[1]
    observations = []
    for n in _POLY_SIZES:
        polygons = generate_polygons(n, seed).tolist()
        frame = _poly_frame(polygons, "eager")
        engine = frame.engine
        engine._build_index_for_calibration("r_tree")
        for selectivity in _case_values(n, middle_n, _BASE_SELECTIVITY, _EXTRA_SELECTIVITY):
            boxes = _uniform_query_boxes(_Q, seed + int(selectivity * 1e6), selectivity)
            elapsed, rows = _measure_operation(
                engine,
                lambda boxes=boxes, engine=engine: _run_ranges(engine, boxes),
                name="range_query",
                index="r_tree",
                calls=_Q,
                runs=probe_runs,
            )
            observations.append(_Observation(_Q * math.log2(n) + rows, elapsed))
        del engine, frame, polygons
        gc.collect()
    return observations


def _knn_observations(
    probe_runs: int,
    seed: int,
    query_xs: np.ndarray,
    query_ys: np.ndarray,
    *,
    polygons: bool,
) -> list[_Observation]:
    sizes = _POLY_SIZES if polygons else _POINT_SIZES
    middle_n = sizes[1]
    index = "r_tree" if polygons else "kd_tree"
    name = "batch_knn_to_polygons" if polygons else "batch_knn_join"
    observations = []
    for n in sizes:
        if polygons:
            data = generate_polygons(n, seed).tolist()
            frame = _poly_frame(data, "eager")
        else:
            data = generate_clustered_points(n, seed)
            frame = _point_frame(data, "eager")
        engine = frame.engine
        engine._build_index_for_calibration(index)

        for k in _case_values(n, middle_n, _BASE_K, _EXTRA_K):
            operation = (
                (lambda k=k, engine=engine: engine.batch_knn_to_polygons(query_xs, query_ys, k))
                if polygons
                else (lambda k=k, engine=engine: engine.batch_knn_join(query_xs, query_ys, k))
            )
            elapsed, _ = _measure_operation(
                engine,
                operation,
                name=name,
                index=index,
                calls=1,
                runs=probe_runs,
            )
            observations.append(_Observation(_Q * (math.log2(n) + k), elapsed))
        del operation, engine, frame, data
        gc.collect()
    return observations


def _fit_factor(observations: list[_Observation]) -> float:
    """Fit elapsed_ns = factor * workload_term through the origin."""
    numerator = sum(observation.term * observation.elapsed_ns for observation in observations)
    denominator = sum(observation.term**2 for observation in observations)
    if denominator <= 0:
        raise ValueError("cannot fit a cost factor without a positive workload term")
    return max(_MIN_NS, numerator / denominator)


def _write_profile(path: Path, factors: dict[str, float]) -> None:
    """Atomically replace a cost profile after every calibration measurement succeeds."""
    profile = {"cost_factors": {name: round(factors[name], 6) for name in _ALL_FIELDS}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            json.dump(profile, temporary, indent=2)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _report_section(title: str, fields: list[str], fits: dict[str, float]) -> list[str]:
    lines = [title]
    for name in fields:
        lines.append(f"    {name + ':':<28} {fits[name]:.3f}")
    lines.append("")
    return lines


def run(probe_runs: int, build_runs: int, seed: int, *, dry_run: bool) -> dict[str, float]:
    """Run the balanced calibration matrix and optionally update the bundled profile."""
    start = time.perf_counter()
    baseline_mb = peak_rss_mb()
    query_xs, query_ys = _query_pts(_Q, seed)

    observations: dict[str, list[_Observation]] = {name: [] for name in _ALL_FIELDS}
    for name, values in _build_observations(build_runs, seed).items():
        observations[name].extend(values)
    for name, values in _scan_observations(probe_runs, seed, query_xs, query_ys).items():
        observations[name].extend(values)
    observations["grid_range_ns"] = _point_range_observations(probe_runs, seed, clustered=False)
    observations["kdtree_range_ns"] = _point_range_observations(probe_runs, seed, clustered=True)
    observations["rtree_range_ns"] = _rtree_range_observations(probe_runs, seed)
    observations["kdtree_knn_ns"] = _knn_observations(
        probe_runs, seed, query_xs, query_ys, polygons=False
    )
    observations["rtree_knn_ns"] = _knn_observations(
        probe_runs, seed, query_xs, query_ys, polygons=True
    )

    fits = {name: _fit_factor(observations[name]) for name in _ALL_FIELDS}
    if not dry_run:
        _write_profile(_PROFILE_PATH, fits)

    destination = "not written (--dry-run)" if dry_run else str(_PROFILE_PATH)
    lines = [
        "Calibrated CostFactors:",
        "",
        *_report_section("Brute Force", _SCAN_FIELDS, fits),
        *_report_section("Points", _POINT_FIELDS, fits),
        *_report_section("Polygons", _POLY_FIELDS, fits),
        f"profile: {destination}",
        f"elapsed: {time.perf_counter() - start:.1f} s   peak RSS: {peak_rss_mb() - baseline_mb:.1f} MiB",
    ]
    report = "\n".join(lines)
    print(report)
    _REPORT_PATH.write_text(report + "\n")
    return fits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the planner CostFactors from Engine metrics and update default.json."
    )
    parser.add_argument("--runs", type=int, default=_PROBE_RUNS, metavar="R")
    parser.add_argument("--build-runs", type=int, default=_BUILD_RUNS, metavar="R")
    parser.add_argument("--seed", type=int, default=42, metavar="S")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print fitted constants without updating default.json",
    )
    args = parser.parse_args(argv)
    if args.runs < 1 or args.build_runs < 1:
        parser.error("--runs and --build-runs must both be at least 1")
    run(args.runs, args.build_runs, args.seed, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
