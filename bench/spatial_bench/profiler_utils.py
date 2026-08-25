"""Profile instrumentation and answer verification for SpatialBench profile runs."""

from __future__ import annotations

import inspect
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl

from bench.spatial_bench.config import ANSWERS_DIR, NS_PER_SECOND, RSS_SAMPLE_INTERVAL

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _rss_bytes() -> int:
    # Current process RSS from procfs, falling back to the rusage peak where unavailable
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * _PAGE_SIZE
    except OSError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024


class StageProfiler:
    """Measure honest harness boundaries and sample the process RSS peak alongside them."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        # Background thread body, observing RSS until stop() is called
        while not self._stop.wait(RSS_SAMPLE_INTERVAL):
            self._observe()

    def _observe(self) -> None:
        # Fold one RSS reading into the run's peak
        self.peak = max(self.peak, _rss_bytes())

    @contextmanager
    def stage(self, name: str):
        """Accumulate wall time under one named harness boundary.

        Args:
            name: Stage name recorded in times.

        Yields:
            None, for the duration of the stage.
        """
        started = time.perf_counter()
        self._observe()
        try:
            yield
        finally:
            self._observe()
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - started

    def stop(self) -> None:
        """Stop the sampler and take a final RSS observation."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._observe()


# Host calls no Engine metric can see paired with the stage each belongs to
_HOST_STAGES = (
    ("read", "pl", "read_parquet"),
    ("read", "pl", "collect_all"),
    ("wkb decode", None, "wkb_points_to_xy"),
    ("wkb decode", None, "wkb_point_distance"),
    ("frame build", "SpatialFrame", "from_wkb_points"),
    ("frame build", "SpatialFrame", "from_wkb_polygons"),
)


def _timed_call(profiler: StageProfiler, stage: str, call):
    # Wrap one host call and accumulate its wall time under a named stage
    def wrapper(*args, **kwargs):
        with profiler.stage(stage):
            return call(*args, **kwargs)

    return wrapper


def _patch_target(profiler: StageProfiler, module, stage: str, holder_name, attr):
    # Replace one name with a timed wrapper and return what restores it
    if holder_name is None:
        original = module.__dict__.get(attr)
        if original is None:
            return None
        module.__dict__[attr] = _timed_call(profiler, stage, original)
        return (module.__dict__, attr, original)
    holder = module.__dict__.get(holder_name)
    original = getattr(holder, attr, None) if holder is not None else None
    if original is None:
        return None
    wrapper = _timed_call(profiler, stage, original)
    restore = vars(holder).get(attr, original)
    setattr(holder, attr, staticmethod(wrapper) if inspect.isclass(holder) else wrapper)
    return (holder, attr, restore)


@contextmanager
def instrument_host_stages(profiler: StageProfiler, module):
    """Time the read and decode work that runs outside the Engine.

    Args:
        profiler: Stage profiler that accumulates the timings.
        module: Query module whose globals hold the names to patch.

    Yields:
        None, for the duration of the instrumented region.
    """
    restore = [_patch_target(profiler, module, *target) for target in _HOST_STAGES]
    try:
        yield
    finally:
        for entry in reversed(restore):
            if entry is None:
                continue
            holder, attr, original = entry
            if isinstance(holder, dict):
                holder[attr] = original
            else:
                setattr(holder, attr, original)


def _aggregate_engine_metrics(engines: list[dict]) -> dict:
    # Fold every Engine created during the run into one set of totals
    construction = {"wkb_decode_ns": 0, "statistics_ns": 0}
    builds: dict[str, dict] = {}
    operations: dict[tuple[str, str], dict] = {}
    for engine in engines:
        for name in construction:
            construction[name] += engine["construction"].get(name, 0)
        for metric in engine["index_builds"]:
            aggregate = builds.setdefault(
                metric["index"],
                {"index": metric["index"], "build_count": 0, "elapsed_compute_ns": 0},
            )
            aggregate["build_count"] += metric["build_count"]
            aggregate["elapsed_compute_ns"] += metric["elapsed_compute_ns"]
        for metric in engine["operations"]:
            key = (metric["name"], metric["index"])
            aggregate = operations.setdefault(
                key,
                {
                    "name": metric["name"],
                    "index": metric["index"],
                    "calls": 0,
                    "elapsed_compute_ns": 0,
                    "output_rows": 0,
                },
            )
            aggregate["calls"] += metric["calls"]
            aggregate["elapsed_compute_ns"] += metric["elapsed_compute_ns"]
            aggregate["output_rows"] += metric["output_rows"]
    return {
        "construction": construction,
        "index_builds": [builds[key] for key in sorted(builds)],
        "operations": [operations[key] for key in sorted(operations)],
        "engines": engines,
    }


def profile_payload(
    profiler: StageProfiler, elapsed: float, engines: list[dict], metrics: bool = True
) -> dict:
    """Build the raw profile payload from harness boundaries and Engine-reported work.

    Args:
        profiler: The stage profiler that observed the run.
        elapsed: Total wall time of the timed region, in seconds.
        engines: Per-Engine metric dicts captured during the run.
        metrics: False when the installed PyCanopy exports no metrics hook, which makes
            the Engine section empty rather than genuinely zero.

    Returns:
        A JSON-serialisable payload with time, memory, host stage and Engine sections.
    """
    engine = _aggregate_engine_metrics(engines)
    engine_ns = sum(engine["construction"].values())
    engine_ns += sum(metric["elapsed_compute_ns"] for metric in engine["index_builds"])
    engine_ns += sum(metric["elapsed_compute_ns"] for metric in engine["operations"])
    return {
        "time": {
            "total": elapsed,
            "non_engine": max(elapsed - engine_ns / NS_PER_SECOND, 0.0),
        },
        "mem": {
            "baseline": profiler.baseline,
            "peak": profiler.peak,
        },
        "stages": dict(profiler.times),
        "engine": engine,
        "metrics": metrics,
    }


_LIMIT_QUERIES = {"q1", "q5", "q7", "q9", "q10", "q12"}
_LIMIT_CAP = 100
_RTOL = 1e-6
_ATOL = 1e-9
_DURATION_ATOL = 1e-3


def _as_frame(result) -> pl.DataFrame:
    # Materialize supported result types without changing their row order
    if isinstance(result, pl.LazyFrame):
        return result.collect()
    if isinstance(result, pl.DataFrame):
        return result
    if hasattr(result, "to_arrow"):
        return pl.from_arrow(result.to_arrow())
    return pl.DataFrame(result)


def _column_kind(dtype: pl.DataType) -> str:
    # Classify the canonical answer type that controls comparison semantics
    if dtype.is_float():
        return "float"
    if dtype.is_integer():
        return "int"
    if dtype.is_temporal():
        return "temporal"
    return "exact"


def _compare(answer: pl.DataFrame, result: pl.DataFrame) -> list[dict]:
    # Compare columns by position while deriving semantics from the answer schema
    if answer.width != result.width:
        return [
            {
                "kind": "shape",
                "first_row": -1,
                "count": 0,
                "message": (
                    f"column count differs: answer {answer.width} ({answer.columns}) "
                    f"vs pycanopy {result.width} ({result.columns})"
                ),
            }
        ]

    issues: list[dict] = []
    if answer.height != result.height:
        issues.append(
            {
                "kind": "shape",
                "first_row": -1,
                "count": abs(answer.height - result.height),
                "message": (
                    f"row count differs: answer {answer.height} vs pycanopy {result.height}"
                ),
            }
        )

    size = min(answer.height, result.height)
    for position, answer_name in enumerate(answer.columns):
        dtype = answer.schema[answer_name]
        kind = _column_kind(dtype)
        expected = answer.get_column(answer_name).head(size)
        actual = result.get_column(result.columns[position]).head(size)

        if kind == "float":
            expected_values = expected.cast(pl.Float64).to_numpy()
            actual_values = actual.cast(pl.Float64, strict=False).to_numpy()
            atol = _DURATION_ATOL if answer_name.endswith("_seconds") else _ATOL
            bad = ~np.isclose(
                expected_values,
                actual_values,
                rtol=_RTOL,
                atol=atol,
                equal_nan=True,
            )
        else:
            actual = actual.cast(dtype, strict=False)
            bad = np.asarray(
                [left != right for left, right in zip(expected.to_list(), actual.to_list())],
                dtype=bool,
            )

        count = int(bad.sum())
        if count:
            first = int(np.flatnonzero(bad)[0])
            issues.append(
                {
                    "kind": kind,
                    "first_row": first,
                    "count": count,
                    "column": position,
                    "name": answer_name,
                    "message": (
                        f"column {position} ({answer_name!r}, {kind}): {count}/{size} differ "
                        f"starting at row {first}"
                    ),
                }
            )
    return issues


def _is_boundary_tie(query_id: str, issues: list[dict], answer: pl.DataFrame) -> bool:
    # Accept only a key or string swap in the final row of an eligible capped query
    if not issues or query_id not in _LIMIT_QUERIES or answer.height != _LIMIT_CAP:
        return False
    final_row = answer.height - 1
    return all(
        issue["kind"] != "shape"
        and issue["kind"] != "float"
        and issue["first_row"] == final_row
        and issue["count"] == 1
        for issue in issues
    )


def verify_output(
    result,
    query_id: str,
    scale_factor: int,
    answers_dir: Path | None = None,
) -> tuple[bool, str]:
    """Compare one result with the pinned upstream SpatialBench answer.

    Args:
        result: Materialized or lazy PyCanopy result frame.
        query_id: SpatialBench query identifier.
        scale_factor: Dataset scale factor, currently 1 or 10.
        answers_dir: Optional answer root used by tests and local tooling.

    Returns:
        Whether the result matches and a concise comparison detail.
    """
    root = answers_dir or ANSWERS_DIR
    answer_path = root / f"sf{scale_factor}" / f"{query_id}.parquet"
    if not answer_path.is_file():
        raise FileNotFoundError(f"no committed SpatialBench answer at {answer_path}")

    answer = pl.read_parquet(answer_path)
    actual = _as_frame(result)
    issues = _compare(answer, actual)
    if not issues:
        return True, f"{actual.height} ordered rows match upstream answer"
    if _is_boundary_tie(query_id, issues, answer):
        return True, f"{actual.height} ordered rows match with final-row boundary tie"
    return False, "; ".join(issue["message"] for issue in issues[:3])
