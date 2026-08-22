"""SpatialBench profile mode: Engine metrics, sampled RSS, and oracle verification."""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from bench.spatial_bench._verify import DATASET_VERSION, WORKLOAD_REVISION
from bench.spatial_bench.utils import _ASSETS_DIR, SpatialBenchTables, spawn_query

_STAGES = ("fetch", "execute", "materialize")
_SAMPLE_INTERVAL = 0.02
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_MIB = 1024 * 1024
_NS_PER_SECOND = 1_000_000_000
_SEP = "=" * 76
_SUBSEP = "-" * 76


def _rss_bytes() -> int:
    # Current process RSS from procfs, falling back to the rusage peak where unavailable.
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except OSError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024


class _StageProfiler:
    """Measure honest harness boundaries and sample process RSS against the active boundary."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.stage_peak: dict[str, int] = {}
        self.current = "execute"
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(_SAMPLE_INTERVAL):
            self._observe()

    def _observe(self) -> None:
        rss = _rss_bytes()
        self.peak = max(self.peak, rss)
        self.stage_peak[self.current] = max(self.stage_peak.get(self.current, 0), rss)

    @contextmanager
    def stage(self, name: str):
        """Accumulate wall time and sampled RSS under one non-overlapping harness boundary."""
        started = time.perf_counter()
        previous = self.current
        self.current = name
        self._observe()
        try:
            yield
        finally:
            self._observe()
            self.current = previous
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - started

    def stop(self) -> None:
        """Stop the sampler and take a final RSS observation."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._observe()


class ProfilingTables(SpatialBenchTables):
    """SpatialBench tables that identify fetch wall time without timing Engine work."""

    def __init__(self, data_dir: str, index_mode: str = "eager") -> None:
        super().__init__(data_dir=data_dir, index_mode=index_mode)
        self.profiler = _StageProfiler()

    def parallel_fetch(
        self, needs: dict[str, list[str] | None]
    ) -> dict[str, pl.DataFrame]:
        """Fetch several tables while attributing the boundary to fetch."""
        with self.profiler.stage("fetch"):
            return super().parallel_fetch(needs)

    def table(self, name, columns=None):
        """Read an uncached table while attributing the boundary to fetch."""
        with self.profiler.stage("fetch"):
            return super().table(name, columns)


def _aggregate_engine_metrics(engines: list[dict]) -> dict:
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


def profile_payload(profiler: _StageProfiler, elapsed: float, engines: list[dict]) -> dict:
    """Build the raw profile payload from harness boundaries and Engine-reported work."""
    engine = _aggregate_engine_metrics(engines)
    engine_ns = sum(engine["construction"].values())
    engine_ns += sum(metric["elapsed_compute_ns"] for metric in engine["index_builds"])
    engine_ns += sum(metric["elapsed_compute_ns"] for metric in engine["operations"])
    fetch = profiler.times.get("fetch", 0.0)
    materialize = profiler.times.get("materialize", 0.0)
    return {
        "time": {
            "total": elapsed,
            "fetch": fetch,
            "execute": max(elapsed - fetch - materialize, 0.0),
            "materialize": materialize,
            "non_engine": max(elapsed - fetch - engine_ns / _NS_PER_SECOND, 0.0),
        },
        "mem": {
            "baseline": profiler.baseline,
            "peak": profiler.peak,
            **{stage: profiler.stage_peak.get(stage, profiler.baseline) for stage in _STAGES},
        },
        "engine": engine,
    }


def profile_query(query, data_dir: str, index_mode: str) -> dict:
    """Run one profiled SF1 query and parse its answer verification."""
    result = spawn_query(query.id, data_dir, 1, index_mode, "--profile")
    if result["status"] != "ok":
        print(f"[testcase] {result['status']} {query.id}: {result.get('error', '')}", flush=True)
        return {
            "status": result["status"],
            "title": query.title,
            "error": result.get("error", ""),
        }

    kv = result["kv"]
    profile = json.loads(kv["PYCANOPY_PROFILE"])
    if "PYCANOPY_MATCH" in kv:
        verify, detail = "match", kv["PYCANOPY_MATCH"]
    elif "PYCANOPY_MISMATCH" in kv:
        verify, detail = "MISMATCH", kv["PYCANOPY_MISMATCH"]
    else:
        verify, detail = "error", kv.get("PYCANOPY_VERIFY_ERROR", "no verification output")

    print(
        f"[testcase] completed {query.id} in {result['time']:.2f}s [verify: {verify}]", flush=True
    )
    if verify != "match":
        print(f"[verification] {verify} on {query.id}: {detail}", flush=True)
    return {
        "status": "ok",
        "title": query.title,
        "profile": profile,
        "verify": verify,
        "verify_detail": detail,
    }


def _section(qid: str, result: dict) -> str:
    lines = [_SEP, f"{qid}  {result.get('title', '')}".rstrip(), _SUBSEP]
    if result["status"] != "ok":
        return "\n".join(
            [*lines, f"status        {result['status']}  {result.get('error', '')}".rstrip()]
        )

    profile = result["profile"]
    wall = profile["time"]
    mib = {name: value / _MIB for name, value in profile["mem"].items()}
    engine = profile["engine"]
    construction = engine["construction"]
    verdict = {"match": "PASS", "MISMATCH": "FAIL", "error": "ERROR"}.get(
        result["verify"], result["verify"]
    )
    lines += [
        f"wall (s)      total {wall['total']:7.3f}  fetch {wall['fetch']:7.3f}  "
        f"execute {wall['execute']:7.3f}  materialize {wall['materialize']:7.3f}",
        f"               non-engine wall after fetch {wall['non_engine']:7.3f}",
        f"memory (MiB)  peak {mib['peak']:8.1f}  baseline {mib['baseline']:8.1f}  "
        f"demand {mib['peak'] - mib['baseline']:+8.1f}",
        f"  stage peak  fetch {mib['fetch']:8.1f}  execute {mib['execute']:8.1f}  "
        f"materialize {mib['materialize']:8.1f}",
        f"engines       {len(engine['engines'])}  WKB decode "
        f"{construction['wkb_decode_ns'] / _NS_PER_SECOND:7.3f}s  statistics "
        f"{construction['statistics_ns'] / _NS_PER_SECOND:7.3f}s",
    ]
    if engine["index_builds"]:
        lines.append("index builds")
        for metric in engine["index_builds"]:
            lines.append(
                f"  {metric['index']:<26} calls {metric['build_count']:5,d}  "
                f"time {metric['elapsed_compute_ns'] / _NS_PER_SECOND:9.4f}s"
            )
    if engine["operations"]:
        lines.append("engine operations")
        for metric in engine["operations"]:
            lines.append(
                f"  {metric['name']:<36} {metric['index']:<12} calls {metric['calls']:5,d}  "
                f"rows {metric['output_rows']:12,d}  "
                f"time {metric['elapsed_compute_ns'] / _NS_PER_SECOND:9.4f}s"
            )
    lines.append(f"verify        {verdict}   {result['verify_detail']}")
    return "\n".join(lines)


def write_profile(results: dict, index_mode: str, path: Path) -> None:
    """Write the human-readable profile report."""
    head = (
        f"PyCanopy Apache SpatialBench SF1 profile (index_mode={index_mode}, 1 run/query)\n"
        f"Dataset {DATASET_VERSION}, workload revision {WORKLOAD_REVISION}\n"
        "Engine times are always-on production metrics. Harness stages are wall boundaries.\n"
        f"RSS is sampled every {int(_SAMPLE_INTERVAL * 1000)} ms. Verification uses the "
        "committed upstream answers."
    )
    parts = [head, *[_section(qid, result) for qid, result in results.items()], _SEP]
    path.write_text("\n".join(parts) + "\n")


def run_profile_suite(query_modules: list, data_dir: str, index_mode: str = "auto") -> Path:
    """Profile and oracle-verify every selected SF1 query exactly once."""
    results = {query.id: profile_query(query, data_dir, index_mode) for query in query_modules}
    text_path = _ASSETS_DIR / "profile.txt"
    write_profile(results, index_mode, text_path)

    invalid = [
        qid
        for qid, result in results.items()
        if result["status"] != "ok" or result.get("verify") != "match"
    ]
    if invalid:
        raise RuntimeError(f"profile verification failed for: {', '.join(invalid)}")
    return text_path
