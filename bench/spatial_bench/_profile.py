"""Profile path for SpatialBench: per-stage time + sampled memory plus oracle verification.

ProfilingTables wraps only fetch / build so query modules stay unchanged. It is entered
only by `_runner --profile` and run_profile_suite."""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from bench.spatial_bench.utils import (
    _ASSETS_DIR,
    SpatialBenchTables,
    spawn_query,
)

# Coarse stages the harness can attribute without touching query or Rust code
_STAGES = ("fetch", "build", "query", "collect")
# How often the background thread samples resident memory, in seconds
_SAMPLE_INTERVAL = 0.02
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_MIB = 1024 * 1024
_SEP = "=" * 64
_SUBSEP = "-" * 64


def _rss_bytes() -> int:
    # Current resident set size in bytes from procfs, falling back to the rusage peak
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except OSError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024


class _StageProfiler:
    """Time each stage and sample resident memory on a background thread for one run.

    Wall time per stage is a perf_counter delta. Memory is process RSS, so it counts the
    Polars / numpy / Rust buffers and not just Python, sampled every _SAMPLE_INTERVAL and
    bucketed by the active stage, with the residual time bucketed as "query".
    """

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.stage_peak: dict[str, int] = {}
        self.current = "query"
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        # Poll RSS until stopped, tracking the overall peak and the active-stage peak
        while not self._stop.wait(_SAMPLE_INTERVAL):
            self._observe()

    def _observe(self) -> None:
        # Record one RSS reading against the overall and current-stage peaks
        rss = _rss_bytes()
        self.peak = max(self.peak, rss)
        self.stage_peak[self.current] = max(self.stage_peak.get(self.current, 0), rss)

    @contextmanager
    def stage(self, name: str):
        """Time the wrapped region under ``name`` and bucket its memory samples to it.

        Args:
            name: Stage label to accumulate into.

        Yields:
            None, for the duration of the wrapped block.
        """
        start = time.perf_counter()
        prev = self.current
        self.current = name
        self._observe()
        try:
            yield
        finally:
            self._observe()
            self.current = prev
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - start

    def stop(self) -> None:
        """Stop the sampler thread and take one final reading."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._observe()


class ProfilingTables(SpatialBenchTables):
    """SpatialBenchTables that times its S3 fetch and frame-build boundaries.

    Each timed method just wraps the parent call, so query modules call the same names
    and the hot-path class stays free of profiling code.
    """

    def __init__(self, data_dir: str, index_mode: str = "eager") -> None:
        super().__init__(data_dir=data_dir, index_mode=index_mode)
        self.profiler = _StageProfiler()

    def parallel_fetch(self, needs: dict[str, list[str] | None]) -> None:
        """Time the concurrent S3 fetch into the fetch stage.

        Args:
            needs: Map of table name to the columns to fetch, or None for all columns.
        """
        with self.profiler.stage("fetch"):
            super().parallel_fetch(needs)

    def table(self, name, columns=None):
        """Time a cold table read into the fetch stage (a warm cache hit is ~free).

        Args:
            name: Table name.
            columns: Optional subset of columns to read.

        Returns:
            The cached table as a Polars DataFrame.
        """
        with self.profiler.stage("fetch"):
            return super().table(name, columns)

    def point_frame(self, df, wkb_col):
        """Time point frame construction (WKB decode + index build) into the build stage.

        Args:
            df: DataFrame holding the WKB point column.
            wkb_col: Name of the WKB point column.

        Returns:
            A point SpatialFrame over ``df``.
        """
        with self.profiler.stage("build"):
            return super().point_frame(df, wkb_col)

    def polygon_frame(self, df, wkb_col):
        """Time polygon frame construction (WKB decode + index build) into the build stage.

        Args:
            df: DataFrame holding the WKB polygon column.
            wkb_col: Name of the WKB polygon column.

        Returns:
            A polygon SpatialFrame over ``df``.
        """
        with self.profiler.stage("build"):
            return super().polygon_frame(df, wkb_col)


def profile_payload(profiler: _StageProfiler, elapsed: float) -> dict:
    """Reduce a finished profiler to a per-stage time and memory payload.

    Args:
        profiler: The stage profiler after the run, with its sampler stopped.
        elapsed: Total wall time of the timed region in seconds.

    Returns:
        A dict with a "time" map (stages plus total) and a "mem" map (RSS bytes: baseline,
        overall peak, and per-stage peak), all order-independent.
    """
    t = profiler.times
    measured = t.get("fetch", 0.0) + t.get("build", 0.0) + t.get("collect", 0.0)
    return {
        "time": {
            "total": elapsed,
            "fetch": t.get("fetch", 0.0),
            "build": t.get("build", 0.0),
            "query": max(elapsed - measured, 0.0),
            "collect": t.get("collect", 0.0),
        },
        "mem": {
            "baseline": profiler.baseline,
            "peak": profiler.peak,
            **{s: profiler.stage_peak.get(s, 0) for s in _STAGES},
        },
    }


def profile_query(query, data_dir: str, index_mode: str) -> dict:
    """Run one query once under profiling and verification, parsing the runner output.

    Args:
        query: Query module exposing id, title, pycanopy(tables), and compare.
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").

    Returns:
        A result dict with status, title, and on success the profile payload and verdict.
    """
    r = spawn_query(query.id, data_dir, index_mode, "--profile")
    if r["status"] != "ok":
        print(f"[testcase] {r['status']} {query.id}: {r.get('error', '')}", flush=True)
        return {"status": r["status"], "title": query.title, "error": r.get("error", "")}

    kv = r["kv"]
    profile = json.loads(kv["PYCANOPY_PROFILE"])
    if "PYCANOPY_MATCH" in kv:
        verify, detail = "match", kv["PYCANOPY_MATCH"]
    elif "PYCANOPY_MISMATCH" in kv:
        verify, detail = "MISMATCH", kv["PYCANOPY_MISMATCH"]
    else:
        verify, detail = "error", kv.get("PYCANOPY_VERIFY_ERROR", "no verification output")

    print(f"[testcase] completed {query.id} in {r['time']:.2f}s [verify: {verify}]", flush=True)
    if verify != "match":
        print(f"[verification] {verify} on {query.id}: {detail}", flush=True)
    return {
        "status": "ok",
        "title": query.title,
        "profile": profile,
        "verify": verify,
        "verify_detail": detail,
    }


def _section(qid: str, r: dict) -> str:
    # Render one per-query block: timing, sampled memory, and the verification verdict
    lines = [_SEP, f"{qid}  {r.get('title', '')}".rstrip(), _SUBSEP]
    if r["status"] != "ok":
        return "\n".join([*lines, f"status        {r['status']}  {r.get('error', '')}".rstrip()])

    t = r["profile"]["time"]
    mib = {k: v / _MIB for k, v in r["profile"]["mem"].items()}
    verdict = {"match": "PASS", "MISMATCH": "FAIL", "error": "ERROR"}.get(r["verify"], r["verify"])
    lines += [
        f"time (s)      total {t['total']:6.2f}   fetch {t['fetch']:5.2f}   "
        f"build {t['build']:5.2f}   query {t['query']:6.2f}   collect {t['collect']:5.2f}",
        f"memory (MiB)  peak {mib['peak']:7.1f}   baseline {mib['baseline']:7.1f}   "
        f"demand {mib['peak'] - mib['baseline']:+7.1f}",
        f"  stage peak    fetch {mib['fetch']:7.1f}   build {mib['build']:7.1f}   "
        f"query {mib['query']:7.1f}   collect {mib['collect']:7.1f}",
        f"verify        {verdict}   {r['verify_detail']}",
    ]
    return "\n".join(lines)


def write_profile(results: dict, index_mode: str, path: Path) -> None:
    """Write the per-query time + memory + verification report to ``path``.

    Args:
        results: Map of query id to its profile_query result dict.
        index_mode: Index policy the run used.
        path: Output text file path.
    """
    head = (
        f"PyCanopy SpatialBench SF1 profile (index_mode={index_mode}, 1 run)\n"
        "Times in seconds, include profiling overhead. Memory is process RSS in MiB, sampled\n"
        f"every {int(_SAMPLE_INTERVAL * 1000)} ms; demand is peak minus the post-import baseline."
    )
    parts = [head, *[_section(qid, r) for qid, r in results.items()], _SEP]
    path.write_text("\n".join(parts) + "\n")


def run_profile_suite(query_modules: list, data_dir: str, index_mode: str = "auto") -> Path:
    """Profile and verify each query once and write assets/profile.txt, returning its path.

    Args:
        query_modules: Query modules to run, each exposing id, pycanopy, and compare.
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").

    Returns:
        The profile.txt path written under assets/.
    """
    results = {query.id: profile_query(query, data_dir, index_mode) for query in query_modules}
    path = _ASSETS_DIR / "profile.txt"
    write_profile(results, index_mode, path)
    return path
