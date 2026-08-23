"""Isolated subprocess entry point: run exactly one SpatialBench query and report its timing."""

from __future__ import annotations

import argparse
import json
import sys
import time

from bench.spatial_bench.config import (
    ENGINE_IDS,
    INDEX_MODES,
    RUNNER_PREFIX,
    SUPPORTED_SCALE_FACTORS,
)
from bench.spatial_bench.engines import load_runner


def _emit(key: str, value) -> None:
    # Structured stdout line, the only channel back to the suite driver
    print(f"{RUNNER_PREFIX}_{key}={value}", flush=True)


def _materialize(result):
    # Force the result to a concrete frame so the full pipeline runs inside the timed region
    if hasattr(result, "collect"):
        result = result.collect()
    _ = len(result)
    return result


def _run_timed(engine: str, query_id: str, data_dir: str, index_mode: str) -> None:
    # Ordinary measurement path: prepare the engine, then time execution plus materialization
    runner = load_runner(engine)
    try:
        runner.prepare(data_dir, index_mode)
        started = time.perf_counter()
        result = runner.execute(query_id)
        materialize_started = time.perf_counter()
        _materialize(result)
        finished = time.perf_counter()
    finally:
        runner.close()
    _emit("MATERIALIZE", f"{finished - materialize_started:.4f}")
    _emit("TIME", f"{finished - started:.6f}")


def _run_profiled(query_id: str, data_dir: str, index_mode: str, scale_factor: int) -> None:
    # Profile path: PyCanopy only, with stage boundaries, Engine metrics, and answer
    # verification. These imports are deferred because an ordinary run of another engine
    # installs neither PyCanopy nor Polars on the box.
    from bench.spatial_bench.fetch_utils import ProfilingTables  # noqa: PLC0415
    from bench.spatial_bench.profiler_utils import (  # noqa: PLC0415
        profile_payload,
        verify_output,
    )
    from bench.spatial_bench.queries import pycanopy as pycanopy_queries  # noqa: PLC0415
    from pycanopy.engine import _capture_engine_metrics  # noqa: PLC0415

    query = pycanopy_queries.BY_ID.get(query_id)
    if query is None:
        raise SystemExit(f"unknown query {query_id!r}")

    tables = ProfilingTables(data_dir=data_dir, index_mode=index_mode)
    with _capture_engine_metrics() as capture:
        started = time.perf_counter()
        result = query.pycanopy(tables)
        with tables.profiler.stage("materialize"):
            result = _materialize(result)
        elapsed = time.perf_counter() - started
        engine_metrics = capture.take_metrics()
    tables.profiler.stop()

    _emit("TIME", f"{elapsed:.6f}")
    _emit("PROFILE", json.dumps(profile_payload(tables.profiler, elapsed, engine_metrics)))
    try:
        matched, detail = verify_output(result, query_id, scale_factor)
        _emit("MATCH" if matched else "MISMATCH", detail)
    except Exception as exc:
        _emit("VERIFY_ERROR", f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    """Run one query in this interpreter and print its structured result to stdout.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        The process exit code, 0 on success and 1 when the query fails.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("engine", choices=ENGINE_IDS)
    parser.add_argument("query_id")
    parser.add_argument("data_dir")
    parser.add_argument("index_mode", choices=INDEX_MODES)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--scale-factor", type=int, choices=SUPPORTED_SCALE_FACTORS, default=1)
    args = parser.parse_args(argv)

    if args.profile and args.engine != "pycanopy":
        _emit("ERROR", f"profile mode measures pycanopy, not {args.engine}")
        return 1

    try:
        if args.profile:
            _run_profiled(args.query_id, args.data_dir, args.index_mode, args.scale_factor)
        else:
            _run_timed(args.engine, args.query_id, args.data_dir, args.index_mode)
    except Exception as exc:
        _emit("ERROR", f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
