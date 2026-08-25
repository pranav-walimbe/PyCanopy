"""Isolated subprocess entry point: run exactly one SpatialBench query and report its timing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager

from bench.spatial_bench.config import ENGINE_IDS, RUNNER_PREFIX, SUPPORTED_SCALE_FACTORS
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


def _run_timed(engine: str, query_id: str, data_dir: str) -> None:
    # Ordinary measurement path: prepare the engine, then time execution plus materialization
    runner = load_runner(engine)
    try:
        runner.prepare(data_dir)
        started = time.perf_counter()
        result = runner.execute(query_id)
        materialize_started = time.perf_counter()
        _materialize(result)
        finished = time.perf_counter()
    finally:
        runner.close()
    _emit("MATERIALIZE", f"{finished - materialize_started:.4f}")
    _emit("TIME", f"{finished - started:.6f}")


@contextmanager
def _capture_metrics():
    # Engine metrics come from a private hook that published wheels need not export
    try:
        from pycanopy.engine import _capture_engine_metrics  # noqa: PLC0415
    except ImportError:
        yield None
        return
    with _capture_engine_metrics() as capture:
        yield capture


def _run_profiled(query_id: str, data_dir: str, scale_factor: int) -> None:
    # Deferred because an ordinary run of another engine installs neither PyCanopy nor Polars
    from bench.spatial_bench.profiler_utils import (  # noqa: PLC0415
        StageProfiler,
        instrument_host_stages,
        profile_payload,
        verify_output,
    )
    from bench.spatial_bench.queries import pycanopy as queries  # noqa: PLC0415

    runner = load_runner("pycanopy")
    profiler = StageProfiler()
    try:
        runner.prepare(data_dir)
        module = queries.BY_ID[query_id]
        with _capture_metrics() as capture, instrument_host_stages(profiler, module):
            started = time.perf_counter()
            result = runner.execute(query_id)
            result = _materialize(result)
            elapsed = time.perf_counter() - started
            engine_metrics = capture.take_metrics() if capture is not None else []
    finally:
        runner.close()
        profiler.stop()

    _emit("TIME", f"{elapsed:.6f}")
    payload = profile_payload(profiler, elapsed, engine_metrics, capture is not None)
    _emit("PROFILE", json.dumps(payload))
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
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--scale-factor", type=int, choices=SUPPORTED_SCALE_FACTORS, default=1)
    args = parser.parse_args(argv)

    if args.profile and args.engine != "pycanopy":
        _emit("ERROR", f"profile mode measures pycanopy, not {args.engine}")
        return 1

    try:
        if args.profile:
            _run_profiled(args.query_id, args.data_dir, args.scale_factor)
        else:
            _run_timed(args.engine, args.query_id, args.data_dir)
    except Exception as exc:
        _emit("ERROR", f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
