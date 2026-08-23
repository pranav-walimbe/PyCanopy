"""Measure one engine and write a transport file for the cloud controller."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from bench.spatial_bench.config import (
    DATASET_VERSION,
    DEFAULT_RUNS,
    DISPLAY_NAMES,
    ENGINE_IDS,
    PACKAGE_NAMES,
    PUBLIC_DATA_ROOT,
    QUERY_TIMEOUT_SECONDS,
    SUPPORTED_SCALE_FACTORS,
    WORKLOAD_REVISION,
)

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
_QUERY_IDS = tuple(f"q{i}" for i in range(1, 13))


def _spawn(engine: str, query_id: str, data_dir: str, index_mode: str) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "bench.spatial_bench.engine_runner",
        engine,
        query_id,
        data_dir,
        index_mode,
    ]
    prefix = "SPATIALBENCH"

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=QUERY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}

    values = {}
    for line in process.stdout.splitlines():
        if line.startswith(f"{prefix}_") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    error = values.get(f"{prefix}_ERROR")
    if error:
        return {"status": "error", "error": error}
    elapsed = values.get(f"{prefix}_TIME")
    if elapsed is None:
        detail = process.stderr[:400] or "runner produced no timing output"
        return {"status": "error", "error": detail}
    return {"status": "ok", "time": float(elapsed)}


def _measure(
    engine: str,
    query_id: str,
    data_dir: str,
    index_mode: str,
    runs: int,
) -> dict:
    samples = []
    for attempt in range(1, runs + 1):
        result = _spawn(engine, query_id, data_dir, index_mode)
        if result["status"] != "ok":
            print(
                f"[testcase] {result['status']} {query_id} using {engine} (run {attempt})",
                flush=True,
            )
            return {**result, "run_times": samples}
        samples.append(result["time"])

    average = sum(samples) / len(samples)
    print(
        f"[testcase] completed {query_id} using {engine} in {average:.2f}s",
        flush=True,
    )
    return {"status": "ok", "seconds": average, "run_times": samples}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "not recorded"


def _metadata(
    engine: str, data_dir: str, scale_factor: int, index_mode: str, runs: int
) -> dict[str, str]:
    memory = None
    try:
        memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    except (AttributeError, OSError, ValueError):
        pass
    values = {
        "timestamp (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workload revision": WORKLOAD_REVISION,
        "dataset": DATASET_VERSION,
        "source": "SpatialBench public S3" if data_dir.startswith(PUBLIC_DATA_ROOT) else "custom",
        "source region": os.environ.get("PYCANOPY_BENCH_REGION", "not recorded"),
        "configuration": (
            f"SF{scale_factor}, {runs} run(s) per query, {QUERY_TIMEOUT_SECONDS}s query timeout"
        ),
        "system": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "CPU": _cpu_model(),
        "logical CPUs": str(os.cpu_count() or "not recorded"),
    }
    if engine == "pycanopy":
        values["PyCanopy index mode"] = index_mode
    if memory is not None:
        values["memory"] = f"{memory:.1f} GiB"
    for label, variable in (
        ("cloud instance", "PYCANOPY_BENCH_INSTANCE_TYPE"),
        ("AMI", "PYCANOPY_BENCH_AMI_ID"),
    ):
        if value := os.environ.get(variable):
            values[label] = value
    if volume_type := os.environ.get("PYCANOPY_BENCH_VOLUME_TYPE"):
        values["storage"] = (
            f"{volume_type}, {os.environ.get('PYCANOPY_BENCH_VOLUME_GB', 'not recorded')} GiB, "
            f"{os.environ.get('PYCANOPY_BENCH_VOLUME_IOPS', 'not recorded')} IOPS, "
            f"{os.environ.get('PYCANOPY_BENCH_VOLUME_THROUGHPUT_MBPS', 'not recorded')} MiB/s"
        )
    return values


def _write_transport(
    path: Path,
    engine: str,
    engine_version: str,
    metadata: dict[str, str],
    results: dict[str, dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("engine", engine, engine_version))
        for key, value in metadata.items():
            writer.writerow(("metadata", key, value))
        for query_id, result in results.items():
            samples = ",".join(f"{sample:.6f}" for sample in result.get("run_times", []))
            writer.writerow(
                (
                    "query",
                    query_id,
                    result["status"],
                    f"{result['seconds']:.6f}" if result.get("seconds") is not None else "",
                    samples,
                    str(result.get("error", "")).replace("\n", " "),
                )
            )


def main(argv: list[str] | None = None) -> int:
    """Run selected queries for one engine."""
    parser = argparse.ArgumentParser(description="Measure one SpatialBench engine.")
    parser.add_argument("--engine", choices=ENGINE_IDS, required=True)
    parser.add_argument("--scale-factor", type=int, choices=SUPPORTED_SCALE_FACTORS, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--index-mode", choices=("auto", "eager", "none"), default="auto")
    parser.add_argument("--n", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--query", nargs="+", choices=_QUERY_IDS)
    args = parser.parse_args(argv)
    if args.n < 1:
        raise SystemExit("--n must be at least 1")

    query_ids = args.query or list(_QUERY_IDS)
    results = {
        query_id: _measure(
            args.engine,
            query_id,
            args.data_dir,
            args.index_mode,
            args.n,
        )
        for query_id in query_ids
    }
    engine_version = version(PACKAGE_NAMES[args.engine])
    _write_transport(
        _ASSETS_DIR / f"{args.engine}-results.tsv",
        args.engine,
        engine_version,
        _metadata(args.engine, args.data_dir, args.scale_factor, args.index_mode, args.n),
        results,
    )
    print(f"[testcase] wrote {DISPLAY_NAMES[args.engine]} results", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
