"""On-box driver: measure one engine over the SpatialBench queries and write its results."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from bench.spatial_bench.config import (
    ASSETS_DIR,
    DATASET_VERSION,
    DEFAULT_RUNS,
    ENGINE_IDS,
    ENGINES,
    PROFILE_VARIANTS,
    PUBLIC_DATA_ROOT,
    PUBLIC_DATA_TEMPLATE,
    QUERY_IDS,
    QUERY_TIMEOUT_SECONDS,
    REPOSITORY_BRANCH,
    RUNNER_PREFIX,
    SUPPORTED_SCALE_FACTORS,
    WORKLOAD_REVISION,
)
from bench.spatial_bench.report_utils import (
    write_profile,
    write_profile_transport,
    write_transport,
)


def _spawn(engine: str, query_id: str, data_dir: str, *flags: str) -> dict:
    # Run one query in an isolated interpreter and parse its structured stdout
    command = [
        sys.executable,
        "-m",
        "bench.spatial_bench.run_query",
        engine,
        query_id,
        data_dir,
        *flags,
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=QUERY_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}

    values: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if line.startswith(f"{RUNNER_PREFIX}_") and "=" in line:
            key, _, value = line.partition("=")
            values[key.removeprefix(f"{RUNNER_PREFIX}_")] = value

    if error := values.get("ERROR"):
        return {"status": "error", "error": error}
    if "TIME" not in values:
        detail = process.stderr[:400] or "runner produced no timing output"
        return {"status": "error", "error": detail}
    return {"status": "ok", "time": float(values["TIME"]), "values": values}


def _measure(engine: str, query_id: str, data_dir: str, runs: int) -> dict:
    # Average the requested number of timed runs, abandoning the query on the first failure
    samples: list[float] = []
    for attempt in range(1, runs + 1):
        result = _spawn(engine, query_id, data_dir)
        if result["status"] != "ok":
            detail = result.get("error", "")
            print(
                f"[testcase] {result['status']} {query_id} using {engine} "
                f"(run {attempt}){f': {detail}' if detail else ''}",
                flush=True,
            )
            return {**{k: v for k, v in result.items() if k != "values"}, "run_times": samples}
        samples.append(result["time"])
        if materialize := result["values"].get("MATERIALIZE"):
            print(
                f"[timing] {query_id} run {attempt}: total={result['time']:.2f}s,"
                f"materialize={float(materialize):.2f}s",
                flush=True,
            )

    average = sum(samples) / len(samples)
    print(f"[testcase] completed {query_id} using {engine} in {average:.2f}s", flush=True)
    return {"status": "ok", "seconds": average, "run_times": samples}


def _profile_query(query, data_dir: str) -> dict:
    # Run one profiled SF1 query and fold its verification verdict into the result
    result = _spawn("pycanopy", query.id, data_dir, "--profile")
    if result["status"] != "ok":
        print(f"[testcase] {result['status']} {query.id}: {result.get('error', '')}", flush=True)
        return {"status": result["status"], "title": query.title, "error": result.get("error", "")}

    values = result["values"]
    if "MATCH" in values:
        verify, detail = "match", values["MATCH"]
    elif "MISMATCH" in values:
        verify, detail = "MISMATCH", values["MISMATCH"]
    else:
        verify, detail = "error", values.get("VERIFY_ERROR", "no verification output")

    print(
        f"[testcase] completed {query.id} in {result['time']:.2f}s [verify: {verify}]", flush=True
    )
    if verify != "match":
        print(f"[verification] {verify} on {query.id}: {detail}", flush=True)
    return {
        "status": "ok",
        "title": query.title,
        "profile": json.loads(values["PROFILE"]),
        "verify": verify,
        "verify_detail": detail,
    }


def _cpu_model() -> str:
    # Public hardware description, avoiding platform-specific tools
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "not recorded"


def _git_revision() -> str:
    # Short commit the box is running, tying a profile back to its source
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _build_description(variant: str) -> str:
    # Name the PyCanopy under test well enough to tell two profile runs apart
    installed = version("pycanopy")
    if variant == "release":
        return f"pycanopy {installed} from PyPI"
    return f"pycanopy {installed} built from {REPOSITORY_BRANCH} at {_git_revision()}"


def _memory_gib() -> float | None:
    # Total physical memory, or None where the platform does not report it
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size / (1024**3)


def collect_metadata(engine: str, data_dir: str, scale_factor: int, runs: int) -> dict[str, str]:
    """Collect public, durable metadata describing where and how this run happened.

    Args:
        engine: Engine id being measured.
        data_dir: SpatialBench dataset root, recorded only as public or custom.
        scale_factor: Dataset scale factor.
        runs: Timed repetitions per query.

    Returns:
        Ordered metadata pairs for the run report.
    """
    values = {
        "timestamp (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run ID": os.environ.get("PYCANOPY_BENCH_RUN_ID", "local"),
        "workload revision": WORKLOAD_REVISION,
        "dataset": DATASET_VERSION,
        "source": "SpatialBench public S3" if data_dir.startswith(PUBLIC_DATA_ROOT) else "custom",
        "source region": os.environ.get(
            "PYCANOPY_BENCH_REGION", os.environ.get("AWS_DEFAULT_REGION", "not recorded")
        ),
        "configuration": (
            f"SF{scale_factor}, {runs} run(s) per query, {QUERY_TIMEOUT_SECONDS}s query timeout"
        ),
        "system": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "CPU": _cpu_model(),
        "logical CPUs": str(os.cpu_count() or "not recorded"),
    }
    if (memory := _memory_gib()) is not None:
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


def run_profile_suite(query_ids: list[str], data_dir: str, variant: str = "branch") -> Path:
    """Profile and oracle-verify every selected SF1 query exactly once for one build.

    Args:
        query_ids: Query ids to profile.
        data_dir: SpatialBench dataset root.
        variant: Build under test, either the repository branch or the published release.

    Returns:
        The path of the profile transport written under assets.
    """
    # Deferred because only a profile run installs PyCanopy and Polars on the box
    from bench.spatial_bench.queries import pycanopy as pycanopy_queries  # noqa: PLC0415

    results = {
        query_id: _profile_query(pycanopy_queries.BY_ID[query_id], data_dir)
        for query_id in query_ids
    }
    metadata = collect_metadata("pycanopy", data_dir, 1, 1)
    metadata["pycanopy build"] = _build_description(variant)
    write_profile(results, ASSETS_DIR / f"profile-{variant}.txt", metadata)
    transport_path = ASSETS_DIR / f"profile-{variant}.json"
    write_profile_transport(transport_path, variant, metadata, results)

    invalid = [
        query_id
        for query_id, result in results.items()
        if result["status"] != "ok" or result.get("verify") != "match"
    ]
    # The released build is a baseline, so only the branch build has to verify clean
    if invalid and variant == "branch":
        raise RuntimeError(f"profile verification failed for: {', '.join(invalid)}")
    if invalid:
        print(f"[verification] {variant} build did not verify: {', '.join(invalid)}", flush=True)
    return transport_path


def run_timing_suite(
    engine: str, query_ids: list[str], data_dir: str, scale_factor: int, runs: int
) -> Path:
    """Measure every selected query for one engine and write its transport file.

    Args:
        engine: Engine id to measure.
        query_ids: Query ids to run.
        data_dir: SpatialBench dataset root.
        scale_factor: Dataset scale factor.
        runs: Timed repetitions per query.

    Returns:
        The path of the transport file written under assets.
    """
    results = {query_id: _measure(engine, query_id, data_dir, runs) for query_id in query_ids}
    transport_path = ASSETS_DIR / f"{engine}-results.tsv"
    write_transport(
        transport_path,
        engine,
        version(ENGINES[engine]["package"]),
        collect_metadata(engine, data_dir, scale_factor, runs),
        results,
    )
    print(f"[testcase] wrote {ENGINES[engine]['display_name']} results", flush=True)
    return transport_path


def _build_parser() -> argparse.ArgumentParser:
    # CLI for the on-box suite driver
    parser = argparse.ArgumentParser(description="Measure one SpatialBench engine.")
    parser.add_argument("--engine", choices=ENGINE_IDS, default="pycanopy")
    parser.add_argument("--scale-factor", type=int, choices=SUPPORTED_SCALE_FACTORS, required=True)
    parser.add_argument(
        "--data-dir", help="Dataset root, defaulting to the published SpatialBench S3 source."
    )
    parser.add_argument("--n", type=int, default=DEFAULT_RUNS, metavar="N")
    parser.add_argument("--query", nargs="+", metavar="ID", help="Run only these query IDs.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--variant", choices=PROFILE_VARIANTS, default="branch")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the SpatialBench suite on the current machine and return an exit code.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        The process exit code, 0 on success.
    """
    args = _build_parser().parse_args(argv)
    if args.n < 1:
        raise SystemExit("--n must be at least 1")
    if args.profile:
        if args.scale_factor != 1:
            raise SystemExit("--profile runs the SF1 workload; pass --scale-factor 1")
        if args.engine != "pycanopy":
            raise SystemExit("--profile measures pycanopy; drop --engine or pass pycanopy")

    query_ids = list(QUERY_IDS)
    if args.query:
        if unknown := sorted(set(args.query) - set(QUERY_IDS)):
            raise SystemExit(f"unknown query IDs: {', '.join(unknown)}")
        query_ids = [query_id for query_id in query_ids if query_id in set(args.query)]

    data_dir = args.data_dir or PUBLIC_DATA_TEMPLATE.format(scale_factor=args.scale_factor)
    if args.profile:
        run_profile_suite(query_ids, data_dir, args.variant)
    else:
        run_timing_suite(args.engine, query_ids, data_dir, args.scale_factor, args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
