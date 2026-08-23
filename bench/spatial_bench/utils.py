"""Shared data, measurement, and reporting machinery for SpatialBench."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl
import shapely

from bench.spatial_bench.config import (
    DATASET_VERSION,
    PUBLIC_DATA_ROOT,
    QUERY_TIMEOUT_SECONDS,
    WORKLOAD_REVISION,
)
from pycanopy import SpatialFrame

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


# Table loading


def read_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read one SpatialBench table as a Polars DataFrame (geometry stays WKB).

    Args:
        data_dir: Local or object-store SpatialBench dataset root.
        table: Table name (e.g. "trip").
        columns: Optional subset of columns to read.

    Returns:
        The table as a Polars DataFrame.
    """
    return pl.read_parquet(
        f"{data_dir.rstrip('/')}/{table}/**/*.parquet",
        columns=columns,
        storage_options={"skip_signature": "true"},
    )


def scan_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lazily scan one SpatialBench table as a LazyFrame (geometry stays WKB).

    Lazy sibling of read_table, for late materialization. A query that narrows rows on
    cheap columns never decodes the wide WKB column for the rows it later discards.

    Args:
        data_dir: Local or object-store SpatialBench dataset root.
        table: Table name (e.g. "trip").
        columns: Optional subset of columns to project.

    Returns:
        A LazyFrame over the table's parquet.
    """
    lf = pl.scan_parquet(
        f"{data_dir.rstrip('/')}/{table}/**/*.parquet",
        storage_options={"skip_signature": "true"},
    )
    return lf.select(columns) if columns is not None else lf


def wkb_to_polygons(series: pl.Series) -> list:
    """Decode a WKB polygon column to shapely Polygons / MultiPolygons.

    Args:
        series: A Polars Series of WKB-encoded polygon geometries.

    Returns:
        A list of shapely Polygon / MultiPolygon objects (each MultiPolygon kept whole).
    """
    return list(shapely.from_wkb(series.to_numpy()))


@dataclass
class SpatialBenchTables:
    """Projected SpatialBench table loader for one query run.

    Args:
        data_dir: Local or object-store SpatialBench dataset root.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
    """

    data_dir: str
    index_mode: str = "eager"

    def parallel_fetch(self, needs: dict[str, list[str] | None]) -> dict[str, pl.DataFrame]:
        """Fetch projected tables concurrently.

        Args:
            needs: Map of table name to the columns to fetch, or None for all columns.

        Returns:
            Query-scoped DataFrames keyed by table name.
        """
        names = list(needs)
        frames = self.collect_all([self.scan(name, needs[name]) for name in names])
        return dict(zip(names, frames, strict=True))

    def collect_all(self, frames: list[pl.LazyFrame]) -> list[pl.DataFrame]:
        """Collect lazy table plans concurrently."""
        return pl.collect_all(frames)

    def table(self, name: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Read one projected table into a query-scoped DataFrame.

        Args:
            name: Table name.
            columns: Optional subset of columns to read.

        Returns:
            The requested table as a Polars DataFrame.
        """
        return read_table(self.data_dir, name, columns)

    def scan(self, name: str, columns: list[str] | None = None) -> pl.LazyFrame:
        """Lazily scan table ``name`` (uncached, for late-materialization access).

        Returns a LazyFrame rather than a cached DataFrame because the point of a lazy
        scan is to defer reads until the collected plan decides what to read.

        Args:
            name: Table name.
            columns: Optional subset of columns to project.

        Returns:
            An uncached LazyFrame over the table.
        """
        return scan_table(self.data_dir, name, columns)

    def point_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a point SpatialFrame from a WKB point column of ``df``.

        Args:
            df: DataFrame holding the WKB point column.
            wkb_col: Name of the WKB point column.

        Returns:
            A point SpatialFrame over ``df``.
        """
        return SpatialFrame.from_wkb_points(df, wkb_col, index_mode=self.index_mode)

    def polygon_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a polygon SpatialFrame straight from the WKB column (decoded in Rust).

        Args:
            df: DataFrame holding the WKB polygon column.
            wkb_col: Name of the WKB polygon column.

        Returns:
            A polygon SpatialFrame over ``df``.
        """
        return SpatialFrame.from_wkb_polygons(df, wkb_col, index_mode=self.index_mode)


# Measure + chart


def _cpu_model() -> str:
    """Return a public hardware description without invoking platform-specific tools."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "not recorded"


def _memory_gib() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size / (1024**3)


def collect_run_metadata(
    data_dir: str,
    query_ids: list[str],
    scale_factor: int,
    index_mode: str,
    runs: int,
) -> dict[str, str]:
    """Collect public, durable metadata for one benchmark run."""
    try:
        engine_version = version("pycanopy")
    except PackageNotFoundError:
        engine_version = "development"

    memory = _memory_gib()
    metadata = {
        "timestamp (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run ID": os.environ.get("PYCANOPY_BENCH_RUN_ID", "local"),
        "workload revision": WORKLOAD_REVISION,
        "dataset": DATASET_VERSION,
        "source": "SpatialBench public S3" if data_dir.startswith(PUBLIC_DATA_ROOT) else "custom",
        "source region": os.environ.get(
            "PYCANOPY_BENCH_REGION", os.environ.get("AWS_DEFAULT_REGION", "not recorded")
        ),
        "engines": f"PyCanopy {engine_version}",
        "queries": ", ".join(query_ids),
        "configuration": f"SF{scale_factor}, index mode {index_mode}, {runs} run(s) per query",
        "system": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "CPU": _cpu_model(),
        "logical CPUs": str(os.cpu_count() or "not recorded"),
    }
    if memory is not None:
        metadata["memory"] = f"{memory:.1f} GiB"

    instance_type = os.environ.get("PYCANOPY_BENCH_INSTANCE_TYPE")
    ami = os.environ.get("PYCANOPY_BENCH_AMI_ID")
    if instance_type:
        metadata["cloud instance"] = instance_type
    if ami:
        metadata["AMI"] = ami

    volume_type = os.environ.get("PYCANOPY_BENCH_VOLUME_TYPE")
    if volume_type:
        metadata["storage"] = (
            f"{volume_type}, {os.environ.get('PYCANOPY_BENCH_VOLUME_GB', 'not recorded')} GiB, "
            f"{os.environ.get('PYCANOPY_BENCH_VOLUME_IOPS', 'not recorded')} IOPS, "
            f"{os.environ.get('PYCANOPY_BENCH_VOLUME_THROUGHPUT_MBPS', 'not recorded')} MiB/s"
        )
    return metadata


def spawn_query(
    query_id: str,
    data_dir: str,
    scale_factor: int,
    index_mode: str,
    *flags: str,
) -> dict:
    """Run one query in an isolated subprocess and parse its structured stdout.

    Args:
        query_id: Query id (e.g. "q1").
        data_dir: SpatialBench dataset root.
        scale_factor: Dataset scale factor used to select the committed answer.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
        flags: Extra flags forwarded to the runner (e.g. "--profile").

    Returns:
        A result dict: status "ok" carries time and the parsed kv lines, otherwise an error.
    """
    cmd = [
        sys.executable,
        "-m",
        "bench.spatial_bench.runner",
        query_id,
        data_dir,
        str(scale_factor),
        index_mode,
    ]
    cmd.extend(flags)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=QUERY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}

    kv: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("PYCANOPY_") and "=" in line:
            k, _, v = line.partition("=")
            kv[k] = v

    if "PYCANOPY_ERROR" in kv:
        return {"status": "error", "error": kv["PYCANOPY_ERROR"]}
    if "PYCANOPY_TIME" not in kv:
        snippet = proc.stderr[:400] if proc.stderr else "(no stderr)"
        return {"status": "error", "error": f"runner produced no timing output; stderr: {snippet}"}
    return {"status": "ok", "time": float(kv["PYCANOPY_TIME"]), "kv": kv}


def measure_query(
    query,
    data_dir: str,
    scale_factor: int,
    index_mode: str = "eager",
    runs: int = 3,
) -> dict:
    """Spawn isolated subprocesses for one query and return the averaged timing.

    Every requested run must complete. Correctness verification is reserved for
    profile mode and does not run in this timing path.

    Args:
        query: A query module exposing id and pycanopy(tables).
        data_dir: SpatialBench dataset root.
        scale_factor: Dataset scale factor passed to the isolated runner.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
        runs: Number of timed repetitions to average (default 3).

    Returns:
        A result dict with status, average time, and raw samples.
    """
    times: list[float] = []
    run_kvs: list[dict] = []

    for i in range(runs):
        r = spawn_query(query.id, data_dir, scale_factor, index_mode)

        if r["status"] == "timeout":
            print(f"[testcase] timeout {query.id} (run {i + 1})", flush=True)
            return {"status": "timeout", "run_times": times}

        if r["status"] == "error":
            print(f"[testcase] failed {query.id} (run {i + 1}): {r['error']}", flush=True)
            return {"status": "error", "error": r["error"], "run_times": times}

        times.append(r["time"])
        run_kvs.append(r.get("kv", {}))

    avg = sum(times) / len(times)
    print(
        f"[testcase] completed {query.id} using pycanopy in {avg:.2f}s"
        + (
            f" (avg of {len(times)} runs: {', '.join(f'{t:.2f}s' for t in times)})"
            if len(times) > 1
            else ""
        ),
        flush=True,
    )
    for i, (t, kv) in enumerate(zip(times, run_kvs), 1):
        mat = kv.get("PYCANOPY_MATERIALIZE", "")
        if mat:
            print(
                f"[timing] {query.id} run {i}: total={t:.2f}s,materialize={float(mat):.2f}s",
                flush=True,
            )
    return {
        "status": "ok",
        "pycanopy_seconds": round(avg, 4),
        "run_times": times,
    }


def _combined_results(results: dict) -> dict:
    metadata = dict(results["metadata"])
    engine_label = metadata.get("engines", "PyCanopy development")
    engine_version = engine_label.removeprefix("PyCanopy ")
    queries = {
        query_id: {
            "status": result.get("status", "error"),
            "seconds": result.get("pycanopy_seconds"),
            "run_times": result.get("run_times", []),
            "error": result.get("error", ""),
        }
        for query_id, result in results["queries"].items()
    }
    return {
        "scale_factor": results["scale_factor"],
        "index_mode": results["index_mode"],
        "engine_order": ["pycanopy"],
        "engines": {
            "pycanopy": {
                "id": "pycanopy",
                "version": engine_version,
                "metadata": metadata,
                "queries": queries,
            }
        },
        "metadata": metadata,
    }


def write_results_txt(results: dict, out_path: Path) -> None:
    """Write a PyCanopy result through the shared multi-engine reporter."""
    from bench.spatial_bench.report import write_results_txt as write_combined  # noqa: PLC0415

    write_combined(_combined_results(results), out_path)


def write_chart(results: dict, out_path: Path) -> None:
    """Render a PyCanopy result through the shared multi-engine reporter."""
    from bench.spatial_bench.report import write_chart as write_combined  # noqa: PLC0415

    write_combined(_combined_results(results), out_path)


def run_suite(
    query_modules: list,
    data_dir: str,
    scale_factor: int,
    index_mode: str = "eager",
    output: str | None = None,
    runs: int = 3,
) -> Path:
    """Measure the selected SpatialBench queries.

    Loops measure_query over each module and writes the comparison chart for the
    given scale and index mode.

    Args:
        query_modules: Query modules to run, each exposing id and pycanopy.
        data_dir: SpatialBench dataset root.
        scale_factor: Scale factor, used for the chart label and output filename.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
        output: Explicit PNG path, or None for assets/spatialbench_sf{N}[_mode].png.
        runs: Number of timed repetitions to average per query.

    Returns:
        The chart PNG path written after every query completes.

    Raises:
        RuntimeError: If any query fails or times out.
    """
    results = {
        "workload": "Apache SpatialBench geometry queries",
        "workload_revision": WORKLOAD_REVISION,
        "dataset_version": DATASET_VERSION,
        "scale_factor": scale_factor,
        "index_mode": index_mode,
        "runs_per_query": runs,
        "metadata": collect_run_metadata(
            data_dir,
            [query.id for query in query_modules],
            scale_factor,
            index_mode,
            runs,
        ),
        "queries": {},
    }
    for query in query_modules:
        results["queries"][query.id] = measure_query(
            query,
            data_dir,
            scale_factor,
            index_mode,
            runs=runs,
        )
    sf = int(scale_factor)
    suffix = "" if index_mode == "eager" else f"_{index_mode}"
    out_path = Path(output) if output else _ASSETS_DIR / f"spatialbench_sf{sf}{suffix}.png"
    write_chart(results, out_path)
    txt_path = out_path.with_name(f"spatial-bench-sf{sf}{suffix}-results.txt")
    write_results_txt(results, txt_path)
    invalid = [qid for qid, result in results["queries"].items() if result["status"] != "ok"]
    if invalid:
        raise RuntimeError(f"SpatialBench validation failed for: {', '.join(invalid)}")
    return out_path
