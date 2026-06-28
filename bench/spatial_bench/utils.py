"""Shared machinery for the SpatialBench suite: data, oracle, verify, measure, chart."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import sedonadb
import shapely
from matplotlib.patches import Patch

from bench.spatial_bench.sedona_sql import SEDONA_SQL
from pycanopy import SpatialFrame

matplotlib.use("Agg")  # headless backend, set before any figure is created

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


# Published Apache SpatialBench baseline (chart reference)


# Source: apache/sedona-spatialbench docs/single-node-benchmarks.md, m7i.2xlarge
# (8 vCPU, 32 GB), 1200 s timeout, cold start (DuckDB external file cache disabled),
# runtimes include full data loading. A value is seconds, "TIMEOUT", or "ERROR"
# (no bar rendered, status annotated on the chart).
PUBLISHED_ENGINES = ("SedonaDB", "DuckDB", "GeoPandas")

PUBLISHED: dict[int, dict[str, dict[str, float | str]]] = {
    1: {
        "q1": {"SedonaDB": 0.66, "DuckDB": 0.96, "GeoPandas": 12.78},
        "q2": {"SedonaDB": 8.07, "DuckDB": 9.95, "GeoPandas": 20.74},
        "q3": {"SedonaDB": 0.80, "DuckDB": 1.17, "GeoPandas": 13.59},
        "q4": {"SedonaDB": 8.41, "DuckDB": 9.83, "GeoPandas": 25.24},
        "q5": {"SedonaDB": 5.10, "DuckDB": 1.80, "GeoPandas": 47.08},
        "q6": {"SedonaDB": 8.59, "DuckDB": 9.36, "GeoPandas": 24.43},
        "q7": {"SedonaDB": 1.66, "DuckDB": 1.82, "GeoPandas": 137.00},
        "q8": {"SedonaDB": 1.10, "DuckDB": 1.08, "GeoPandas": 16.08},
        "q9": {"SedonaDB": 0.23, "DuckDB": 50.15, "GeoPandas": 0.28},
        "q10": {"SedonaDB": 18.79, "DuckDB": 207.84, "GeoPandas": 46.13},
        "q11": {"SedonaDB": 32.98, "DuckDB": "TIMEOUT", "GeoPandas": 51.01},
        "q12": {"SedonaDB": 14.55, "DuckDB": "ERROR", "GeoPandas": "TIMEOUT"},
    },
    10: {
        "q1": {"SedonaDB": 3.04, "DuckDB": 4.58, "GeoPandas": "ERROR"},
        "q2": {"SedonaDB": 8.89, "DuckDB": 8.26, "GeoPandas": "ERROR"},
        "q3": {"SedonaDB": 4.09, "DuckDB": 5.17, "GeoPandas": "TIMEOUT"},
        "q4": {"SedonaDB": 7.52, "DuckDB": 8.51, "GeoPandas": "ERROR"},
        "q5": {"SedonaDB": 50.81, "DuckDB": 14.40, "GeoPandas": "ERROR"},
        "q6": {"SedonaDB": 9.11, "DuckDB": 10.67, "GeoPandas": "ERROR"},
        "q7": {"SedonaDB": 14.44, "DuckDB": 14.03, "GeoPandas": "ERROR"},
        "q8": {"SedonaDB": 7.24, "DuckDB": 7.57, "GeoPandas": "TIMEOUT"},
        "q9": {"SedonaDB": 0.38, "DuckDB": 942.98, "GeoPandas": 0.49},
        "q10": {"SedonaDB": 42.02, "DuckDB": "ERROR", "GeoPandas": "ERROR"},
        "q11": {"SedonaDB": 97.52, "DuckDB": "ERROR", "GeoPandas": "ERROR"},
        "q12": {"SedonaDB": 145.66, "DuckDB": "ERROR", "GeoPandas": "TIMEOUT"},
    },
}


# Table loading


def read_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read one SpatialBench table as a Polars DataFrame (geometry stays WKB).

    Args:
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
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
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
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
    """Lazily-built, cached handles to the SpatialBench tables for one run.

    Args:
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
    """

    data_dir: str
    index_mode: str = "eager"
    _cache: dict[str, pl.DataFrame] | None = None

    def parallel_fetch(self, needs: dict[str, list[str] | None]) -> None:
        """Fetch several tables concurrently into the cache, with projection pushdown.

        Args:
            needs: Map of table name to the columns to fetch, or None for all columns.
        """
        if self._cache is None:
            self._cache = {}
        pending = {name: cols for name, cols in needs.items() if name not in self._cache}
        if not pending:
            return
        frames = pl.collect_all([self.scan(name, cols) for name, cols in pending.items()])
        self._cache.update(zip(pending, frames, strict=True))

    def table(self, name: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Return table ``name``, reading and caching it on first access.

        Args:
            name: Table name.
            columns: Optional subset of columns to read.

        Returns:
            The cached table as a Polars DataFrame.
        """
        if self._cache is None:
            self._cache = {}
        if name not in self._cache:
            self._cache[name] = read_table(self.data_dir, name, columns)
        df = self._cache[name]
        return df.select(columns) if columns is not None else df

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


# SedonaDB oracle (output verification only)


# Base tables the SedonaDB queries reference, registered as views over the parquet
_ORACLE_TABLES = ("trip", "zone", "building", "customer")


def oracle_result(query_id: str, data_dir: str) -> pl.DataFrame:
    """Run query_id through SedonaDB and return its full result.

    Args:
        query_id: Query id (e.g. "q1") indexing SEDONA_SQL.
        data_dir: ``s3://`` URI of the SpatialBench dataset root.

    Returns:
        The complete SedonaDB result as a Polars DataFrame.
    """
    sd = sedonadb.connect()
    for table in _ORACLE_TABLES:
        # Read via the working Polars S3 path and register the frame, so SedonaDB never reads S3
        sd.create_data_frame(read_table(data_dir, table)).to_view(table)
    return pl.from_arrow(sd.sql(SEDONA_SQL[query_id]).to_arrow_table())


# Output verification (PyCanopy result vs SedonaDB result)


def _pairs(spec) -> list[tuple[str, str]]:
    # Normalise each spec entry to a (pycanopy_col, sedona_col) pair
    return [(c, c) if isinstance(c, str) else tuple(c) for c in spec]


def _as_polars(df) -> pl.DataFrame:
    # Represent a polars or pandas result as polars, zero-copy where the buffers allow
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def verify_outputs(
    pc_df,
    query_id: str,
    data_dir: str,
    keys=(),
    values=(),
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-2,
) -> tuple[bool, str]:
    """Compare a full PyCanopy result against the full SedonaDB oracle result.

    Both results are sorted by their compared columns and checked row for row, so the two
    unordered outputs line up even when keys repeat (q12's k rows per trip).

    Args:
        pc_df: PyCanopy result (polars LazyFrame, polars DataFrame, or pandas DataFrame).
        query_id: Query id (e.g. "q1") indexing the SedonaDB oracle.
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
        keys: Key columns compared exactly (same name on both sides).
        values: Value column specs compared within tolerance, each a name or (pc_col, sedona_col) pair.
        rel_tol: Relative tolerance on each value column.
        abs_tol: Absolute tolerance on each value column.

    Returns:
        A (passed, detail) tuple, where detail describes the match or the first failing check.
    """
    pairs = _pairs(values)
    key_cols = list(keys)
    val_cols = [b for _, b in pairs]

    pc = pc_df.collect() if isinstance(pc_df, pl.LazyFrame) else _as_polars(pc_df)
    pc = pc.select(key_cols + [a for a, _ in pairs]).rename({a: b for a, b in pairs})
    oracle = oracle_result(query_id, data_dir).select(key_cols + val_cols)
    casts = [pl.col(c).cast(pl.Float64) for c in val_cols]
    pc = pc.with_columns(casts)
    oracle = oracle.with_columns(casts)

    if pc.height != oracle.height:
        return False, f"row count pycanopy={pc.height} sedona={oracle.height}"

    order = key_cols + val_cols
    pc = pc.sort(order, nulls_last=True)
    oracle = oracle.sort(order, nulls_last=True).rename({c: f"__o_{c}" for c in order})

    # Side by side, the two sorted frames must agree row for row. Position alignment holds
    # because both were sorted the same way, so any real difference falls out of step here.
    checks = []
    for c in key_cols:
        a, b = pl.col(c), pl.col(f"__o_{c}")
        checks.append((a == b).fill_null(False) | (a.is_null() & b.is_null()))
    for c in val_cols:
        a, b = pl.col(c), pl.col(f"__o_{c}")
        near = a.is_not_null() & b.is_not_null() & ((a - b).abs() <= abs_tol + rel_tol * b.abs())
        checks.append(near | (a.is_null() & b.is_null()))

    bad = pl.concat([pc, oracle], how="horizontal").filter(~pl.all_horizontal(checks))
    if bad.height:
        return False, f"{bad.height} row(s) differ, first: {bad.row(0, named=True)}"
    return True, f"{pc.height} rows match"


# Measure + chart


def spawn_query(query_id: str, data_dir: str, index_mode: str, *flags: str) -> dict:
    """Run one query in an isolated subprocess and parse its structured stdout.

    Args:
        query_id: Query id (e.g. "q1").
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
        flags: Extra flags forwarded to the runner (e.g. "--profile").

    Returns:
        A result dict: status "ok" carries time and the parsed kv lines, otherwise an error.
    """
    cmd = [sys.executable, "-m", "bench.spatial_bench._runner", query_id, data_dir, index_mode]
    cmd.extend(flags)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
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


def measure_query(query, data_dir: str, index_mode: str = "eager", runs: int = 3) -> dict:
    """Spawn isolated subprocesses for one query and return the averaged timing.

    Runs the query up to ``runs`` times in fresh subprocesses, stopping early if the
    first attempt fails or times out.

    Args:
        query: A query module exposing id, pycanopy(tables), and compare.
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
        runs: Number of timed repetitions to average (default 3).

    Returns:
        A result dict with status, pycanopy_seconds (average), and run_times.
    """
    times: list[float] = []
    run_kvs: list[dict] = []

    for i in range(runs):
        r = spawn_query(query.id, data_dir, index_mode)

        if r["status"] == "timeout":
            print(f"[testcase] timeout {query.id} (run {i + 1})", flush=True)
            if not times:
                return {"status": "timeout"}
            break

        if r["status"] == "error":
            print(f"[testcase] failed {query.id} (run {i + 1}): {r['error']}", flush=True)
            if not times:
                return {"status": "error", "error": r["error"]}
            break

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
    return {"status": "ok", "pycanopy_seconds": round(avg, 4), "run_times": times}


def _nice_cap(v: float) -> float:
    # Round a value up to a clean axis bound (1/1.5/2/2.5/3/4/5/6/8 times a power of ten)
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * mag >= v:
            return m * mag
    return 10 * mag


def _pct(values: list[float], p: float) -> float:
    # Linear-interpolated percentile, used to cap the x axis just past the bulk of the bars
    s = sorted(values)
    if not s:
        return 1.0
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def write_results_txt(results: dict, out_path: Path) -> None:
    """Write a plain-text results table alongside the chart PNG.

    Rows are sorted by query id. Each row shows the averaged PyCanopy time and, when
    more than one run was recorded, the individual run times in parentheses.

    Args:
        results: Measured results dict (scale_factor, index_mode, per-query timings).
        out_path: Destination text file path.
    """
    sf = int(results["scale_factor"])
    mode = results["index_mode"]
    qs = results["queries"]
    qids = sorted(qs, key=lambda q: int(q[1:]))

    lines = [f"SpatialBench SF{sf}  index mode: {mode}", ""]
    header = f"{'query':<8}  {'avg (s)':>10}  runs (s)"
    lines.append(header)
    lines.append("-" * len(header))

    for qid in qids:
        q = qs[qid]
        status = q.get("status", "error")
        if status == "timeout":
            lines.append(f"{qid:<8}  {'TIMEOUT':>10}")
        elif status != "ok" or q.get("pycanopy_seconds") is None:
            lines.append(f"{qid:<8}  {'ERROR':>10}")
        else:
            avg = q["pycanopy_seconds"]
            run_times = q.get("run_times", [])
            avg_str = f"{avg:.2f}"
            if len(run_times) > 1:
                runs_str = ", ".join(f"{t:.2f}" for t in run_times)
                lines.append(f"{qid:<8}  {avg_str:>10}  ({runs_str})")
            else:
                lines.append(f"{qid:<8}  {avg_str:>10}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_chart(results: dict, out_path: Path) -> None:
    """Render a horizontal grouped bar chart: live PyCanopy vs published SedonaDB/DuckDB/GeoPandas.

    Queries run down the y axis against a linear x axis capped just past the bulk, so outliers
    truncate to a value label, tiny bars print their value, and a TIMEOUT/ERROR annotates instead.

    Args:
        results: Measured results dict (scale_factor, index_mode, per-query timings).
        out_path: Destination PNG path.
    """
    sf = int(results["scale_factor"])
    mode = results["index_mode"]
    qs = results["queries"]
    qids = sorted(qs, key=lambda q: int(q[1:]))
    baseline = PUBLISHED.get(sf, {})

    colors = {
        "PyCanopy": "#2C7FB8",
        "SedonaDB": "#DD8452",
        "DuckDB": "#8C8C8C",
        "GeoPandas": "#C9BBA8",
    }
    series = ["PyCanopy", *PUBLISHED_ENGINES]
    n_s, n_q = len(series), len(qids)

    def value(label: str, q: str):
        if label == "PyCanopy":
            s = qs[q].get("pycanopy_seconds")
            if s is not None:
                return s
            return "ERROR" if qs[q].get("status") == "error" else None
        return baseline.get(q, {}).get(label)

    finite = [v for q in qids for s in series if isinstance(v := value(s, q), (int, float))]
    cap = _nice_cap(_pct(finite, 0.90)) if finite else 1.0
    truncated = any(v > cap for v in finite)

    fig, ax = plt.subplots(figsize=(8.2, 1.2 + 0.70 * n_q))
    ax.set_axisbelow(True)

    band = 0.82
    bar_h = band / n_s
    for qi in range(n_q):
        if qi % 2:  # tint alternating query rows
            ax.axhspan(qi - 0.5, qi + 0.5, color="#F4F7FA", zorder=0)
    for qi in range(1, n_q):
        ax.axhline(qi - 0.5, color="#DBDBDB", lw=0.6, ls=(0, (1, 2)), zorder=1)

    for si, label in enumerate(series):
        color = colors[label]
        for qi, q in enumerate(qids):
            y = qi + (si - (n_s - 1) / 2) * bar_h
            v = value(label, q)
            if isinstance(v, str):  # TIMEOUT / ERROR
                ax.text(
                    cap * 0.012,
                    y,
                    v.lower(),
                    ha="left",
                    va="center",
                    fontsize=6.5,
                    color="#3C7FA6",
                    fontstyle="italic",
                )
            elif v is not None:
                ax.barh(y, min(v, cap), height=bar_h * 0.9, color=color, zorder=2)
                if v > cap:
                    ax.text(
                        cap * 1.015,
                        y,
                        f"... {v:.1f}",
                        ha="left",
                        va="center",
                        fontsize=6.5,
                        color=color,
                    )
                elif v < cap * 0.03:
                    txt = f"{v:.2f}" if v < 1 else f"{v:.1f}"
                    ax.text(
                        v + cap * 0.008,
                        y,
                        txt,
                        ha="left",
                        va="center",
                        fontsize=6.5,
                        color="#555555",
                    )

    step = _nice_cap(cap / 6)
    ticks, t = [], 0.0
    while t <= cap + 1e-9:
        ticks.append(round(t, 6))
        t += step
    ax.set_xticks(ticks)
    ax.set_xlim(0, cap * 1.16)
    ax.set_ylim(-0.5, n_q - 0.5)
    ax.invert_yaxis()  # q1 at the top
    ax.set_yticks(range(n_q))
    ax.set_yticklabels([q + (" *" if qs[q].get("match") == "MISMATCH" else "") for q in qids])
    ax.set_xlabel("run time (seconds)")
    ax.grid(axis="x", which="major", color="#E6E6E6", lw=0.6, zorder=0)
    ax.tick_params(length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    subtitle = f"index mode: {mode}    PyCanopy measured, baselines from published SpatialBench"
    if truncated:
        subtitle += f"    bars past {cap:g}s truncated"
    if any(qs[q].get("match") == "MISMATCH" for q in qids):
        subtitle += "    * output mismatch"
    ax.set_title(
        f"Apache SpatialBench SF{sf}: PyCanopy vs SedonaDB / DuckDB / GeoPandas\n{subtitle}",
        fontsize=10,
    )
    ax.legend(
        handles=[Patch(facecolor=colors[s], label=s) for s in series],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_suite(
    query_modules: list,
    data_dir: str,
    scale_factor: float,
    index_mode: str = "eager",
    output: str | None = None,
    runs: int = 3,
) -> Path:
    """Measure each query module and render the comparison chart, returning its path.

    Loops measure_query over each module and writes the comparison chart for the
    given scale and index mode.

    Args:
        query_modules: Query modules to run, each exposing id, pycanopy, and compare.
        data_dir: ``s3://`` URI of the SpatialBench dataset root.
        scale_factor: Scale factor, used for the chart label and output filename.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
        output: Explicit PNG path, or None for assets/spatialbench_sf{N}[_mode].png.
        runs: Number of timed repetitions to average per query.

    Returns:
        The chart PNG path written.
    """
    results = {"scale_factor": scale_factor, "index_mode": index_mode, "queries": {}}
    for query in query_modules:
        results["queries"][query.id] = measure_query(query, data_dir, index_mode, runs=runs)
    sf = int(scale_factor)
    suffix = "" if index_mode == "eager" else f"_{index_mode}"
    out_path = Path(output) if output else _ASSETS_DIR / f"spatialbench_sf{sf}{suffix}.png"
    write_chart(results, out_path)
    txt_path = out_path.with_name(f"spatial-bench-sf{sf}{suffix}-results.txt")
    write_results_txt(results, txt_path)
    return out_path
