"""Shared machinery for the SpatialBench suite: data, oracle, verify, measure, chart."""

from __future__ import annotations

import math
import os
import time
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


# Source: apache/sedona-spatialbench Actions run #152 (2026-06-20)
# m7i.2xlarge (8 vCPU, 32 GB), 600 s timeout, SedonaDB 0.3.0 / DuckDB 1.5.4 /
# GeoPandas 1.1.3 / Spatial Polars 0.3.0. A value is seconds, "TIMEOUT", or "ERROR"
# (no bar rendered, status annotated on the chart). Missing entries (no key) render
# as no bar. DuckDB SF1 Q5-Q10 and GeoPandas SF1 Q5 were not reported in that run.
PUBLISHED_ENGINES = ("SedonaDB", "DuckDB", "GeoPandas", "Spatial Polars")

PUBLISHED: dict[int, dict[str, dict[str, float | str]]] = {
    1: {
        "q1": {"SedonaDB": 0.72, "DuckDB": 0.21, "GeoPandas": 14.26, "Spatial Polars": 3.60},
        "q2": {"SedonaDB": 0.93, "DuckDB": 0.33, "GeoPandas": 14.49, "Spatial Polars": 3.84},
        "q3": {"SedonaDB": 1.19, "DuckDB": 0.36, "GeoPandas": 15.16, "Spatial Polars": 2.67},
        "q4": {"SedonaDB": 0.99, "DuckDB": 0.47, "GeoPandas": 19.06, "Spatial Polars": 4.36},
        "q5": {"SedonaDB": 7.53, "Spatial Polars": 16.77},
        "q6": {"SedonaDB": 1.32, "GeoPandas": 20.03, "Spatial Polars": 8.07},
        "q7": {"SedonaDB": 3.97, "GeoPandas": 203.21, "Spatial Polars": 7.41},
        "q8": {"SedonaDB": 0.94, "GeoPandas": 16.61, "Spatial Polars": 6.41},
        "q9": {"SedonaDB": 0.39, "GeoPandas": 0.12, "Spatial Polars": 0.06},
        "q10": {"SedonaDB": 9.40, "GeoPandas": 41.45, "Spatial Polars": 19.17},
        "q11": {
            "SedonaDB": 13.36,
            "DuckDB": "TIMEOUT",
            "GeoPandas": 62.91,
            "Spatial Polars": 29.01,
        },
        "q12": {
            "SedonaDB": 42.09,
            "DuckDB": "TIMEOUT",
            "GeoPandas": "TIMEOUT",
            "Spatial Polars": 67.45,
        },
    },
    10: {
        "q1": {"SedonaDB": 3.06, "DuckDB": 2.23, "GeoPandas": "ERROR", "Spatial Polars": 32.22},
        "q2": {"SedonaDB": 3.84, "DuckDB": 3.24, "GeoPandas": "ERROR", "Spatial Polars": 31.21},
        "q3": {"SedonaDB": 5.75, "DuckDB": 3.18, "GeoPandas": "ERROR", "Spatial Polars": 36.51},
        "q4": {"SedonaDB": 1.78, "DuckDB": 1.20, "GeoPandas": "ERROR", "Spatial Polars": "ERROR"},
        "q5": {"SedonaDB": 99.43, "DuckDB": 357.99, "GeoPandas": "ERROR", "Spatial Polars": 128.31},
        "q6": {"SedonaDB": 4.93, "DuckDB": 4.62, "GeoPandas": "ERROR", "Spatial Polars": "ERROR"},
        "q7": {
            "SedonaDB": 37.92,
            "DuckDB": "ERROR",
            "GeoPandas": "TIMEOUT",
            "Spatial Polars": 82.56,
        },
        "q8": {"SedonaDB": 9.56, "DuckDB": 10.35, "GeoPandas": "ERROR", "Spatial Polars": "ERROR"},
        "q9": {"SedonaDB": 0.44, "DuckDB": 0.20, "GeoPandas": 0.30, "Spatial Polars": 0.34},
        "q10": {
            "SedonaDB": 71.60,
            "DuckDB": "TIMEOUT",
            "GeoPandas": "ERROR",
            "Spatial Polars": "ERROR",
        },
        "q11": {
            "SedonaDB": 103.21,
            "DuckDB": "ERROR",
            "GeoPandas": "ERROR",
            "Spatial Polars": "ERROR",
        },
        "q12": {
            "SedonaDB": "ERROR",
            "DuckDB": "TIMEOUT",
            "GeoPandas": "ERROR",
            "Spatial Polars": "ERROR",
        },
    },
}


# Table loading


def _resolve_table(data_dir: str, table: str) -> tuple[str, bool]:
    # Locate table under data_dir as (path, is_directory), handling the single-file and
    # directory-of-files layouts of the public datasets. s3:// URIs are directories.
    base = data_dir.rstrip("/")
    if base.startswith("s3://"):
        return f"{base}/{table}", True
    single = f"{base}/{table}.parquet"
    if os.path.exists(single):
        return single, False
    if os.path.isdir(f"{base}/{table}"):
        return f"{base}/{table}", True
    return single, False


def read_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read one SpatialBench table as a Polars DataFrame (geometry stays WKB).

    Args:
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        table: Table name (e.g. "trip").
        columns: Optional subset of columns to read.

    Returns:
        The table as a Polars DataFrame.
    """
    path, is_dir = _resolve_table(data_dir, table)
    return pl.read_parquet(f"{path}/**/*.parquet" if is_dir else path, columns=columns)


def scan_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lazily scan one SpatialBench table as a LazyFrame (geometry stays WKB).

    Lazy sibling of read_table, for late materialization. A query that narrows rows on
    cheap columns never decodes the wide WKB column for the rows it later discards.

    Args:
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        table: Table name (e.g. "trip").
        columns: Optional subset of columns to project.

    Returns:
        A LazyFrame over the table's parquet.
    """
    path, is_dir = _resolve_table(data_dir, table)
    lf = pl.scan_parquet(f"{path}/**/*.parquet" if is_dir else path)
    return lf.select(columns) if columns else lf


def warm_tables(data_dir: str, tables: tuple[str, ...]) -> None:
    """Read each table's raw parquet bytes into the OS page cache (untimed).

    Run before a measurement so the timed PyCanopy load reads from RAM, matching the
    resident-data condition of the published baseline. No-op for s3:// inputs.

    Args:
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        tables: Table names to warm.
    """
    if data_dir.rstrip("/").startswith("s3://"):
        return
    for table in tables:
        path, is_dir = _resolve_table(data_dir, table)
        files = Path(path).rglob("*.parquet") if is_dir else [Path(path)]
        for f in files:
            with open(f, "rb", buffering=0) as fh:
                while fh.read(1 << 20):
                    pass


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
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
    """

    data_dir: str
    index_mode: str = "eager"
    _cache: dict[str, pl.DataFrame] | None = None

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
        key = name if columns is None else f"{name}:{','.join(columns)}"
        if key not in self._cache:
            self._cache[key] = read_table(self.data_dir, name, columns)
        return self._cache[key]

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


def _strip_outer_order_by(sql: str) -> str:
    # Drop the final top-level ORDER BY so the aggregate wrapper never sorts the full
    # result. The scan skips string literals and -- comments and tracks paren depth, so an
    # ORDER BY inside a subquery (q4's top-tips LIMIT) or the words inside a comment
    # (q5's "ST_Collect_Agg (with _Agg suffix)") are left untouched.
    lowered = sql.lower()
    depth, i, n, cut = 0, 0, len(sql), None
    while i < n:
        c = sql[i]
        if c == "'":
            i += 1
            while i < n and sql[i] != "'":
                i += 1
        elif c == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                i += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and lowered.startswith("order by", i):
            cut = i
            i += 8
            continue
        i += 1
    return sql[:cut] if cut is not None else sql


def _aggregate_sql(inner: str, value_cols: list[str]) -> str:
    # Wrap a query so SedonaDB returns one row: COUNT(*) plus each value column's sum.
    # Pushing the reduction into the engine means the full result never materialises, and
    # since COUNT and SUM are order-independent the outer ORDER BY is stripped first.
    body = _strip_outer_order_by(inner)
    sums = "".join(f", SUM(CAST({c} AS DOUBLE)) AS {c}" for c in value_cols)
    return f"SELECT COUNT(*) AS __h__{sums}\nFROM (\n{body}\n) __agg__"


def oracle_summary(query_id: str, data_dir: str, value_cols: list[str]) -> tuple[int, dict]:
    """Reduce query_id through SedonaDB to (row count, per-column float sums).

    The COUNT and SUMs are pushed into the SedonaDB SQL, so the oracle returns a single
    row, the full result never leaves the engine, and verification adds no per-query
    memory load. The check needs only the count and sums, both of which are order-independent.

    Args:
        query_id: Query id (e.g. "q1") indexing SEDONA_SQL.
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        value_cols: SedonaDB result columns to sum (empty checks the row count only).

    Returns:
        A (row count, {column: sum}) tuple summarizing the SedonaDB result.
    """
    sd = sedonadb.connect()
    for table in _ORACLE_TABLES:
        # SedonaDB reads the bare directory or file directly, with ST_GeomFromWKB in SQL
        sd.read_parquet(_resolve_table(data_dir, table)[0]).to_view(table)
    sql = _aggregate_sql(SEDONA_SQL[query_id], value_cols)
    row = pl.from_arrow(sd.sql(sql).to_arrow_table())
    return int(row["__h__"][0]), {c: row[c][0] for c in value_cols}


# Output verification (PyCanopy result vs SedonaDB result)


def _pairs(spec) -> list[tuple[str, str]]:
    # Normalise each spec entry to a (pycanopy_col, sedona_col) pair
    return [(c, c) if isinstance(c, str) else tuple(c) for c in spec]


def _as_polars(df) -> pl.DataFrame:
    # Represent a polars or pandas result as polars, zero-copy where the buffers allow
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def _height_and_sums(df, value_cols) -> tuple[int, dict[str, float]]:
    # Return (row count, Float64 column sums) in one bounded streaming pass, so a result
    # larger than RAM (a LazyFrame over an out-of-core sink) never materialises in memory.
    exprs = [pl.len().alias("__h__")]
    exprs += [pl.col(c).cast(pl.Float64, strict=False).sum().alias(c) for c in value_cols]
    lf = df if isinstance(df, pl.LazyFrame) else _as_polars(df).lazy()
    row = lf.select(exprs).collect(engine="streaming")
    return row["__h__"][0], {c: row[c][0] for c in value_cols}


def verify_outputs(
    pc_df,
    query_id: str,
    data_dir: str,
    keys=(),
    values=(),
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-2,
) -> tuple[bool, str]:
    """Sanity-check a PyCanopy result against the SedonaDB oracle.

    Compares row count then each value column's float sum within tolerance. The oracle
    reduces to the count and sums in SQL (one row back), and the PyCanopy side streams the
    same aggregates, so neither full result ever materialises. Both checks are order-independent.

    Args:
        pc_df: PyCanopy result (polars LazyFrame, polars DataFrame, or pandas DataFrame).
        query_id: Query id (e.g. "q1") indexing the SedonaDB oracle.
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        keys: Key columns from the compare spec (accepted but not used in the check).
        values: Column specs to sum-compare, each a name or (pc_col, sedona_col) pair.
        rel_tol: Relative tolerance on each column sum.
        abs_tol: Absolute tolerance on each column sum.

    Returns:
        A (passed, detail) tuple, where detail describes the match or the first mismatch.
    """
    pairs = _pairs(values)
    pc_h, pc_sums = _height_and_sums(pc_df, [a for a, _ in pairs])
    sed_h, sed_sums = oracle_summary(query_id, data_dir, [b for _, b in pairs])
    if pc_h != sed_h:
        return False, f"row count pycanopy={pc_h} sedona={sed_h}"

    for pc_col, sed_col in pairs:
        a, b = pc_sums[pc_col], sed_sums[sed_col]
        if abs(a - b) > abs_tol + rel_tol * abs(b):
            return False, f"sum mismatch in {pc_col}: {a} vs {b}"
    return True, f"{pc_h} rows match"


# Measure + chart


def measure_query(query, data_dir: str, index_mode: str = "eager", verify: bool = True) -> dict:
    """Time PyCanopy on one query, then verify its output against SedonaDB.

    The dataset is warmed into the page cache (untimed) and decoded fresh inside the timed
    region, so the load is included but always reads from RAM. SedonaDB only checks output.

    Args:
        query: A query module exposing id, pycanopy(tables), and compare.
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
        verify: Run the SedonaDB output check when True.

    Returns:
        A result dict with status, pycanopy_seconds, and (when verified) match fields.
    """
    warm_tables(data_dir, _ORACLE_TABLES)
    tables = SpatialBenchTables(data_dir=data_dir, index_mode=index_mode)
    try:
        t0 = time.perf_counter()
        pc_df = query.pycanopy(tables)
        pc_s = time.perf_counter() - t0
        print(f"[testcase] completed {query.id} using pycanopy in {pc_s:.2f}s", flush=True)
    except Exception as exc:
        print(f"[testcase] failed {query.id}: {type(exc).__name__}: {exc}", flush=True)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    out = {"status": "ok", "pycanopy_seconds": round(pc_s, 4)}
    if verify:
        try:
            ok, detail = verify_outputs(pc_df, query.id, data_dir, **query.compare)
            out["match"] = "match" if ok else "MISMATCH"
            out["match_detail"] = detail
            if not ok:
                print(f"[verification] mismatch on testcase {query.id}: {detail}", flush=True)
        except Exception as exc:
            out["match"] = "skipped"
            out["match_detail"] = f"oracle error: {type(exc).__name__}: {exc}"
            print(f"[verification] skipped {query.id}: {type(exc).__name__}: {exc}", flush=True)
    return out


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
        "Spatial Polars": "#59A96A",
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
        f"Apache SpatialBench SF{sf}: PyCanopy vs SedonaDB / DuckDB / GeoPandas / Spatial Polars\n{subtitle}",
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
    verify: bool = True,
) -> Path:
    """Measure each query module and render the comparison chart, returning its path.

    This is the on-box driver bootstrap.sh re-enters through the package. It loops
    measure_query over the modules then writes the chart for the scale and index mode.

    Args:
        query_modules: Query modules to run, each exposing id, pycanopy, and compare.
        data_dir: Local directory or ``s3://`` URI of the parquet tables.
        scale_factor: Scale factor, used for the chart label and output filename.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
        output: Explicit PNG path, or None for assets/spatialbench_sf{N}[_mode].png.
        verify: Run the SedonaDB output check per query when True.

    Returns:
        The chart PNG path written.
    """
    results = {"scale_factor": scale_factor, "index_mode": index_mode, "queries": {}}
    for query in query_modules:
        results["queries"][query.id] = measure_query(query, data_dir, index_mode, verify=verify)
    sf = int(scale_factor)
    suffix = "" if index_mode == "eager" else f"_{index_mode}"
    out_path = Path(output) if output else _ASSETS_DIR / f"spatialbench_sf{sf}{suffix}.png"
    write_chart(results, out_path)
    return out_path
