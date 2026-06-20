"""Shared machinery for the SpatialBench suite: data, oracle, verify, measure, chart."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import sedonadb
import shapely

from bench.spatial_bench.sedona_sql import SEDONA_SQL
from pycanopy import SpatialFrame

matplotlib.use("Agg")  # headless backend, set before any figure is created

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


# Published Apache SpatialBench baseline (chart reference)


# Source: https://sedona.apache.org/spatialbench/single-node-benchmarks/
# m7i.2xlarge (8 vCPU, 32 GB), 1200 s timeout, SedonaDB 0.1.0 / DuckDB 1.4.0 / GeoPandas
# 1.1.1. A value is seconds, or "TIMEOUT" / "ERROR" (no bar, annotated on the chart).
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


def run_oracle(query_id: str, data_dir: str) -> pl.DataFrame:
    """Run query_id through SedonaDB and return its result as a polars DataFrame.

    The result comes back over Arrow (zero-copy into polars), avoiding a numpy or
    geopandas copy of what can be a tens-of-millions-row result.

    Args:
        query_id: Query id (e.g. "q1") indexing SEDONA_SQL.
        data_dir: Local directory or ``s3://`` URI of the parquet tables.

    Returns:
        The SedonaDB result as a Polars DataFrame.
    """
    sd = sedonadb.connect()
    for table in _ORACLE_TABLES:
        # SedonaDB reads the bare directory or file directly, with ST_GeomFromWKB in SQL
        sd.read_parquet(_resolve_table(data_dir, table)[0]).to_view(table)
    return pl.from_arrow(sd.sql(SEDONA_SQL[query_id]).to_arrow_table())


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
    pc_df, sedona_df, keys=(), values=(), rel_tol: float = 1e-2, abs_tol: float = 1e-2
) -> tuple[bool, str]:
    """Sanity-check a PyCanopy result against a SedonaDB result.

    Compares row count then each value column's float sum within tolerance. Both checks
    are order-independent and run in one streaming pass, so the result never leaves polars.

    Args:
        pc_df: PyCanopy result (polars LazyFrame, polars DataFrame, or pandas DataFrame).
        sedona_df: SedonaDB result in any of the same accepted forms.
        keys: Key columns from the compare spec (accepted but not used in the check).
        values: Column specs to sum-compare, each a name or (pc_col, sedona_col) pair.
        rel_tol: Relative tolerance on each column sum.
        abs_tol: Absolute tolerance on each column sum.

    Returns:
        A (passed, detail) tuple, where detail describes the match or the first mismatch.
    """
    pairs = _pairs(values)
    pc_h, pc_sums = _height_and_sums(pc_df, [a for a, _ in pairs])
    sed_h, sed_sums = _height_and_sums(sedona_df, [b for _, b in pairs])
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
            sed_df = run_oracle(query.id, data_dir)
            ok, detail = verify_outputs(pc_df, sed_df, **query.compare)
            out["match"] = "match" if ok else "MISMATCH"
            out["match_detail"] = detail
            if not ok:
                print(f"[verification] mismatch on testcase {query.id}: {detail}", flush=True)
        except Exception as exc:
            out["match"] = "skipped"
            out["match_detail"] = f"oracle error: {type(exc).__name__}: {exc}"
            print(f"[verification] skipped {query.id}: {type(exc).__name__}: {exc}", flush=True)
    return out


def write_chart(results: dict, out_path: Path) -> None:
    """Render a grouped bar chart: live PyCanopy vs published SedonaDB/DuckDB/GeoPandas.

    Bars carry their value in seconds, and a published TIMEOUT/ERROR draws no bar and is
    annotated. A query whose output did not match SedonaDB is flagged with a ``*`` label.

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
        "PyCanopy": "#4C72B0",
        "SedonaDB": "#DD8452",
        "DuckDB": "#55A868",
        "GeoPandas": "#C44E52",
    }
    series = ["PyCanopy", *PUBLISHED_ENGINES]

    def value(label: str, q: str):
        if label == "PyCanopy":
            return qs[q].get("pycanopy_seconds")
        return baseline.get(q, {}).get(label)

    fig, ax = plt.subplots(figsize=(max(10.0, 1.3 * len(qids)), 5.5))
    ax.set_axisbelow(True)  # gridlines sit behind the bars
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.5, alpha=0.35)
    bar_w = 0.8 / len(series)
    for li, label in enumerate(series):
        xs, heights = [], []
        for qi, q in enumerate(qids):
            v = value(label, q)
            x = qi + li * bar_w
            if isinstance(v, (int, float)):
                xs.append(x)
                heights.append(v)
            elif isinstance(v, str):  # TIMEOUT / ERROR
                ax.text(x, 1.0, v, rotation=90, ha="center", va="bottom", fontsize=6, color="grey")
        bars = ax.bar(xs, heights, width=bar_w, label=label, color=colors[label])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=6, rotation=90)

    has_mismatch = any(qs[q].get("match") == "MISMATCH" for q in qids)
    labels = [q + (" *" if qs[q].get("match") == "MISMATCH" else "") for q in qids]
    ax.set_xticks([i + bar_w * (len(series) - 1) / 2 for i in range(len(qids))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("seconds")
    ax.margins(x=0.01)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    subtitle = f"index mode: {mode}    PyCanopy measured, others from the published baseline"
    if has_mismatch:
        subtitle += "    * = output mismatch vs SedonaDB"
    ax.set_title(
        f"Apache SpatialBench SF{sf}: PyCanopy vs SedonaDB / DuckDB / GeoPandas\n{subtitle}",
        fontsize=11,
    )
    ax.legend(frameon=False, ncol=len(series), loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
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
