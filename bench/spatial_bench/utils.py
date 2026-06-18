"""Shared machinery for the SpatialBench suite: data, oracle, verify, measure, chart.

The flow, run on the box via ``python -m bench.spatial_bench.utils``: for each query
warm the dataset into the page cache, time the PyCanopy pipeline, then run the same
query through SedonaDB only to check the two results agree, and render one PNG of
PyCanopy (measured) vs the published SedonaDB/DuckDB/GeoPandas numbers per query with
output mismatches flagged.

A query module (queries/qNN.py) only provides id, title, pycanopy(tables), and a
``compare`` spec; the SedonaDB SQL lives in sedona_sql.py; everything else lives here.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import sedonadb
import shapely

from bench.spatial_bench.sedona_sql import SEDONA_SQL
from pycanopy import SpatialFrame

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
    """Locate ``table`` under ``data_dir``, returning (path, is_directory).

    Handles the single-file layout (``trip.parquet``) and the directory-of-files
    layout (``trip/``) of the public S3 datasets. ``s3://`` URIs are assumed to be
    directories. Shared by the polars (glob) and SedonaDB (directory) readers.
    """
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
    """Read one SpatialBench table as a Polars DataFrame (geometry stays WKB)."""
    path, is_dir = _resolve_table(data_dir, table)
    return pl.read_parquet(f"{path}/**/*.parquet" if is_dir else path, columns=columns)


def scan_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lazily scan one SpatialBench table as a LazyFrame (geometry stays WKB).

    Lazy sibling of read_table: nothing is read until collected, so Polars can push
    projection and row limits below the column reads. Use this for late materialization,
    where a query narrows rows on cheap columns before it needs geometry and so never
    decodes the wide WKB column for discarded rows. When every row's geometry is needed
    (the common case) read_table is the right tool.
    """
    path, is_dir = _resolve_table(data_dir, table)
    lf = pl.scan_parquet(f"{path}/**/*.parquet" if is_dir else path)
    return lf.select(columns) if columns else lf


def warm_tables(data_dir: str, tables: tuple[str, ...]) -> None:
    """Read each table's raw parquet bytes into the OS page cache (untimed).

    Run before a measurement so the timed PyCanopy load reads from RAM, matching the
    resident-data condition of the published baseline regardless of prior eviction.
    No-op for s3:// inputs, which cannot be warmed locally.
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

    MultiPolygons are kept whole: the engine treats each as one logical polygon spanning
    all its parts, so a point in any part matches the zone (as ST_Within does).
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
        """Return table ``name``, reading and caching it on first access."""
        if self._cache is None:
            self._cache = {}
        key = name if columns is None else f"{name}:{','.join(columns)}"
        if key not in self._cache:
            self._cache[key] = read_table(self.data_dir, name, columns)
        return self._cache[key]

    def scan(self, name: str, columns: list[str] | None = None) -> pl.LazyFrame:
        """Lazily scan table ``name`` (uncached; for late-materialization access).

        Returns a LazyFrame rather than a cached DataFrame because the point of a
        lazy scan is to defer reads until the collected plan decides what to read.
        """
        return scan_table(self.data_dir, name, columns)

    def point_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a point SpatialFrame from a WKB point column of ``df``."""
        return SpatialFrame.from_wkb_points(df, wkb_col, index_mode=self.index_mode)

    def polygon_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a polygon SpatialFrame straight from the WKB column (decoded in Rust)."""
        return SpatialFrame.from_wkb_polygons(df, wkb_col, index_mode=self.index_mode)


# SedonaDB oracle (output verification only)


# Base tables the SedonaDB queries reference, registered as views over the parquet.
_ORACLE_TABLES = ("trip", "zone", "building", "customer")


def run_oracle(query_id: str, data_dir: str) -> pl.DataFrame:
    """Run query_id through SedonaDB and return its result as a polars DataFrame.

    The result comes back over Arrow (zero-copy into polars), avoiding a numpy or
    geopandas copy of what can be a tens-of-millions-row result.
    """
    sd = sedonadb.connect()
    for table in _ORACLE_TABLES:
        # SedonaDB reads the bare directory (or file) directly; ST_GeomFromWKB in SQL.
        sd.read_parquet(_resolve_table(data_dir, table)[0]).to_view(table)
    return pl.from_arrow(sd.sql(SEDONA_SQL[query_id]).to_arrow_table())


# Output verification (PyCanopy result vs SedonaDB result)


def _pairs(spec) -> list[tuple[str, str]]:
    """Normalise each spec entry to a (pycanopy_col, sedona_col) pair."""
    return [(c, c) if isinstance(c, str) else tuple(c) for c in spec]


def _as_polars(df) -> pl.DataFrame:
    """Represent a polars or pandas result as polars, zero-copy where the buffers allow."""
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def verify_outputs(
    pc_df, sedona_df, keys=(), values=(), rel_tol: float = 1e-2, abs_tol: float = 1e-2
) -> tuple[bool, str]:
    """Sanity-check a PyCanopy result against a SedonaDB result.

    Compares row count, then the float sum of each value column within tolerance. Both
    checks are order-independent and stay columnar, so a tens-of-millions-row result
    never leaves polars. Summing in Float64 avoids integer overflow.
    """
    pc = _as_polars(pc_df)
    sed = _as_polars(sedona_df)
    if pc.height != sed.height:
        return False, f"row count pycanopy={pc.height} sedona={sed.height}"

    for pc_col, sed_col in _pairs(values):
        a = pc[pc_col].cast(pl.Float64, strict=False).sum()
        b = sed[sed_col].cast(pl.Float64, strict=False).sum()
        if abs(a - b) > abs_tol + rel_tol * abs(b):
            return False, f"sum mismatch in {pc_col}: {a} vs {b}"
    return True, f"{pc.height} rows match"


# Measure + chart


def measure_query(query, data_dir: str, index_mode: str = "eager", verify: bool = True) -> dict:
    """Time PyCanopy on one query, then verify its output against SedonaDB.

    The dataset is warmed into the page cache (untimed) and decoded fresh inside the
    timed region, so the load is included but always reads from RAM. SedonaDB is run
    only to check the result, not for timing (the chart baseline is the published
    numbers). Returns: status, pycanopy_seconds, match.
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

    Bars carry their value in seconds. A published TIMEOUT/ERROR draws no bar and is
    annotated. A query whose output did not match SedonaDB is flagged with ``*`` on its
    x label.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: render straight to file
    import matplotlib.pyplot as plt

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

    labels = [q + (" *" if qs[q].get("match") == "MISMATCH" else "") for q in qids]
    ax.set_xticks([i + bar_w * (len(series) - 1) / 2 for i in range(len(qids))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("seconds (log scale)")
    ax.set_yscale("log")
    ax.set_title(
        f"SpatialBench SF{sf} ({mode}): PyCanopy measured vs published "
        f"SedonaDB/DuckDB/GeoPandas   (* = output mismatch)"
    )
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    from bench.spatial_bench import queries as query_registry

    parser = argparse.ArgumentParser(description="Run SpatialBench (PyCanopy vs SedonaDB).")
    parser.add_argument("--data-dir", required=True, help="Local directory or s3:// URI of tables.")
    parser.add_argument(
        "--scale-factor", type=float, required=True, help="Scale factor (chart label)."
    )
    parser.add_argument("--output", default=None, help="PNG path (default: assets/spatialbench_*).")
    parser.add_argument(
        "--queries", nargs="*", default=None, help="Subset of query ids (e.g. q1 q4)."
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip the SedonaDB output check.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--index-eager",
        action="store_const",
        const="eager",
        dest="index_mode",
        help="Build an index whenever a kind is selected (default).",
    )
    mode.add_argument(
        "--index-auto",
        action="store_const",
        const="auto",
        dest="index_mode",
        help="Build an index only when the cost model beats a brute-force scan.",
    )
    mode.add_argument(
        "--index-none",
        action="store_const",
        const="none",
        dest="index_mode",
        help="Never index; every query scans brute-force.",
    )
    parser.set_defaults(index_mode="eager")
    args = parser.parse_args(argv)

    results = {"scale_factor": args.scale_factor, "index_mode": args.index_mode, "queries": {}}
    for query in query_registry.select(args.queries):
        results["queries"][query.id] = measure_query(
            query, args.data_dir, args.index_mode, verify=not args.no_verify
        )

    sf = int(args.scale_factor)
    suffix = "" if args.index_mode == "eager" else f"_{args.index_mode}"
    out_path = (
        Path(args.output) if args.output else _ASSETS_DIR / f"spatialbench_sf{sf}{suffix}.png"
    )
    write_chart(results, out_path)


if __name__ == "__main__":
    main()
