"""Shared machinery for the SpatialBench suite: data, oracle, verify, measure, chart.

The flow, run on the box via ``python -m bench.spatial_bench.utils``: for each query
time the PyCanopy pipeline and the same query run through SedonaDB (the live
verification and timing baseline), check that the two results agree, and render one
PNG of PyCanopy-vs-SedonaDB seconds per query with output mismatches flagged.

A query module (queries/qNN.py) only provides id, title, pycanopy(tables), and a
``compare`` spec; the SedonaDB SQL lives in sedona_sql.py; everything else lives here.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import shapely

from bench.spatial_bench.sedona_sql import SEDONA_SQL
from pycanopy import SpatialFrame

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
_SHAPELY_MULTIPOLYGON = 6


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


def wkb_to_polygons(series: pl.Series) -> list:
    """Convert a WKB polygon column to shapely Polygons, reducing MultiPolygons.

    MultiPolygons are reduced to their largest-area constituent Polygon so the result
    is a flat list of single Polygons suitable for Engine.from_polygons.
    """
    geoms = shapely.from_wkb(series.to_numpy())
    out = []
    for geom, tid in zip(geoms, shapely.get_type_id(geoms)):
        if tid == _SHAPELY_MULTIPOLYGON:
            out.append(max(geom.geoms, key=lambda p: p.area))
        else:
            out.append(geom)
    return out


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

    def point_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a point SpatialFrame from a WKB point column of ``df``."""
        return SpatialFrame.from_wkb_points(df, wkb_col, index_mode=self.index_mode)

    def polygon_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a polygon SpatialFrame, dropping the raw WKB column once decoded."""
        polys = wkb_to_polygons(df[wkb_col])
        enriched = df.drop(wkb_col).with_columns(pl.Series("_geom", polys, dtype=pl.Object))
        return SpatialFrame.from_polygons(
            enriched, geometry_col="_geom", index_mode=self.index_mode
        )


# SedonaDB oracle (verification + timing baseline)


# Base tables the SedonaDB queries reference, registered as views over the parquet.
_ORACLE_TABLES = ("trip", "zone", "building", "customer")


def run_oracle(query_id: str, data_dir: str):
    """Run query_id through SedonaDB and return its result as a pandas DataFrame.

    The result is materialised so it can be diffed against PyCanopy's output and so
    the call's wall clock reflects full query execution.
    """
    import sedonadb

    sd = sedonadb.connect()
    for table in _ORACLE_TABLES:
        # SedonaDB reads the bare directory (or file) directly; ST_GeomFromWKB in SQL.
        sd.read_parquet(_resolve_table(data_dir, table)[0]).to_view(table)
    return sd.sql(SEDONA_SQL[query_id]).to_pandas()


# Output verification (PyCanopy result vs SedonaDB result)


def _pairs(spec) -> list[tuple[str, str]]:
    """Normalise each spec entry to a (pycanopy_col, sedona_col) pair."""
    return [(c, c) if isinstance(c, str) else tuple(c) for c in spec]


def _rows(df, cols: list[str]) -> list[tuple]:
    """Materialise columns as row tuples; works for polars and pandas (empty -> none)."""
    return list(zip(*(df[c].to_list() for c in cols), strict=False))


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _sortable(x):
    """Map a cell to a totally ordered key so mixed-type rows sort deterministically."""
    if _missing(x):
        return (0, 0.0)
    if isinstance(x, (bool, int, float)):
        return (1, float(x))
    if hasattr(x, "timestamp"):  # datetime / pandas Timestamp
        return (2, x.timestamp())
    return (3, str(x))


def _close(a: tuple, b: tuple, rel_tol: float) -> bool:
    """True when two value tuples agree (exact for integers, rel_tol for floats)."""
    for x, y in zip(a, b, strict=False):
        if _missing(x) or _missing(y):
            if _missing(x) != _missing(y):
                return False
            continue
        x, y = float(x), float(y)
        if x.is_integer() and y.is_integer():
            if x != y:
                return False
        elif not math.isclose(x, y, rel_tol=rel_tol, abs_tol=1e-12):
            return False
    return True


def verify_outputs(pc_df, sedona_df, keys=(), values=(), rel_tol: float = 1e-6) -> tuple[bool, str]:
    """Check a PyCanopy result against a SedonaDB result by value, order-independent.

    Args:
        pc_df: PyCanopy result (polars DataFrame).
        sedona_df: SedonaDB result (pandas DataFrame).
        keys: Identifying columns, compared exactly. Each entry is a column name
            (same on both sides) or a (pycanopy_col, sedona_col) pair.
        values: Numeric columns compared within rel_tol, same entry form as keys.
        rel_tol: Relative tolerance for the value columns.

    Returns:
        (ok, detail).
    """
    keys = _pairs(keys)
    values = _pairs(values)
    pc = _rows(pc_df, [k[0] for k in keys] + [v[0] for v in values])
    sed = _rows(sedona_df, [k[1] for k in keys] + [v[1] for v in values])
    if len(pc) != len(sed):
        return False, f"row count pycanopy={len(pc)} sedona={len(sed)}"

    n = len(keys)
    pc.sort(key=lambda r: tuple(_sortable(x) for x in r))
    sed.sort(key=lambda r: tuple(_sortable(x) for x in r))
    for a, b in zip(pc, sed, strict=False):
        if tuple(_sortable(x) for x in a[:n]) != tuple(_sortable(x) for x in b[:n]):
            return False, f"key mismatch: {a[:n]} vs {b[:n]}"
        if not _close(a[n:], b[n:], rel_tol):
            return False, f"value mismatch at {a[:n]}: {a[n:]} vs {b[n:]}"
    return True, f"{len(pc)} rows match"


# Measure + chart


def measure_query(query, data_dir: str, index_mode: str = "eager", verify: bool = True) -> dict:
    """Time PyCanopy and SedonaDB for one query and check their outputs agree.

    Returns a result dict: status, pycanopy_seconds, sedonadb_seconds, match.
    """
    tables = SpatialBenchTables(data_dir=data_dir, index_mode=index_mode)
    try:
        t0 = time.perf_counter()
        pc_df = query.pycanopy(tables)
        pc_s = time.perf_counter() - t0
        print(f"[testcase] completed {query.id} using pycanopy in {pc_s:.2f}s", flush=True)
        t0 = time.perf_counter()
        sed_df = run_oracle(query.id, data_dir)
        sed_s = time.perf_counter() - t0
        print(f"[testcase] completed {query.id} using sedonadb in {sed_s:.2f}s", flush=True)
    except Exception as exc:
        print(f"[testcase] failed {query.id}: {type(exc).__name__}: {exc}", flush=True)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    out = {
        "status": "ok",
        "pycanopy_seconds": round(pc_s, 4),
        "sedonadb_seconds": round(sed_s, 4),
    }
    if verify:
        ok, detail = verify_outputs(pc_df, sed_df, **query.compare)
        out["match"] = "match" if ok else "MISMATCH"
        out["match_detail"] = detail
        if not ok:
            print(f"[verification] mismatch on testcase {query.id}: {detail}", flush=True)
    return out


def write_chart(results: dict, out_path: Path) -> None:
    """Render one grouped bar chart (PyCanopy vs SedonaDB) labelled with seconds.

    Bars carry their value in seconds; a query whose output did not match SedonaDB is
    flagged with a ``*`` on its x label. Queries that errored contribute no bars.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: render straight to file
    import matplotlib.pyplot as plt

    sf = int(results["scale_factor"])
    mode = results["index_mode"]
    qs = results["queries"]
    qids = sorted(qs, key=lambda q: int(q[1:]))
    series = [
        ("PyCanopy", "pycanopy_seconds", "#4C72B0"),
        ("SedonaDB", "sedonadb_seconds", "#DD8452"),
    ]

    fig, ax = plt.subplots(figsize=(max(9.0, 1.1 * len(qids)), 5.5))
    bar_w = 0.8 / len(series)
    for li, (label, key, color) in enumerate(series):
        xs = [qi + li * bar_w for qi, q in enumerate(qids) if qs[q].get(key)]
        heights = [qs[q][key] for q in qids if qs[q].get(key)]
        bars = ax.bar(xs, heights, width=bar_w, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    labels = [q + (" *" if qs[q].get("match") == "MISMATCH" else "") for q in qids]
    ax.set_xticks([i + bar_w / 2 for i in range(len(qids))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("seconds (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"SpatialBench SF{sf} ({mode}): PyCanopy vs SedonaDB   (* = output mismatch)")
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
