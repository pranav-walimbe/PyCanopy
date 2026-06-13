"""Run the SpatialBench query suite with PyCanopy and emit a results JSON.

This is the heavy, portable core of spatial_bench. It is intended to be invoked
verbatim by the AWS bootstrap script as well as locally:

    python -m bench.spatial_bench.measure --data-dir <local|s3://...> \
        --scale-factor 1 --output results/sf1.json

For each query it times the PyCanopy pipeline (cold, then warm), optionally runs
the GeoPandas reference for correctness checking, and records the outcome. The
JSON it writes is consumed by report.py to render comparison tables and charts.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from bench.spatial_bench import queries as query_registry
from bench.spatial_bench.data import SpatialBenchTables, table_path
from bench.spatial_bench.references import REFERENCE_META
from bench.utils.core import time_one


def _time_s(fn) -> tuple[float, object]:
    """Run fn once via the shared timing primitive. Returns (elapsed_seconds, result)."""
    elapsed_ms, result = time_one(fn)
    return elapsed_ms / 1000.0, result


def _data_paths(data_dir: str) -> dict[str, str]:
    """Map each table name to a parquet path/glob for the GeoPandas reference."""
    tables = ("trip", "zone", "building", "customer", "driver", "vehicle")
    return {t: table_path(data_dir, t) for t in tables}


def measure_query(query, data_dir: str, run_reference: bool, index_mode: str = "eager") -> dict:
    """Measure one query: PyCanopy cold/warm timings plus optional oracle check.

    Args:
        query: A query module exposing id, title, pycanopy(tables), reference(paths),
            and validate(pc_df, ref_df).
        data_dir: Local directory or s3:// URI of the parquet tables.
        run_reference: When True, run the GeoPandas reference and checksum outputs.
        index_mode: Index build policy ("eager" / "none" / "auto") for the frames the
            query builds. "none" gives brute-force timings (the --no-index comparison).

    Returns:
        Result dict for this query (timings, status, row_count, validation).
    """
    out: dict = {"title": query.title}
    # Fresh table handles per query so the cold run pays the real load + index cost.
    tables = SpatialBenchTables(data_dir=data_dir, scale_factor=0, index_mode=index_mode)

    try:
        cold_s, pc_df = _time_s(lambda: query.pycanopy(tables))
        warm_s, _ = _time_s(lambda: query.pycanopy(tables))
        out["pycanopy_seconds"] = round(cold_s, 4)
        out["pycanopy_warm_seconds"] = round(warm_s, 4)
        out["row_count"] = len(pc_df)
        out["status"] = "ok"
    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        return out

    if run_reference:
        try:
            paths = _data_paths(data_dir)
            ref_s, ref_df = _time_s(lambda: query.reference(paths))
            out["geopandas_seconds"] = round(ref_s, 4)
            ok, detail = query.validate(pc_df, ref_df)
            out["validation"] = "match" if ok else "MISMATCH"
            if not ok:
                out["validation_detail"] = detail
        except Exception as exc:
            out["validation"] = "reference_error"
            out["validation_detail"] = f"{type(exc).__name__}: {exc}"

    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SpatialBench queries with PyCanopy.")
    parser.add_argument(
        "--data-dir", required=True, help="Local directory or s3:// URI of parquet tables."
    )
    parser.add_argument(
        "--scale-factor", type=float, required=True, help="Scale factor (metadata + report key)."
    )
    parser.add_argument(
        "--output", default=None, help="Path to write the results JSON (default: stdout only)."
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Subset of query ids to run (e.g. q1 q4). Default: all.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip the GeoPandas reference run and correctness check.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--index-eager",
        action="store_const",
        const="eager",
        dest="index_mode",
        help="Build an index whenever a kind is selected (default).",
    )
    mode_group.add_argument(
        "--index-auto",
        action="store_const",
        const="auto",
        dest="index_mode",
        help="Build an index only when the cost model beats a brute-force scan.",
    )
    mode_group.add_argument(
        "--index-none",
        action="store_const",
        const="none",
        dest="index_mode",
        help="Never index; every query scans brute-force.",
    )
    parser.set_defaults(index_mode="eager")
    args = parser.parse_args(argv)

    index_mode = args.index_mode
    selected = query_registry.select(args.queries)
    results: dict = {
        "scale_factor": args.scale_factor,
        "data_dir": args.data_dir,
        "index_mode": index_mode,
        "reference_meta": REFERENCE_META,
        "queries": {},
    }

    for query in selected:
        tag = "" if index_mode == "eager" else f" [{index_mode}]"
        print(f"running {query.id}: {query.title}{tag} ...", flush=True)
        res = measure_query(
            query, args.data_dir, run_reference=not args.no_reference, index_mode=index_mode
        )
        results["queries"][query.id] = res
        status = res.get("status")
        val = res.get("validation", "")
        timing = res.get("pycanopy_seconds")
        print(f"  {query.id}: status={status} time={timing}s {val}", flush=True)

    payload = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload)
        print(f"\nwrote {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
