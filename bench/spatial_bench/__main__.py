"""Entry point: run SpatialBench (PyCanopy vs SedonaDB) and render the comparison chart."""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.spatial_bench import queries as query_registry
from bench.spatial_bench.utils import measure_query, write_chart

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


def main(argv: list[str] | None = None) -> None:
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
        help="Never index, every query scans brute-force.",
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
