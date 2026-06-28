"""On-box entry point: measure PyCanopy on SpatialBench and render the chart.

Called by bootstrap.sh after building PyCanopy on the EC2 box.
"""

from __future__ import annotations

import argparse
import sys

from bench.spatial_bench import queries as query_registry
from bench.spatial_bench._profile import run_profile_suite
from bench.spatial_bench.utils import run_suite

_DATA_TEMPLATE = "s3://wherobots-examples/data/spatialbench/SpatialBench_sf{sf}"


def _build_parser() -> argparse.ArgumentParser:
    # CLI for the on-box benchmark runner
    parser = argparse.ArgumentParser(description="Measure PyCanopy on SpatialBench.")
    parser.add_argument("--scale-factor", type=int, required=True)
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument(
        "--index-eager", action="store_const", const="eager", dest="index_mode"
    )
    index_group.add_argument("--index-auto", action="store_const", const="auto", dest="index_mode")
    index_group.add_argument("--index-none", action="store_const", const="none", dest="index_mode")
    parser.set_defaults(index_mode="auto")
    parser.add_argument("--n", type=int, default=3, metavar="N")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--query", nargs="+", metavar="ID", help="Run only these query IDs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run SpatialBench on the current machine and return an exit code.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        The process exit code, 0 on success.
    """
    args = _build_parser().parse_args(argv)
    data_dir = _DATA_TEMPLATE.replace("{sf}", str(args.scale_factor))
    qs = query_registry.ALL
    if args.query:
        ids = set(args.query)
        qs = [q for q in qs if q.id in ids]
    if args.profile:
        run_profile_suite(qs, data_dir, args.index_mode)
    else:
        run_suite(qs, data_dir, args.scale_factor, args.index_mode, runs=args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
