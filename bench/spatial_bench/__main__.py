"""Measure PyCanopy on the Apache SpatialBench query suite and render a comparison chart.

Run on the EC2 box after building PyCanopy in bootstrap.sh:

    python -m bench.spatial_bench --scale-factor 1
"""

from __future__ import annotations

import argparse
import sys

from bench.spatial_bench import queries as query_registry
from bench.spatial_bench.utils import run_suite

_DATA_TEMPLATE = "s3://wherobots-examples/data/spatialbench/SpatialBench_sf{sf}"


def _build_parser() -> argparse.ArgumentParser:
    # Build the CLI for the on-box benchmark runner
    parser = argparse.ArgumentParser(description="Measure PyCanopy on SpatialBench.")
    parser.add_argument(
        "--scale-factor",
        type=int,
        required=True,
        help="Scale factor (1 or 10), used for chart labels and the output filename.",
    )
    parser.add_argument(
        "--index-eager",
        action="store_const",
        const="eager",
        dest="index_mode",
        help="Build an index whenever a kind is selected (default is cost-based auto).",
    )
    parser.set_defaults(index_mode="auto")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the SedonaDB output check per query.",
    )
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
    run_suite(
        query_registry.ALL,
        data_dir,
        args.scale_factor,
        args.index_mode,
        verify=args.verify,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
