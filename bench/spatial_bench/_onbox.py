"""On-box entry point: measure PyCanopy on SpatialBench and render the chart.

Called by bootstrap.sh after building PyCanopy on the EC2 box:

    python -m bench.spatial_bench._onbox --scale-factor 1
"""

from __future__ import annotations

import argparse
import sys

from bench.spatial_bench import queries as query_registry
from bench.spatial_bench.utils import run_suite

_DATA_TEMPLATE = "s3://wherobots-examples/data/spatialbench/SpatialBench_sf{sf}"


def _build_parser() -> argparse.ArgumentParser:
    # CLI for the on-box benchmark runner
    parser = argparse.ArgumentParser(description="Measure PyCanopy on SpatialBench.")
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument(
        "--index-eager",
        action="store_const",
        const="eager",
        dest="index_mode",
    )
    parser.set_defaults(index_mode="auto")
    parser.add_argument("--verify", action="store_true")
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
