"""Isolated subprocess runner for ordinary SpatialBench queries."""

from __future__ import annotations

import argparse
import sys
import time

from bench.spatial_bench.engines import ENGINE_IDS, load_runner


def main() -> None:
    """Prepare one engine, execute one query, and print its timing."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("engine", choices=ENGINE_IDS)
    parser.add_argument("query_id")
    parser.add_argument("data_dir")
    parser.add_argument("index_mode", choices=("auto", "eager", "none"))
    args = parser.parse_args()

    runner = load_runner(args.engine)
    try:
        runner.prepare(args.data_dir, args.index_mode)
        started = time.perf_counter()
        result = runner.execute(args.query_id)
        if hasattr(result, "collect"):
            result = result.collect()
        _ = len(result)
        elapsed = time.perf_counter() - started
    except Exception as exc:
        print(f"SPATIALBENCH_ERROR={type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
    finally:
        runner.close()

    print(f"SPATIALBENCH_TIME={elapsed:.6f}", flush=True)


if __name__ == "__main__":
    main()
