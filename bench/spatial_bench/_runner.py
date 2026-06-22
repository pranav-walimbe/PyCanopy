"""Subprocess entry point: runs one SpatialBench query in an isolated interpreter.

Structured output lines written to stdout:
    PYCANOPY_TIME=<seconds>          always on success
    PYCANOPY_MATCH=<detail>          verification passed
    PYCANOPY_MISMATCH=<detail>       verification failed
    PYCANOPY_VERIFY_ERROR=<detail>   oracle raised
    PYCANOPY_ERROR=<detail>          query raised
"""

from __future__ import annotations

import argparse
import sys
import time

from bench.spatial_bench import queries
from bench.spatial_bench.utils import SpatialBenchTables, verify_outputs


def main() -> None:
    """Parse args, run one query with timing, and print structured output to stdout."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("query_id")
    parser.add_argument("data_dir")
    parser.add_argument("index_mode")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    qmodule = queries._BY_ID.get(args.query_id)
    if qmodule is None:
        print(f"PYCANOPY_ERROR=unknown query {args.query_id!r}", flush=True)
        sys.exit(1)

    # Construction is free: no data is read until pycanopy() calls table() or scan()
    tables = SpatialBenchTables(data_dir=args.data_dir, index_mode=args.index_mode)

    try:
        t0 = time.perf_counter()
        result = qmodule.pycanopy(tables)
        if hasattr(result, "collect"):
            result = result.collect()
        _ = len(result)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        print(f"PYCANOPY_ERROR={type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)

    print(f"PYCANOPY_TIME={elapsed:.6f}", flush=True)

    if args.verify:
        try:
            ok, detail = verify_outputs(result, args.query_id, args.data_dir, **qmodule.compare)
            tag = "PYCANOPY_MATCH" if ok else "PYCANOPY_MISMATCH"
            print(f"{tag}={detail}", flush=True)
        except Exception as exc:
            print(f"PYCANOPY_VERIFY_ERROR={type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
