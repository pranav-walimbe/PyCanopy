"""Registry of SpatialBench query modules.

Each query module exposes:
  id: str                          query identifier, e.g. "q1"
  title: str                       short description
  pycanopy(tables) -> pl.DataFrame the PyCanopy + Polars implementation
  compare: dict                    keys/values to check against SedonaDB (utils.verify_outputs)

The SedonaDB oracle (utils.oracle_result) runs the same query and verify_outputs
compares the two full results row for row. Verification is SF1 only.
"""

from __future__ import annotations

from bench.spatial_bench.queries import (
    q01,
    q02,
    q03,
    q04,
    q05,
    q06,
    q07,
    q08,
    q09,
    q10,
    q11,
    q12,
)

# Ordered list of all implemented query modules
ALL = [q01, q02, q03, q04, q05, q06, q07, q08, q09, q10, q11, q12]

_BY_ID = {q.id: q for q in ALL}
