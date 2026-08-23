"""Registry of PyCanopy SpatialBench queries."""

from __future__ import annotations

from bench.spatial_bench.queries.pycanopy import (
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

ALL = [q01, q02, q03, q04, q05, q06, q07, q08, q09, q10, q11, q12]
BY_ID = {query.id: query for query in ALL}
