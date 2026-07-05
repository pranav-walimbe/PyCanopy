"""
Registry of SpatialBench query modules. Each module exposes id, title, a pycanopy(tables) implementation, and a compare dict for SedonaDB verification.
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
