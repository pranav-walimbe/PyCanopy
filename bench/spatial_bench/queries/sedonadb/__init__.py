"""Pinned SedonaDB SpatialBench query registry."""

from bench.spatial_bench.queries.sedonadb import (
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

QUERIES = {
    "q1": q01.SQL,
    "q2": q02.SQL,
    "q3": q03.SQL,
    "q4": q04.SQL,
    "q5": q05.SQL,
    "q6": q06.SQL,
    "q7": q07.SQL,
    "q8": q08.SQL,
    "q9": q09.SQL,
    "q10": q10.SQL,
    "q11": q11.SQL,
    "q12": q12.SQL,
}
