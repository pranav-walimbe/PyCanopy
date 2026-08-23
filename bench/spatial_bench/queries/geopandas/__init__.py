"""Pinned GeoPandas SpatialBench query registry."""

from bench.spatial_bench.queries.geopandas import (
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
    "q1": q01.q1,
    "q2": q02.q2,
    "q3": q03.q3,
    "q4": q04.q4,
    "q5": q05.q5,
    "q6": q06.q6,
    "q7": q07.q7,
    "q8": q08.q8,
    "q9": q09.q9,
    "q10": q10.q10,
    "q11": q11.q11,
    "q12": q12.q12,
}
