"""
Q7: Detect route detours by comparing reported vs straight-line trip distance.
"""

from __future__ import annotations

import polars as pl

from pycanopy import wkb_point_distance

id = "q7"
title = "Route detour ratio (reported vs straight-line distance)"

DEG_PER_M = 0.000009  # 1 meter ~= 0.000009 degrees

compare = {
    "keys": ["t_tripkey"],
    "values": ["reported_distance_m", "line_distance_m", "detour_ratio"],
}


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", ["t_tripkey", "t_distance", "t_pickuploc", "t_dropoffloc"])
    line_m = wkb_point_distance(trip["t_pickuploc"], trip["t_dropoffloc"]) / DEG_PER_M

    df = trip.select("t_tripkey", "t_distance").with_columns(
        pl.Series("line_distance_m", line_m),
        reported_distance_m=pl.col("t_distance").cast(pl.Float64),
    )
    df = df.with_columns(
        detour_ratio=pl.when(pl.col("line_distance_m") != 0.0)
        .then(pl.col("reported_distance_m") / pl.col("line_distance_m"))
        .otherwise(None)
    )
    return df.select("t_tripkey", "reported_distance_m", "line_distance_m", "detour_ratio").sort(
        ["detour_ratio", "reported_distance_m", "t_tripkey"],
        descending=[True, True, False],
        nulls_last=True,
    )
