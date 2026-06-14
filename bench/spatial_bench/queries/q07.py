"""Q7: Detect route detours by comparing reported vs straight-line trip distance.

This query is not spatial-index bound: the straight-line distance is the Euclidean
distance between pickup and dropoff, computed directly in Polars. Kept for full suite
coverage and to show where PyCanopy adds no index value.

Note: PyCanopy filters to trips with t_distance > 0; the canonical SedonaDB query
keeps all trips (using NULLIF on the divide), so the row counts differ by design.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pycanopy import wkb_points_to_xy

id = "q7"
title = "Route detour ratio (reported vs straight-line distance)"

DEG_PER_M = 0.000009  # 1 meter ~= 0.000009 degrees

compare = {
    "keys": ["t_tripkey"],
    "values": ["reported_distance_m", "line_distance_m", "detour_ratio"],
}


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", ["t_tripkey", "t_distance", "t_pickuploc", "t_dropoffloc"])
    px, py = wkb_points_to_xy(trip["t_pickuploc"])
    dx, dy = wkb_points_to_xy(trip["t_dropoffloc"])
    line_m = np.sqrt((px - dx) ** 2 + (py - dy) ** 2) / DEG_PER_M

    df = trip.select("t_tripkey", "t_distance").with_columns(
        pl.Series("line_distance_m", line_m),
        reported_distance_m=pl.col("t_distance").cast(pl.Float64),
    )
    df = df.filter(pl.col("t_distance") > 0)
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
