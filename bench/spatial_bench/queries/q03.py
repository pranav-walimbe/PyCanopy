"""Q3: Monthly trip stats within ~5km of a 10km bounding box around Sedona.

PyCanopy: filter trip points to those within 0.045 degrees of the bounding-box
polygon, then aggregate per pickup month.
"""

from __future__ import annotations

import polars as pl
from shapely.geometry import Polygon

id = "q3"
title = "Monthly trip stats within ~5km of a Sedona bounding box"

DISTANCE = 0.045  # degrees (~5km)
BASE_POLY = Polygon(
    [
        (-111.9060, 34.7347),
        (-111.6160, 34.7347),
        (-111.6160, 35.0047),
        (-111.9060, 35.0047),
        (-111.9060, 34.7347),
    ]
)

_COLS = ["t_pickuploc", "t_pickuptime", "t_dropofftime", "t_distance", "t_fare"]

# avg_duration is an interval in SedonaDB (Timedelta) vs float seconds here, so it is
# left out of the value check; the counts and other averages are compared.
compare = {"keys": ["pickup_month"], "values": ["total_trips", "avg_distance", "avg_fare"]}


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", _COLS)
    sf = tables.point_frame(trip, "t_pickuploc")
    filtered = sf.points_within_distance_of_polygon(BASE_POLY, DISTANCE)
    filtered = filtered.with_columns(
        pickup_month=pl.col("t_pickuptime").dt.truncate("1mo"),
        duration_seconds=(pl.col("t_dropofftime") - pl.col("t_pickuptime")).dt.total_seconds(),
    )
    return (
        filtered.group_by("pickup_month")
        .agg(
            total_trips=pl.len(),
            avg_distance=pl.col("t_distance").mean(),
            avg_duration=pl.col("duration_seconds").mean(),
            avg_fare=pl.col("t_fare").mean(),
        )
        .sort("pickup_month")
    )
