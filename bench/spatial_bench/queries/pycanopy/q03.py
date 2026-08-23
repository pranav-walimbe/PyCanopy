"""Q3: Monthly trip stats within ~5km of a ~26.5 x 30km box around Sedona."""

from __future__ import annotations

import polars as pl
from shapely.geometry import Polygon

from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame

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


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    trip = pl.read_parquet(
        data_paths["trip"],
        columns=_COLS,
        storage_options=STORAGE_OPTIONS,
    )
    sf = SpatialFrame.from_wkb_points(trip, "t_pickuploc")
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
