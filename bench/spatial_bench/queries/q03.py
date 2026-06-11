"""Q3: Monthly trip stats within ~5km of a 10km bounding box around Sedona.

PyCanopy: filter trip points to those within 0.045 degrees of the bounding-box
polygon, then aggregate per pickup month. The reference filters by distance to the
same polygon.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import polars as pl
from shapely.geometry import Polygon

from bench.spatial_bench import check

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

_COLS = ["t_tripkey", "t_pickuploc", "t_pickuptime", "t_dropofftime", "t_distance", "t_fare"]


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


def reference(paths) -> pd.DataFrame:
    trip_df = pd.read_parquet(paths["trip"], columns=_COLS)
    pickup = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    mask = pickup.distance(BASE_POLY) <= DISTANCE
    filtered = trip_df.loc[mask].copy()
    filtered["_duration_seconds"] = (
        filtered["t_dropofftime"] - filtered["t_pickuptime"]
    ).dt.total_seconds()
    filtered["pickup_month"] = filtered["t_pickuptime"].dt.to_period("M").dt.to_timestamp()
    return (
        filtered.groupby("pickup_month", as_index=False)
        .agg(
            total_trips=("t_tripkey", "count"),
            avg_distance=("t_distance", "mean"),
            avg_duration=("_duration_seconds", "mean"),
            avg_fare=("t_fare", "mean"),
        )
        .sort_values("pickup_month")
        .reset_index(drop=True)
    )


def validate(pc_df, ref_df) -> tuple[bool, str]:
    pc_map = {str(r["pickup_month"]): r["total_trips"] for r in pc_df.iter_rows(named=True)}
    ref_map = {
        str(m): c for m, c in zip(ref_df["pickup_month"], ref_df["total_trips"], strict=False)
    }
    return check.grouped(pc_map, ref_map)
