"""
Q5: Monthly travel spread of repeat customers (convex hull of dropoff points).
"""

from __future__ import annotations

import polars as pl

from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import Engine, wkb_points_to_xy

id = "q5"
title = "Monthly travel hull area for repeat customers"

MIN_TRIPS = 5


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    trip, cust = pl.collect_all(
        [
            pl.scan_parquet(data_paths["trip"], storage_options=STORAGE_OPTIONS).select(
                ["t_custkey", "t_dropoffloc", "t_pickuptime"]
            ),
            pl.scan_parquet(data_paths["customer"], storage_options=STORAGE_OPTIONS).select(
                ["c_custkey", "c_name"]
            ),
        ]
    )

    dx, dy = wkb_points_to_xy(trip["t_dropoffloc"])
    t = (
        trip.select(["t_custkey", "t_pickuptime"])
        .with_columns(
            pl.Series("dx", dx),
            pl.Series("dy", dy),
            pickup_month=pl.col("t_pickuptime").dt.truncate("1mo"),
        )
        .select(["t_custkey", "pickup_month", "dx", "dy"])
    )
    del trip

    grouped = (
        t.group_by(["t_custkey", "pickup_month"])
        .agg(trip_count=pl.len(), dxs=pl.col("dx"), dys=pl.col("dy"))
        .filter(pl.col("trip_count") > MIN_TRIPS)
    )

    areas = Engine.group_convex_hull_areas(grouped["dxs"], grouped["dys"])
    grouped = grouped.with_columns(
        monthly_travel_hull_area=pl.Series("monthly_travel_hull_area", areas, dtype=pl.Float64)
    )
    grouped = grouped.join(cust, left_on="t_custkey", right_on="c_custkey", how="inner")
    grouped = (
        grouped.lazy()
        .sort(
            ["monthly_travel_hull_area", "t_custkey", "pickup_month"],
            descending=[True, False, False],
        )
        .head(100)
        .collect()
    )

    return grouped.select(
        ["t_custkey", "c_name", "pickup_month", "monthly_travel_hull_area", "trip_count"]
    ).rename(
        {
            "t_custkey": "c_custkey",
            "c_name": "customer_name",
            "trip_count": "dropoff_count",
        }
    )
