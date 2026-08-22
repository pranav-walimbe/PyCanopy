"""
Q5: Monthly travel spread of repeat customers (convex hull of dropoff points).
"""

from __future__ import annotations

import polars as pl

from pycanopy import Engine, wkb_points_to_xy

id = "q5"
title = "Monthly travel hull area for repeat customers"

MIN_TRIPS = 5

TABLES_NEEDED = {
    "trip": ["t_custkey", "t_dropoffloc", "t_pickuptime"],
    "customer": ["c_custkey", "c_name"],
}


def pycanopy(tables) -> pl.DataFrame:
    tables.parallel_fetch(TABLES_NEEDED)
    trip = tables.table("trip", ["t_custkey", "t_dropoffloc", "t_pickuptime"])
    cust = tables.table("customer", ["c_custkey", "c_name"])

    dx, dy = wkb_points_to_xy(trip["t_dropoffloc"])
    t = trip.with_columns(
        pl.Series("dx", dx),
        pl.Series("dy", dy),
        pickup_month=pl.col("t_pickuptime").dt.truncate("1mo"),
    )
    joined = t.join(cust, left_on="t_custkey", right_on="c_custkey", how="inner")
    grouped = (
        joined.group_by(["t_custkey", "c_name", "pickup_month"])
        .agg(trip_count=pl.len(), dxs=pl.col("dx"), dys=pl.col("dy"))
        .filter(pl.col("trip_count") > MIN_TRIPS)
    )

    areas = Engine.group_convex_hull_areas(grouped["dxs"], grouped["dys"])
    grouped = grouped.with_columns(
        monthly_travel_hull_area=pl.Series("monthly_travel_hull_area", areas, dtype=pl.Float64)
    ).sort(
        ["monthly_travel_hull_area", "t_custkey", "pickup_month"],
        descending=[True, False, False],
    )

    return (
        grouped.select(
            ["t_custkey", "c_name", "pickup_month", "monthly_travel_hull_area", "trip_count"]
        )
        .rename(
            {
                "t_custkey": "c_custkey",
                "c_name": "customer_name",
                "trip_count": "dropoff_count",
            }
        )
        .head(100)
    )
