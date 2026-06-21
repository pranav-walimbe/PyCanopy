"""Q5: Monthly travel spread of repeat customers (convex hull of dropoff points).

PyCanopy: join trips to customers, group by customer and month, and compute the
convex hull area of each group's dropoff points for groups with more than five trips.
"""

from __future__ import annotations

import polars as pl

from pycanopy import Engine, wkb_points_to_xy

id = "q5"
title = "Monthly travel hull area for repeat customers"

MIN_TRIPS = 5

compare = {
    "keys": ["c_custkey", "pickup_month"],
    "values": ["monthly_travel_hull_area"],
    "rel_tol": 1e-4,
}


def pycanopy(tables) -> pl.DataFrame:
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

    areas = [
        Engine.convex_hull_area(dxs, dys)
        for dxs, dys in zip(grouped["dxs"], grouped["dys"], strict=True)
    ]
    grouped = grouped.with_columns(
        monthly_travel_hull_area=pl.Series("monthly_travel_hull_area", areas, dtype=pl.Float64)
    ).sort(["trip_count", "t_custkey"], descending=[True, False])

    return grouped.select(
        ["t_custkey", "c_name", "pickup_month", "monthly_travel_hull_area"]
    ).rename({"t_custkey": "c_custkey", "c_name": "customer_name"})
