"""Q5: Monthly travel spread of repeat customers (convex hull of dropoff points).

PyCanopy: join trips to customers, group by customer and month, and compute the
convex hull area of each group's dropoff points for groups with more than five
trips. The reference builds a MultiPoint convex hull per group.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import polars as pl
from shapely.geometry import MultiPoint

from bench.spatial_bench import check
from pycanopy import Engine, wkb_points_to_xy

id = "q5"
title = "Monthly travel hull area for repeat customers"

MIN_TRIPS = 5


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", ["t_tripkey", "t_custkey", "t_dropoffloc", "t_pickuptime"])
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

    areas = [Engine.convex_hull_area(r["dxs"], r["dys"]) for r in grouped.iter_rows(named=True)]
    grouped = grouped.with_columns(
        monthly_travel_hull_area=pl.Series("monthly_travel_hull_area", areas, dtype=pl.Float64)
    ).sort(["trip_count", "t_custkey"], descending=[True, False])

    return grouped.select(
        ["t_custkey", "c_name", "pickup_month", "monthly_travel_hull_area"]
    ).rename({"t_custkey": "c_custkey", "c_name": "customer_name"})


def reference(paths) -> pd.DataFrame:
    trip_df = pd.read_parquet(
        paths["trip"], columns=["t_tripkey", "t_custkey", "t_dropoffloc", "t_pickuptime"]
    )
    cust_df = pd.read_parquet(paths["customer"], columns=["c_custkey", "c_name"])
    trip_df["dropoff_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_dropoffloc"], crs="EPSG:4326")
    joined = trip_df.merge(cust_df, left_on="t_custkey", right_on="c_custkey", how="inner")
    joined["pickup_month"] = joined["t_pickuptime"].dt.to_period("M").dt.to_timestamp()
    grouped = (
        joined.groupby(["c_custkey", "c_name", "pickup_month"], as_index=False)
        .agg(trip_count=("t_tripkey", "count"), dropoff_points=("dropoff_geom", lambda x: list(x)))
        .loc[lambda d: d["trip_count"] > MIN_TRIPS]
    )
    grouped["monthly_travel_hull_area"] = gpd.GeoSeries(
        grouped["dropoff_points"].map(MultiPoint), crs="EPSG:4326"
    ).convex_hull.area
    return (
        grouped.sort_values(["trip_count", "c_custkey"], ascending=[False, True])[
            ["c_custkey", "c_name", "pickup_month", "monthly_travel_hull_area"]
        ]
        .rename(columns={"c_name": "customer_name"})
        .reset_index(drop=True)
    )


def validate(pc_df, ref_df) -> tuple[bool, str]:
    pc_map = {
        (r["c_custkey"], str(r["pickup_month"])): r["monthly_travel_hull_area"]
        for r in pc_df.iter_rows(named=True)
    }
    ref_map = {
        (int(k), str(m)): float(a)
        for k, m, a in zip(
            ref_df["c_custkey"],
            ref_df["pickup_month"],
            ref_df["monthly_travel_hull_area"],
            strict=False,
        )
    }
    return check.grouped(pc_map, ref_map, rel_tol=1e-4)
