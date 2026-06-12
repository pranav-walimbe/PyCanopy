"""Q7: Detect route detours by comparing reported vs straight-line trip distance.

This query is not spatial-index bound: the straight-line distance is the Euclidean
distance between pickup and dropoff, computed directly in Polars. Kept for full
suite coverage and to show where PyCanopy adds no index value.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import LineString

from bench.spatial_bench import check
from pycanopy import wkb_points_to_xy

id = "q7"
title = "Route detour ratio (reported vs straight-line distance)"

DEG_PER_M = 0.000009  # 1 meter ~= 0.000009 degrees, as in the reference


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


def reference(paths) -> pd.DataFrame:
    trip_df = pd.read_parquet(
        paths["trip"], columns=["t_tripkey", "t_distance", "t_pickuploc", "t_dropoffloc"]
    )
    pickup = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    dropoff = gpd.GeoSeries.from_wkb(trip_df["t_dropoffloc"], crs="EPSG:4326")
    trip_df["reported_distance_m"] = trip_df["t_distance"].astype(float)
    line_lengths = np.fromiter(
        (
            LineString([pg, dg]).length / DEG_PER_M
            if (pg is not None and dg is not None)
            else np.nan
            for pg, dg in zip(pickup, dropoff, strict=False)
        ),
        dtype=float,
        count=len(trip_df),
    )
    trip_df = trip_df[trip_df["t_distance"] > 0].copy()
    line_lengths = line_lengths[(trip_df.index).to_numpy()]
    trip_df["line_distance_m"] = line_lengths
    trip_df["detour_ratio"] = np.divide(
        trip_df["reported_distance_m"].to_numpy(),
        line_lengths,
        out=np.full(len(trip_df), np.nan),
        where=(line_lengths != 0.0),
    )
    return trip_df[["t_tripkey", "reported_distance_m", "line_distance_m", "detour_ratio"]]


def validate(pc_df, ref_df) -> tuple[bool, str]:
    return check.rowcount(pc_df, ref_df)
