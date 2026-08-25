"""
Q11: Count trips that start and end in different zones.
"""

from __future__ import annotations

import polars as pl

from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame, wkb_points_to_xy

id = "q11"
title = "Count trips that cross between different zones"


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    trip, zone = pl.collect_all(
        [
            pl.scan_parquet(data_paths["trip"], storage_options=STORAGE_OPTIONS).select(
                ["t_tripkey", "t_pickuploc", "t_dropoffloc"]
            ),
            pl.scan_parquet(data_paths["zone"], storage_options=STORAGE_OPTIONS).select(
                ["z_zonekey", "z_boundary"]
            ),
        ]
    )
    sf = SpatialFrame.from_wkb_polygons(zone, "z_boundary")

    px, py = wkb_points_to_xy(trip["t_pickuploc"])
    dx, dy = wkb_points_to_xy(trip["t_dropoffloc"])
    keys = trip.select("t_tripkey")
    pickup_df = keys.with_columns(pl.Series("px", px), pl.Series("py", py))
    dropoff_df = keys.with_columns(pl.Series("dx", dx), pl.Series("dy", dy))
    del trip

    pickup_batches = (
        sf.lazy()
        .within_join(pickup_df, "px", "py")
        .select(["t_tripkey", "z_zonekey"])
        .collect_batched()
    )
    dropoff_batches = (
        sf.lazy()
        .within_join(dropoff_df, "dx", "dy")
        .select(["t_tripkey", "z_zonekey"])
        .collect_batched()
    )

    # Aligned morsels carry the same trips on each side and per-morsel counts sum to the global count
    count = 0
    for pickup, dropoff in zip(pickup_batches, dropoff_batches, strict=True):
        count += (
            pickup.rename({"z_zonekey": "pickup_zone"})
            .join(dropoff.rename({"z_zonekey": "dropoff_zone"}), on="t_tripkey", how="inner")
            .filter(pl.col("pickup_zone") != pl.col("dropoff_zone"))
            .height
        )
    return pl.DataFrame({"cross_zone_trip_count": [count]})
