"""
Q10: Per-zone trip statistics, retaining zones with no trips.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame, wkb_points_to_xy

id = "q10"
title = "Per-zone trip stats (zones with zero trips retained)"

_TRIP_COLS = ["t_pickuploc", "t_pickuptime", "t_dropofftime", "t_distance"]


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    zone, trip = pl.collect_all(
        [
            pl.scan_parquet(data_paths["zone"], storage_options=STORAGE_OPTIONS).select(
                ["z_zonekey", "z_name", "z_boundary"]
            ),
            pl.scan_parquet(data_paths["trip"], storage_options=STORAGE_OPTIONS).select(_TRIP_COLS),
        ]
    )
    sf = SpatialFrame.from_wkb_polygons(zone, "z_boundary")

    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    qdf = trip.with_columns(
        pl.Series("qx", qx),
        pl.Series("qy", qy),
        t_distance=pl.col("t_distance").cast(pl.Float64),
        duration_seconds=(pl.col("t_dropofftime") - pl.col("t_pickuptime")).dt.total_seconds(),
    ).select(["qx", "qy", "t_distance", "duration_seconds"])
    del trip

    agg = (
        sf.lazy()
        .within_join(qdf, "qx", "qy")
        .group_by(["z_zonekey", "z_name"])
        .agg(
            avg_duration=pc.agg.mean("duration_seconds"),
            avg_distance=pc.agg.mean("t_distance"),
            num_trips=pc.agg.count(),
        )
    )

    all_zones = zone.select(["z_zonekey", "z_name"])
    result = (
        all_zones.join(agg, on=["z_zonekey", "z_name"], how="left")
        .with_columns(num_trips=pl.col("num_trips").fill_null(0))
        .rename({"z_name": "pickup_zone"})
    )
    return (
        result.lazy()
        .sort(
            ["avg_duration", "z_zonekey"],
            descending=[True, False],
            nulls_last=True,
        )
        .head(100)
        .collect()
    )
