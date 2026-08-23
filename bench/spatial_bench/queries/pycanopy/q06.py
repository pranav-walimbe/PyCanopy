"""
Q6: Zone statistics for trips whose pickup falls in zones intersecting a bbox.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame, wkb_points_to_xy

id = "q6"
title = "Zone stats for trips intersecting a bounding box"

# Axis-aligned bounding box (min_x, min_y, max_x, max_y).
BBOX = (-112.2110, 34.4197, -111.3110, 35.3197)

_TRIP_COLS = ["t_pickuploc", "t_distance", "t_pickuptime", "t_dropofftime"]


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    zone, trip = pl.collect_all(
        [
            pl.scan_parquet(data_paths["zone"], storage_options=STORAGE_OPTIONS).select(
                ["z_zonekey", "z_name", "z_boundary"]
            ),
            pl.scan_parquet(data_paths["trip"], storage_options=STORAGE_OPTIONS).select(_TRIP_COLS),
        ]
    )
    zsf = SpatialFrame.from_wkb_polygons(zone, "z_boundary")
    cand_sf = zsf.range_filter(*BBOX)
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    qdf = trip.select(["t_distance", "t_pickuptime", "t_dropofftime"]).with_columns(
        pl.Series("qx", qx),
        pl.Series("qy", qy),
        t_distance=pl.col("t_distance").cast(pl.Float64),
        duration_seconds=(pl.col("t_dropofftime") - pl.col("t_pickuptime")).dt.total_seconds(),
    )
    del trip

    return (
        cand_sf.lazy()
        .within_join(qdf, "qx", "qy")
        .group_by(["z_zonekey", "z_name"])
        .agg(
            total_pickups=pc.agg.count(),
            avg_distance=pc.agg.mean("t_distance"),
            avg_duration=pc.agg.mean("duration_seconds"),
        )
        .sort(["total_pickups", "z_zonekey"], descending=[True, False])
    )
