"""
Q8: Count trip pickups within ~500m of each building.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame, wkb_points_to_xy

id = "q8"
title = "Trip pickups within ~500m of each building"

THRESHOLD = 0.0045  # degrees (~500m)


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    buildings, trip = pl.collect_all(
        [
            pl.scan_parquet(data_paths["building"], storage_options=STORAGE_OPTIONS).select(
                ["b_buildingkey", "b_name", "b_boundary"]
            ),
            pl.scan_parquet(data_paths["trip"], storage_options=STORAGE_OPTIONS).select(
                ["t_pickuploc"]
            ),
        ]
    )
    sf = SpatialFrame.from_wkb_polygons(buildings, "b_boundary")

    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = pl.DataFrame({"qx": qx, "qy": qy})
    del trip

    counts = (
        sf.lazy()
        .polygon_within_distance_join(query_df, "qx", "qy", distance=THRESHOLD)
        .group_by(["b_buildingkey", "b_name"])
        .agg(nearby_pickup_count=pc.agg.count())
    )
    return (
        counts.lazy()
        .sort(["nearby_pickup_count", "b_buildingkey"], descending=[True, False])
        .head(100)
        .collect()
    )
