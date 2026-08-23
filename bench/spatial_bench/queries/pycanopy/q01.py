"""
Q1: Trips starting within ~50km (0.45 degrees) of the Sedona city center.
"""

from __future__ import annotations

import polars as pl

from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame

id = "q1"
title = "Trips starting within ~50km of Sedona center"

CENTER = (-111.7610, 34.8697)
RADIUS = 0.45  # degrees (~50km, planar)


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    trip = pl.read_parquet(
        data_paths["trip"],
        columns=["t_tripkey", "t_pickuploc", "t_pickuptime"],
        storage_options=STORAGE_OPTIONS,
    )
    sf = SpatialFrame.from_wkb_points(trip, "t_pickuploc")
    near = sf.radius_query(CENTER[0], CENTER[1], RADIUS)
    result = near.with_columns(
        distance_to_center=(
            (pl.col("_x") - CENTER[0]) ** 2 + (pl.col("_y") - CENTER[1]) ** 2
        ).sqrt()
    ).select(
        "t_tripkey",
        pl.col("_x").alias("pickup_lon"),
        pl.col("_y").alias("pickup_lat"),
        "t_pickuptime",
        "distance_to_center",
    )
    del sf, trip, near
    return result.lazy().sort(["distance_to_center", "t_tripkey"]).head(100).collect()
