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
    sf = SpatialFrame.scan_parquet(
        data_paths["trip"],
        geometry_col="t_pickuploc",
        geometry_kind="point",
        storage_options=STORAGE_OPTIONS,
    )
    # The select keeps the scan narrow and carries the decoded coordinates out of the source
    near = (
        sf.lazy()
        .within_distance_of_point(CENTER[0], CENTER[1], RADIUS)
        .select("t_tripkey", "t_pickuptime", "_x", "_y")
        .collect()
    )
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
    del sf, near
    return result.lazy().sort(["distance_to_center", "t_tripkey"]).head(100).collect()
