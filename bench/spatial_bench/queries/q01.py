"""Q1: Trips starting within ~50km (0.45 degrees) of the Sedona city center.

PyCanopy: a single-point radius filter of all trip pickup points against the
Sedona center.
"""

from __future__ import annotations

import polars as pl

id = "q1"
title = "Trips starting within ~50km of Sedona center"

CENTER = (-111.7610, 34.8697)
RADIUS = 0.45  # degrees (~50km, planar)

# SedonaDB returns t_tripkey, pickup_lon/lat, t_pickuptime, distance_to_center.
compare = {"keys": ["t_tripkey"], "values": ["distance_to_center"]}


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", ["t_tripkey", "t_pickuploc", "t_pickuptime"])
    sf = tables.point_frame(trip, "t_pickuploc")
    near = sf.radius_query(CENTER[0], CENTER[1], RADIUS)
    return (
        near.with_columns(
            distance_to_center=(
                (pl.col("_x") - CENTER[0]) ** 2 + (pl.col("_y") - CENTER[1]) ** 2
            ).sqrt()
        )
        .select(
            "t_tripkey",
            pl.col("_x").alias("pickup_lon"),
            pl.col("_y").alias("pickup_lat"),
            "t_pickuptime",
            "distance_to_center",
        )
        .sort(["distance_to_center", "t_tripkey"])
    )
