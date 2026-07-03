"""Q1: Trips starting within ~50km (0.45 degrees) of the Sedona city center.

PyCanopy: a within-distance join of all trip pickup points against the single
center point.
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
    center_df = pl.DataFrame({"cx": [CENTER[0]], "cy": [CENTER[1]]})
    joined = sf.lazy().within_distance_join(center_df, "cx", "cy", distance=RADIUS).collect()
    return (
        joined.with_columns(
            distance_to_center=(
                (pl.col("_x") - pl.col("cx")) ** 2 + (pl.col("_y") - pl.col("cy")) ** 2
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
