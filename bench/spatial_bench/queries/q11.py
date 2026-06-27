"""Q11: Count trips that start and end in different zones.

PyCanopy: within-join trip pickups and dropoffs against zones, streamed in aligned morsels;
per morsel join the assignments on trip key and count pairs whose pickup and dropoff zones
differ. Streaming bounds the assignment-join transient instead of materialising two full frames.
"""

from __future__ import annotations

import polars as pl

from pycanopy import wkb_points_to_xy

id = "q11"
title = "Count trips that cross between different zones"

TABLES_NEEDED = {
    "trip": ["t_tripkey", "t_pickuploc", "t_dropoffloc"],
    "zone": ["z_zonekey", "z_boundary"],
}

compare = {"keys": [], "values": ["cross_zone_trip_count"]}


def pycanopy(tables) -> pl.DataFrame:
    tables.parallel_fetch(TABLES_NEEDED)
    trip = tables.table("trip", ["t_tripkey", "t_pickuploc", "t_dropoffloc"])
    zone = tables.table("zone", ["z_zonekey", "z_boundary"])
    sf = tables.polygon_frame(zone, "z_boundary")

    px, py = wkb_points_to_xy(trip["t_pickuploc"])
    dx, dy = wkb_points_to_xy(trip["t_dropoffloc"])
    keys = trip.select("t_tripkey")
    pickup_df = keys.with_columns(pl.Series("px", px), pl.Series("py", py))
    dropoff_df = keys.with_columns(pl.Series("dx", dx), pl.Series("dy", dy))

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

    # Aligned morsels carry the same trips on each side, so per-morsel counts sum to the global count
    count = 0
    for pickup, dropoff in zip(pickup_batches, dropoff_batches):
        count += (
            pickup.rename({"z_zonekey": "pickup_zone"})
            .join(dropoff.rename({"z_zonekey": "dropoff_zone"}), on="t_tripkey", how="inner")
            .filter(pl.col("pickup_zone") != pl.col("dropoff_zone"))
            .height
        )
    return pl.DataFrame({"cross_zone_trip_count": [count]})
