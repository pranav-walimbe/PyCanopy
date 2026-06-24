"""Q11: Count trips that start and end in different zones.

PyCanopy: within-join trip pickups and dropoffs against zones separately, join the
zone assignments on trip key, and count pairs where pickup zone differs from dropoff zone.
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

    pickup_zones = (
        sf.lazy()
        .within_join(pickup_df, "px", "py")
        .select(["t_tripkey", "z_zonekey"])
        .collect()
        .rename({"z_zonekey": "pickup_zone"})
    )
    dropoff_zones = (
        sf.lazy()
        .within_join(dropoff_df, "dx", "dy")
        .select(["t_tripkey", "z_zonekey"])
        .collect()
        .rename({"z_zonekey": "dropoff_zone"})
    )

    count = (
        pickup_zones.join(dropoff_zones, on="t_tripkey", how="inner")
        .filter(pl.col("pickup_zone") != pl.col("dropoff_zone"))
        .height
    )
    return pl.DataFrame({"cross_zone_trip_count": [count]})
