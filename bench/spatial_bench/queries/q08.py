"""Q8: Count trip pickups within ~500m of each building.

PyCanopy: a polygon within-distance join of building footprints against trip pickup
points, grouped per building.
"""

from __future__ import annotations

import polars as pl

from pycanopy import wkb_points_to_xy

id = "q8"
title = "Trip pickups within ~500m of each building"

THRESHOLD = 0.0045  # degrees (~500m)

compare = {"keys": ["b_buildingkey"], "values": ["nearby_pickup_count"]}


def pycanopy(tables) -> pl.DataFrame:
    buildings = tables.table("building", ["b_buildingkey", "b_name", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")

    trip = tables.table("trip", ["t_pickuploc"])
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = pl.DataFrame({"qx": qx, "qy": qy})

    joined = (
        sf.lazy().polygon_within_distance_join(query_df, "qx", "qy", distance=THRESHOLD).collect()
    )
    return (
        joined.group_by(["b_buildingkey", "b_name"])
        .agg(pl.len().alias("nearby_pickup_count"))
        .sort(["nearby_pickup_count", "b_buildingkey"], descending=[True, False])
    )
