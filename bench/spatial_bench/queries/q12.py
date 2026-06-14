"""Q12: The 5 nearest buildings to each trip pickup location.

PyCanopy: a point-to-polygon kNN join of trip pickups against building footprints.
The library streams the trip points through the building index in morsels and
concatenates internally, so the join intermediate stays bounded (collect auto-streams
a large-probe join).
"""

from __future__ import annotations

import polars as pl

from pycanopy import wkb_points_to_xy

id = "q12"
title = "5 nearest buildings to each trip pickup"

K = 5

# kNN ties can pick different buildings, so the per-trip distances are compared rather
# than building identity (the SedonaDB column is distance_to_building).
compare = {"keys": ["t_tripkey"], "values": [("distance_to_polygon", "distance_to_building")]}


def pycanopy(tables) -> pl.DataFrame:
    buildings = tables.table("building", ["b_buildingkey", "b_name", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")

    trip = tables.table("trip", ["t_tripkey", "t_pickuploc"])
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = trip.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    joined = sf.lazy().polygon_knn_join(query_df, "qx", "qy", k=K).collect()
    return joined.sort(["distance_to_polygon", "b_buildingkey"])
