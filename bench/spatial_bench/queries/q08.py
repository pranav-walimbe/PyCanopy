"""Q8: Count trip pickups within ~500m of each building.

PyCanopy: a polygon within-distance aggregate-join of building footprints against trip
pickup points, counted per building. The join is high fanout (each pickup can fall near
many buildings), so the aggregate-join reduces each morsel to per-building partial counts
rather than materialising the full pair frame.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from pycanopy import wkb_points_to_xy

id = "q8"
title = "Trip pickups within ~500m of each building"

THRESHOLD = 0.0045  # degrees (~500m)

TABLES_NEEDED = {"building": ["b_buildingkey", "b_name", "b_boundary"], "trip": ["t_pickuploc"]}

compare = {"keys": ["b_buildingkey"], "values": ["nearby_pickup_count"]}


def pycanopy(tables) -> pl.DataFrame:
    tables.parallel_fetch(TABLES_NEEDED)
    buildings = tables.table("building", ["b_buildingkey", "b_name", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")

    trip = tables.table("trip", ["t_pickuploc"])
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = pl.DataFrame({"qx": qx, "qy": qy})

    return (
        sf.lazy()
        .polygon_within_distance_join(query_df, "qx", "qy", distance=THRESHOLD)
        .group_by(["b_buildingkey", "b_name"])
        .agg(nearby_pickup_count=pc.agg.count())
        .sort(["nearby_pickup_count", "b_buildingkey"], descending=[True, False])
    )
