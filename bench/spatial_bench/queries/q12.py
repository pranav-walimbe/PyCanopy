"""Q12: Rank pickups by average distance to their five nearest buildings."""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from pycanopy import wkb_points_to_xy

id = "q12"
title = "Most isolated pickups by average distance to five nearest buildings"

K = 5

TABLES_NEEDED = {
    "building": ["b_buildingkey", "b_boundary"],
    "trip": ["t_tripkey", "t_pickuploc"],
}


def pycanopy(tables) -> pl.DataFrame:
    tables.parallel_fetch(TABLES_NEEDED)
    buildings = tables.table("building", ["b_buildingkey", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")

    trip = tables.table("trip", ["t_tripkey", "t_pickuploc"])
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = trip.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    averages = (
        sf.lazy()
        .polygon_knn_join(query_df, "qx", "qy", k=K)
        .group_by("t_tripkey")
        .agg(avg_distance_to_5_nearest=pc.agg.mean("distance_to_polygon"))
    )
    return averages.sort(["avg_distance_to_5_nearest", "t_tripkey"], descending=[True, False]).head(
        100
    )
