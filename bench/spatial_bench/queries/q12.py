"""Q12: Rank pickups by average distance to their five nearest buildings."""

from __future__ import annotations

import polars as pl

from pycanopy import wkb_points_to_xy

id = "q12"
title = "Most isolated pickups by average distance to five nearest buildings"

K = 5

TABLES_NEEDED = {
    "building": ["b_buildingkey", "b_boundary"],
    "trip": ["t_tripkey", "t_pickuploc"],
}


def pycanopy(tables) -> pl.DataFrame:
    inputs = tables.parallel_fetch(TABLES_NEEDED)
    buildings, trip = inputs["building"], inputs["trip"]
    sf = tables.polygon_frame(buildings, "b_boundary")

    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = trip.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))
    del trip

    joined = (
        sf.lazy()
        .polygon_knn_join(query_df, "qx", "qy", k=K)
        .select(["t_tripkey", "distance_to_polygon"])
    )

    candidates = []
    for morsel in joined.collect_batched():
        averages = morsel.group_by("t_tripkey").agg(
            avg_distance_to_5_nearest=pl.col("distance_to_polygon").mean()
        )
        candidates.append(
            averages.lazy()
            .sort(
                ["avg_distance_to_5_nearest", "t_tripkey"],
                descending=[True, False],
            )
            .head(100)
            .collect()
        )
    if not candidates:
        return pl.DataFrame(schema={"t_tripkey": pl.Int64, "avg_distance_to_5_nearest": pl.Float64})
    return (
        pl.concat(candidates, how="vertical", rechunk=False)
        .lazy()
        .sort(
            ["avg_distance_to_5_nearest", "t_tripkey"],
            descending=[True, False],
        )
        .head(100)
        .collect()
    )
