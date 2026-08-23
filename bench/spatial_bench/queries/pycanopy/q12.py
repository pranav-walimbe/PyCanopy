"""Q12: Rank pickups by average distance to their five nearest buildings."""

from __future__ import annotations

import polars as pl

from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame, wkb_points_to_xy

id = "q12"
title = "Most isolated pickups by average distance to five nearest buildings"

K = 5


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    buildings, trip = pl.collect_all(
        [
            pl.scan_parquet(data_paths["building"], storage_options=STORAGE_OPTIONS).select(
                ["b_buildingkey", "b_boundary"]
            ),
            pl.scan_parquet(data_paths["trip"], storage_options=STORAGE_OPTIONS).select(
                ["t_tripkey", "t_pickuploc"]
            ),
        ]
    )
    sf = SpatialFrame.from_wkb_polygons(buildings, "b_boundary")

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
