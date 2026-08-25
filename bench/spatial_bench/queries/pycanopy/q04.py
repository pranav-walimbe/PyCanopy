"""
Q4: Zone distribution of the top 1000 trips by tip amount.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame, wkb_points_to_xy

id = "q4"
title = "Zone distribution of the top 1000 trips by tip"

TOP_N = 1000


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    trip_scan = pl.scan_parquet(data_paths["trip"], storage_options=STORAGE_OPTIONS)
    top, zone = pl.collect_all(
        [
            trip_scan.select(["t_tripkey", "t_tip"])
            .top_k(TOP_N, by=["t_tip", "t_tripkey"], reverse=[False, True])
            .select("t_tripkey"),
            pl.scan_parquet(data_paths["zone"], storage_options=STORAGE_OPTIONS).select(
                ["z_zonekey", "z_name", "z_boundary"]
            ),
        ]
    )
    # Only the top trips need geometry and this second scan runs after the first result
    trip = (
        trip_scan.select(["t_tripkey", "t_pickuploc"])
        .filter(pl.col("t_tripkey").is_in(top["t_tripkey"].implode()))
        .collect()
    )

    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = trip.select("t_tripkey").with_columns(
        pl.Series("qx", qx),
        pl.Series("qy", qy),
    )
    del trip

    sf = SpatialFrame.from_wkb_polygons(zone, "z_boundary")

    return (
        sf.lazy()
        .within_join(query_df, "qx", "qy")
        .group_by(["z_zonekey", "z_name"])
        .agg(trip_count=pc.agg.count())
        .sort(["trip_count", "z_zonekey"], descending=[True, False])
    )
