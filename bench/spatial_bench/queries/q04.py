"""
Q4: Zone distribution of the top 1000 trips by tip amount.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from pycanopy import wkb_points_to_xy

id = "q4"
title = "Zone distribution of the top 1000 trips by tip"

TOP_N = 1000

TABLES_NEEDED = {
    "trip": ["t_tripkey", "t_tip", "t_pickuploc"],
    "zone": ["z_zonekey", "z_name", "z_boundary"],
}


def pycanopy(tables) -> pl.DataFrame:
    top, zone = tables.collect_all(
        [
            tables.scan("trip", ["t_tripkey", "t_tip"])
            .sort(["t_tip", "t_tripkey"], descending=[True, False])
            .head(TOP_N)
            .select("t_tripkey"),
            tables.scan("zone", TABLES_NEEDED["zone"]),
        ]
    )
    trip = tables.collect_all(
        [
            tables.scan("trip", ["t_tripkey", "t_pickuploc"]).filter(
                pl.col("t_tripkey").is_in(top["t_tripkey"].implode())
            )
        ]
    )[0]

    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = trip.select("t_tripkey").with_columns(
        pl.Series("qx", qx),
        pl.Series("qy", qy),
    )
    del trip

    sf = tables.polygon_frame(zone, "z_boundary")

    return (
        sf.lazy()
        .within_join(query_df, "qx", "qy")
        .group_by(["z_zonekey", "z_name"])
        .agg(trip_count=pc.agg.count())
        .sort(["trip_count", "z_zonekey"], descending=[True, False])
    )
