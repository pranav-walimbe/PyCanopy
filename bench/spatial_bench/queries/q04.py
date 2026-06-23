"""Q4: Zone distribution of the top 1000 trips by tip amount.

PyCanopy: single-pass scan of t_tripkey, t_tip, and t_pickuploc together,
sort+head to find the top-1000 rows, then decode WKB only for those 1000.
The within-join then maps each pickup point to its zone.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from pycanopy import wkb_points_to_xy

id = "q4"
title = "Zone distribution of the top 1000 trips by tip"

TOP_N = 1000

compare = {"keys": ["z_zonekey"], "values": ["trip_count"]}


def pycanopy(tables) -> pl.DataFrame:
    top = (
        tables.scan("trip", ["t_tripkey", "t_tip", "t_pickuploc"])
        .sort(["t_tip", "t_tripkey"], descending=[True, False])
        .head(TOP_N)
        .collect()
    )

    qx, qy = wkb_points_to_xy(top["t_pickuploc"])
    query_df = top.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    zone = tables.table("zone", ["z_zonekey", "z_name", "z_boundary"])
    sf = tables.polygon_frame(zone, "z_boundary")

    return (
        sf.lazy()
        .within_join(query_df, "qx", "qy")
        .group_by(["z_zonekey", "z_name"])
        .agg(trip_count=pc.agg.count())
        .sort(["trip_count", "z_zonekey"], descending=[True, False])
    )
