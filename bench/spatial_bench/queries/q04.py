"""Q4: Zone distribution of the top 1000 trips by tip amount.

PyCanopy: take the top 1000 trips by tip, within-join their pickup points against
zone polygons, and count per zone.
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
    # Two scans: the first reads only (tripkey, tip) to find the top-N cheaply without
    # touching the WKB column. The second reads WKB for just those N rows via the
    # is_in predicate pushed into the parquet scan.
    top_keys = (
        tables.scan("trip", ["t_tripkey", "t_tip"])
        .sort(["t_tip", "t_tripkey"], descending=[True, False])
        .head(TOP_N)
        .collect()["t_tripkey"]
    )
    winners = (
        tables.scan("trip", ["t_tripkey", "t_pickuploc"])
        .filter(pl.col("t_tripkey").is_in(top_keys))
        .collect()
    )
    qx, qy = wkb_points_to_xy(winners["t_pickuploc"])
    query_df = winners.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    zone = tables.table("zone", ["z_zonekey", "z_name", "z_boundary"])
    sf = tables.polygon_frame(zone, "z_boundary")

    return (
        sf.lazy()
        .within_join(query_df, "qx", "qy")
        .group_by(["z_zonekey", "z_name"])
        .agg(trip_count=pc.agg.count())
        .sort(["trip_count", "z_zonekey"], descending=[True, False])
    )
