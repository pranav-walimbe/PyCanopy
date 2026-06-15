"""Q4: Zone distribution of the top 1000 trips by tip amount.

PyCanopy: take the top 1000 trips by tip, within-join their pickup points against
zone polygons, and count per zone.
"""

from __future__ import annotations

import polars as pl

from pycanopy import wkb_points_to_xy

id = "q4"
title = "Zone distribution of the top 1000 trips by tip"

TOP_N = 1000

compare = {"keys": ["z_zonekey"], "values": ["trip_count"]}


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", ["t_tripkey", "t_tip", "t_pickuploc"])
    # Lazy so the head is pushed into the sort as a bounded top-N, instead of fully
    # sorting all 6M rows and dragging the wide WKB column through the permutation.
    top = trip.lazy().sort(["t_tip", "t_tripkey"], descending=[True, False]).head(TOP_N).collect()
    qx, qy = wkb_points_to_xy(top["t_pickuploc"])
    query_df = top.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    zone = tables.table("zone", ["z_zonekey", "z_name", "z_boundary"])
    sf = tables.polygon_frame(zone, "z_boundary")

    joined = sf.lazy().within_join(query_df, "qx", "qy").collect()
    return (
        joined.group_by(["z_zonekey", "z_name"])
        .agg(pl.len().alias("trip_count"))
        .sort(["trip_count", "z_zonekey"], descending=[True, False])
    )
