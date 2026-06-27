"""Q6: Zone statistics for trips whose pickup falls in zones intersecting a bbox.

PyCanopy: a polygon range query finds zones whose exterior boundary truly intersects the
bounding box (the Engine performs exact ring intersection, not just an MBR overlap check),
then a within aggregate-join counts and averages trip pickups per zone without materialising
the pair frame.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from pycanopy import wkb_points_to_xy

id = "q6"
title = "Zone stats for trips intersecting a bounding box"

# Axis-aligned bounding box (min_x, min_y, max_x, max_y).
BBOX = (-112.2110, 34.4197, -111.3110, 35.3197)

_TRIP_COLS = ["t_pickuploc", "t_totalamount", "t_pickuptime", "t_dropofftime"]

TABLES_NEEDED = {"zone": ["z_zonekey", "z_name", "z_boundary"], "trip": _TRIP_COLS}

# avg_distance here is AVG(t_totalamount) on both sides
compare = {"keys": ["z_zonekey"], "values": ["total_pickups", "avg_distance"]}


def pycanopy(tables) -> pl.DataFrame:
    tables.parallel_fetch(TABLES_NEEDED)
    zone = tables.table("zone", ["z_zonekey", "z_name", "z_boundary"])
    zsf = tables.polygon_frame(zone, "z_boundary")
    cand_sf = zsf.range_filter(*BBOX)
    if cand_sf.engine.n == 0:
        return pl.DataFrame(
            schema={
                "z_zonekey": pl.Int64,
                "z_name": pl.Utf8,
                "total_pickups": pl.UInt32,
                "avg_distance": pl.Float64,
                "avg_duration": pl.Float64,
            }
        )

    trip = tables.table("trip", _TRIP_COLS)
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    qdf = trip.select(["t_totalamount", "t_pickuptime", "t_dropofftime"]).with_columns(
        pl.Series("qx", qx),
        pl.Series("qy", qy),
        duration_seconds=(pl.col("t_dropofftime") - pl.col("t_pickuptime")).dt.total_seconds(),
    )

    return (
        cand_sf.lazy()
        .within_join(qdf, "qx", "qy")
        .group_by(["z_zonekey", "z_name"])
        .agg(
            total_pickups=pc.agg.count(),
            avg_distance=pc.agg.mean("t_totalamount"),
            avg_duration=pc.agg.mean("duration_seconds"),
        )
        .sort(["total_pickups", "z_zonekey"], descending=[True, False])
    )
