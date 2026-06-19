"""Q6: Zone statistics for trips whose pickup falls in zones intersecting a bbox.

PyCanopy: a polygon range query selects zones whose MBR overlaps the bounding box,
those candidates are refined to zones truly intersecting it (matching SedonaDB's
ST_Intersects, which the MBR test only over-approximates), then a within aggregate-join
counts and averages trip pickups per surviving zone without materialising the pair frame.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from shapely.geometry import box

import pycanopy as pc
from bench.spatial_bench.utils import wkb_to_polygons
from pycanopy import wkb_points_to_xy

id = "q6"
title = "Zone stats for trips intersecting a bounding box"

# Axis-aligned bounding box (min_x, min_y, max_x, max_y).
BBOX = (-112.2110, 34.4197, -111.3110, 35.3197)

_TRIP_COLS = ["t_tripkey", "t_pickuploc", "t_totalamount", "t_pickuptime", "t_dropofftime"]

# avg_distance here is AVG(t_totalamount) on both sides; avg_duration (an interval in
# SedonaDB) is left out of the value check.
compare = {"keys": ["z_zonekey"], "values": ["total_pickups", "avg_distance"]}


def pycanopy(tables) -> pl.DataFrame:
    zone = tables.table("zone", ["z_zonekey", "z_name", "z_boundary"])
    zsf = tables.polygon_frame(zone, "z_boundary")
    cand_idx = zsf.engine.range_query(*BBOX)
    if not cand_idx:
        return pl.DataFrame(
            schema={
                "z_zonekey": pl.Int64,
                "z_name": pl.Utf8,
                "total_pickups": pl.UInt32,
                "avg_distance": pl.Float64,
                "avg_duration": pl.Float64,
            }
        )
    cand = zone[pl.Series(np.asarray(cand_idx, dtype=np.uint32))]
    # range_query is an MBR test, so refine to zones that truly intersect the bbox.
    bbox = box(*BBOX)
    keep = [
        i for i, poly in enumerate(wkb_to_polygons(cand["z_boundary"])) if poly.intersects(bbox)
    ]
    if not keep:
        return pl.DataFrame(
            schema={
                "z_zonekey": pl.Int64,
                "z_name": pl.Utf8,
                "total_pickups": pl.UInt32,
                "avg_distance": pl.Float64,
                "avg_duration": pl.Float64,
            }
        )
    cand = cand[pl.Series(np.asarray(keep, dtype=np.uint32))]
    cand_sf = tables.polygon_frame(cand, "z_boundary")

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
