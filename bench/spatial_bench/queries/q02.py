"""Q2: Count trips whose pickup point falls within the Coconino County zone.

PyCanopy: filter the trip points to those within (distance 0 of) the single named
zone polygon.
"""

from __future__ import annotations

import polars as pl

from bench.spatial_bench.utils import wkb_to_polygons

id = "q2"
title = "Count trips starting within Coconino County zone"

ZONE_NAME = "Coconino County"

TABLES_NEEDED = {"zone": ["z_name", "z_boundary"], "trip": ["t_pickuploc"]}

compare = {"keys": [], "values": ["trip_count_in_coconino_county"]}


def pycanopy(tables) -> pl.DataFrame:
    tables.parallel_fetch(TABLES_NEEDED)
    zone = tables.table("zone", ["z_name", "z_boundary"])
    target = zone.filter(pl.col("z_name") == ZONE_NAME).head(1)
    if target.height == 0:
        return pl.DataFrame({"trip_count_in_coconino_county": [0]})
    poly = wkb_to_polygons(target["z_boundary"])[0]

    trip = tables.table("trip", ["t_pickuploc"])
    sf = tables.point_frame(trip, "t_pickuploc")
    # Only the count is needed, so take the engine's matching indices directly and skip
    # gathering the in-zone rows into a DataFrame.
    idx = sf.engine.points_within_distance_of_polygon(poly, 0.0)
    return pl.DataFrame({"trip_count_in_coconino_county": [len(idx)]})
