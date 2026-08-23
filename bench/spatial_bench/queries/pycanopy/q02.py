"""
Q2: Count trips whose pickup point falls within the Coconino County zone.
"""

from __future__ import annotations

import polars as pl
import shapely

from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame

id = "q2"
title = "Count trips starting within Coconino County zone"

ZONE_NAME = "Coconino County"


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    zone, trip = pl.collect_all(
        [
            pl.scan_parquet(data_paths["zone"], storage_options=STORAGE_OPTIONS).select(
                ["z_name", "z_boundary"]
            ),
            pl.scan_parquet(data_paths["trip"], storage_options=STORAGE_OPTIONS).select(
                ["t_pickuploc"]
            ),
        ]
    )
    target = zone.filter(pl.col("z_name") == ZONE_NAME).head(1)
    # from_wkb keeps a MultiPolygon whole rather than exploding it into parts
    poly = shapely.from_wkb(target["z_boundary"][0])

    sf = SpatialFrame.from_wkb_points(trip, "t_pickuploc")
    # Only the count is needed, so take the engine's matching indices directly and skip
    # gathering the in-zone rows into a DataFrame.
    idx = sf.engine.points_within_distance_of_polygon(poly, 0.0)
    return pl.DataFrame({"trip_count_in_coconino_county": [len(idx)]})
