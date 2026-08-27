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
    zone = (
        pl.scan_parquet(data_paths["zone"], storage_options=STORAGE_OPTIONS)
        .filter(pl.col("z_name") == ZONE_NAME)
        .select(["z_boundary"])
        .collect()
    )
    # from_wkb keeps a MultiPolygon whole rather than exploding it into parts
    poly = shapely.from_wkb(zone["z_boundary"][0])

    sf = SpatialFrame.scan_parquet(
        data_paths["trip"],
        geometry_col="t_pickuploc",
        geometry_kind="point",
        storage_options=STORAGE_OPTIONS,
    )
    count = sf.lazy().points_within_distance_of_polygon(poly, 0.0).count()
    return pl.DataFrame({"trip_count_in_coconino_county": [count]})
