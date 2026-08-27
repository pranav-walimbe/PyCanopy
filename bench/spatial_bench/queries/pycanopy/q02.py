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
    # Zone row group statistics span the whole name range so the predicate prunes no bytes
    names = (
        pl.scan_parquet(
            data_paths["zone"], storage_options=STORAGE_OPTIONS, include_file_paths="_file"
        )
        .select(["z_name", "_file"])
        .collect()
    )
    hit = names.with_row_index("_row").filter(pl.col("z_name") == ZONE_NAME).head(1)
    zone_file = hit["_file"][0]
    # Row offset inside its own file lets the slice skip every other row group
    local_row = hit["_row"][0] - int((names["_file"] == zone_file).arg_max())

    zone = (
        pl.scan_parquet(zone_file, storage_options=STORAGE_OPTIONS)
        .select(["z_boundary"])
        .slice(local_row, 1)
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
