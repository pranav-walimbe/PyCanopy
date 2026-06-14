"""Q9: Building conflation via IoU — find overlapping building footprints.

PyCanopy: a polygon self-intersection join over building footprints, with overlap
area and IoU computed per pair.
"""

from __future__ import annotations

import numpy as np
import polars as pl

id = "q9"
title = "Building overlap detection via IoU"

compare = {"keys": ["building_1", "building_2"], "values": ["iou"], "rel_tol": 1e-4}


def pycanopy(tables) -> pl.DataFrame:
    buildings = tables.table("building", ["b_buildingkey", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")
    pairs = sf.intersects_pairs()

    schema = {"building_1": pl.Int64, "building_2": pl.Int64, "iou": pl.Float64}
    if pairs.height == 0:
        return pl.DataFrame(schema=schema)

    keys = buildings["b_buildingkey"].to_numpy()
    k1 = keys[pairs["left"].to_numpy()]
    k2 = keys[pairs["right"].to_numpy()]
    # Order each pair so building_1 < building_2 by key, matching the canonical query.
    swap = k1 > k2
    b1 = np.where(swap, k2, k1)
    b2 = np.where(swap, k1, k2)
    return pl.DataFrame(
        {"building_1": b1, "building_2": b2, "iou": pairs["iou"].to_numpy()},
        schema=schema,
    ).sort(["iou", "building_1", "building_2"], descending=[True, False, False])
