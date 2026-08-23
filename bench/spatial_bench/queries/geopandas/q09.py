"""Pinned Apache SpatialBench q9 for GeoPandas."""

from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import DataFrame


def q9(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q9 (GeoPandas): Building conflation via IoU (intersection over union) detection.

    Uses spatial self-join (predicate='intersects') to find overlapping (intersecting) building boundary polygons.
    Robust to differing GeoPandas suffix behaviors by detecting column names and falling back to index_right.
    Output columns: building_1, building_2, area1, area2, overlap_area, iou ordered by
    iou DESC, building_1 ASC, building_2 ASC.
    """
    buildings_df = pd.read_parquet(data_paths["building"])
    buildings_df["boundary_geom"] = gpd.GeoSeries.from_wkb(
        buildings_df["b_boundary"], crs="EPSG:4326"
    )
    bdf = gpd.GeoDataFrame(buildings_df, geometry="boundary_geom", crs="EPSG:4326")[
        ["b_buildingkey", "boundary_geom"]
    ].rename(columns={"b_buildingkey": "building_key"})

    pairs = gpd.sjoin(bdf, bdf, how="inner", predicate="intersects")

    left_key_candidates = ["building_key_left", "building_key_1", "building_key"]
    right_key_candidates = ["building_key_right", "building_key_2"]
    left_key_col = next(c for c in left_key_candidates if c in pairs.columns)
    right_key_col = next((c for c in right_key_candidates if c in pairs.columns), None)
    if right_key_col is None:
        pairs["_building_key_right_temp"] = bdf.loc[pairs["index_right"], "building_key"].to_numpy()
        right_key_col = "_building_key_right_temp"

    pairs = pairs.rename(
        columns={left_key_col: "building_1", right_key_col: "building_2"}
    ).rename_geometry("boundary_geom_1")
    pairs["boundary_geom_2"] = bdf.loc[pairs["index_right"], "boundary_geom"].to_numpy()

    # Filter to only building_1 < building_2 (exclude self-pairs)
    pairs = pairs[pairs["building_1"] < pairs["building_2"]]

    # Compute metrics
    boundary_geom_1_gs = gpd.GeoSeries(pairs["boundary_geom_1"], crs=pairs.crs)
    boundary_geom_2_gs = gpd.GeoSeries(pairs["boundary_geom_2"], crs=pairs.crs)
    pairs["area1"] = boundary_geom_1_gs.area
    pairs["area2"] = boundary_geom_2_gs.area
    intersection = boundary_geom_1_gs.intersection(boundary_geom_2_gs)
    pairs["overlap_area"] = intersection.area
    overlap = pairs["overlap_area"].to_numpy(dtype=float, copy=False)
    area1 = pairs["area1"].to_numpy(dtype=float, copy=False)
    area2 = pairs["area2"].to_numpy(dtype=float, copy=False)
    union = area1 + area2 - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union != 0.0)
    mask_union_zero = (union == 0.0) & (overlap > 0.0)
    if mask_union_zero.any():
        iou[mask_union_zero] = 1.0
    pairs["iou"] = iou
    result = (
        pairs[["building_1", "building_2", "area1", "area2", "overlap_area", "iou"]]
        .sort_values(["iou", "building_1", "building_2"], ascending=[False, True, True])
        .head(100)  # Return only the top 100 most-overlapping building pairs (bounded result set)
        .reset_index(drop=True)
    )
    return cast(DataFrame, result)
