"""Pinned Apache SpatialBench q12 for GeoPandas."""

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import DataFrame


def q12(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q12 (GeoPandas): Rank trip pickups by average distance to their 5 nearest buildings (NLJ, memory-efficient).

    Uses a Python loop (nested loop join) to avoid materializing the full cross join.
    For each pickup, computes distances to buildings, selects the 5 closest (ties by building key ASC),
    then averages those 5 distances to produce one row per trip. Ordered by that average descending
    (most isolated pickups first) and bounded to the top 100.
    Output columns: t_tripkey, avg_distance_to_5_nearest
    """
    trips_df = pd.read_parquet(data_paths["trip"])
    buildings_df = pd.read_parquet(data_paths["building"])

    trips_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trips_df["t_pickuploc"], crs="EPSG:4326")
    buildings_df["boundary_geom"] = gpd.GeoSeries.from_wkb(
        buildings_df["b_boundary"], crs="EPSG:4326"
    )
    trips_gdf = gpd.GeoDataFrame(trips_df, geometry="pickup_geom", crs="EPSG:4326")
    buildings_gdf = gpd.GeoDataFrame(buildings_df, geometry="boundary_geom", crs="EPSG:4326")

    pickup_geoms = trips_gdf["pickup_geom"].to_list()
    trip_keys = trips_gdf["t_tripkey"].to_numpy()
    building_geoms = buildings_gdf["boundary_geom"].to_list()
    building_keys = buildings_gdf["b_buildingkey"].to_numpy()

    results = []
    # Since geopandas doesn't support KNN join, we had to choose either a cross join + filter or a NLJ.
    # The cross join would be more pandas-esque, but would require too much memory.
    # The NLJ is arguably methodologically unfair (a hand optimization) but the only way to
    # actually get the query to run.
    for i, pt in enumerate(pickup_geoms):
        dists = [pt.distance(geom) for geom in building_geoms]
        # 5 nearest buildings (ties by building key ASC), then average their distances
        nearest_idx = np.lexsort((building_keys, dists))[:5]
        avg_distance = float(np.mean([dists[idx] for idx in nearest_idx]))
        results.append(
            {
                "t_tripkey": trip_keys[i],
                "avg_distance_to_5_nearest": avg_distance,
            }
        )
    return (
        pd.DataFrame(results)
        .sort_values(["avg_distance_to_5_nearest", "t_tripkey"], ascending=[False, True])
        .head(100)  # Return only the top 100 most-isolated pickups (bounded result set)
        .reset_index(drop=True)
    )
