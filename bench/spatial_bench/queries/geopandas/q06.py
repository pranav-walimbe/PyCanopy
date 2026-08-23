"""Pinned Apache SpatialBench q6 for GeoPandas."""

import geopandas as gpd
import pandas as pd
from pandas import DataFrame
from shapely.geometry import Polygon


def q6(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q6 (GeoPandas): Zone statistics for trips intersecting a bounding box.

    Mirrors original SQL intent:
      * Filter zones intersecting the provided bounding box polygon.
      * Count trips whose pickup point lies within each zone (inner semantics: zones with 0 pickups excluded).
      * Compute:
          total_pickups = COUNT(t_tripkey)
          avg_distance  = AVG(t_distance)
          avg_duration  = AVG(t_dropofftime - t_pickuptime) in seconds
      * Order by total_pickups DESC, z_zonekey ASC.
    Returns DataFrame with columns: z_zonekey, z_name, total_pickups, avg_distance, avg_duration
    """
    trip_df = pd.read_parquet(data_paths["trip"])
    zone_df = pd.read_parquet(data_paths["zone"])

    trip_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    pickup_points = gpd.GeoDataFrame(trip_df, geometry="pickup_geom", crs="EPSG:4326")

    zone_df["zone_geom"] = gpd.GeoSeries.from_wkb(zone_df["z_boundary"], crs="EPSG:4326")
    zones_gdf = gpd.GeoDataFrame(zone_df, geometry="zone_geom", crs="EPSG:4326")[
        ["z_zonekey", "z_name", "zone_geom"]
    ]

    bbox_poly = Polygon(
        [
            (-112.2110, 34.4197),
            (-111.3110, 34.4197),
            (-111.3110, 35.3197),
            (-112.2110, 35.3197),
            (-112.2110, 34.4197),
        ]
    )

    candidate_zones = zones_gdf[
        zones_gdf["zone_geom"].notna() & zones_gdf["zone_geom"].intersects(bbox_poly)
    ]

    distance_col = "t_distance" if "t_distance" in trip_df.columns else None

    result = (
        gpd.sjoin(pickup_points, candidate_zones, how="inner", predicate="within")
        .assign(
            _duration_seconds=lambda d: (d["t_dropofftime"] - d["t_pickuptime"]).dt.total_seconds(),
            _distance_metric=lambda d: d[distance_col] if distance_col else pd.NA,
        )
        .groupby(["z_zonekey", "z_name"], as_index=False)
        .agg(
            total_pickups=("t_tripkey", "count"),
            avg_distance=("_distance_metric", "mean"),
            avg_duration=("_duration_seconds", "mean"),
        )
        .sort_values(["total_pickups", "z_zonekey"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return result
