"""Pinned Apache SpatialBench q4 for GeoPandas."""

import geopandas as gpd
import pandas as pd
from pandas import DataFrame


def q4(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q4 (GeoPandas): Zone distribution of top 1000 trips by tip amount.

    Steps:
      * Select top 1000 trips ordered by t_tip DESC, t_tripkey ASC.
      * Spatial join (pickup point within zone polygon).
      * Group by z_zonekey, z_name counting trips.
      * Order by trip_count DESC, z_zonekey ASC.
    Returns columns: z_zonekey, z_name, trip_count.
    """
    trip_df = pd.read_parquet(data_paths["trip"])
    if "t_tip" not in trip_df.columns:
        return pd.DataFrame(columns=["z_zonekey", "z_name", "trip_count"])
    top_trips = trip_df.sort_values(["t_tip", "t_tripkey"], ascending=[False, True]).head(1000)
    top_trips["pickup_geom"] = gpd.GeoSeries.from_wkb(top_trips["t_pickuploc"], crs="EPSG:4326")
    top_gdf = gpd.GeoDataFrame(top_trips, geometry="pickup_geom", crs="EPSG:4326")
    zone_df = pd.read_parquet(data_paths["zone"])[["z_zonekey", "z_name", "z_boundary"]]
    zone_df["zone_geom"] = gpd.GeoSeries.from_wkb(zone_df["z_boundary"], crs="EPSG:4326")
    zones_gdf = gpd.GeoDataFrame(zone_df, geometry="zone_geom", crs="EPSG:4326")[
        ["z_zonekey", "z_name", "zone_geom"]
    ]

    result = (
        gpd.sjoin(top_gdf, zones_gdf, how="inner", predicate="within")
        .groupby(["z_zonekey", "z_name"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
        .sort_values(["trip_count", "z_zonekey"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return result
