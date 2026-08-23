"""Pinned Apache SpatialBench q11 for GeoPandas."""

import geopandas as gpd
import pandas as pd
from pandas import DataFrame


def q11(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q11 (GeoPandas): Count trips that cross between different zones.

    Returns a single-row DataFrame with column: cross_zone_trip_count
    """
    trip_df = pd.read_parquet(data_paths["trip"])
    zone_df = pd.read_parquet(data_paths["zone"])

    pickup_df = trip_df[["t_tripkey", "t_pickuploc"]].copy()
    pickup_df["pickup_geom"] = gpd.GeoSeries.from_wkb(pickup_df["t_pickuploc"], crs="EPSG:4326")
    pickup_points = gpd.GeoDataFrame(pickup_df, geometry="pickup_geom", crs="EPSG:4326")

    dropoff_df = trip_df[["t_tripkey", "t_dropoffloc"]].copy()
    dropoff_df["dropoff_geom"] = gpd.GeoSeries.from_wkb(dropoff_df["t_dropoffloc"], crs="EPSG:4326")
    dropoff_points = gpd.GeoDataFrame(dropoff_df, geometry="dropoff_geom", crs="EPSG:4326")

    zones_pickup = (
        zone_df[["z_zonekey", "z_boundary"]].rename(columns={"z_zonekey": "pickup_zonekey"}).copy()
    )
    zones_pickup["zone_geom"] = gpd.GeoSeries.from_wkb(zones_pickup["z_boundary"], crs="EPSG:4326")
    zones_gdf = gpd.GeoDataFrame(zones_pickup, geometry="zone_geom", crs="EPSG:4326")

    zones_dropoff = (
        zone_df[["z_zonekey", "z_boundary"]].rename(columns={"z_zonekey": "dropoff_zonekey"}).copy()
    )
    zones_dropoff["zone_geom"] = gpd.GeoSeries.from_wkb(
        zones_dropoff["z_boundary"], crs="EPSG:4326"
    )
    zones2_gdf = gpd.GeoDataFrame(zones_dropoff, geometry="zone_geom", crs="EPSG:4326")

    pickup_join = gpd.sjoin(
        pickup_points,
        zones_gdf,
        how="left",
        predicate="within",
    )
    dropoff_join = gpd.sjoin(
        dropoff_points,
        zones2_gdf,
        how="left",
        predicate="within",
    )

    merged = pickup_join[["t_tripkey", "pickup_zonekey"]].merge(
        dropoff_join[["t_tripkey", "dropoff_zonekey"]], on="t_tripkey", how="inner"
    )

    mask = (
        merged["pickup_zonekey"].notna()
        & merged["dropoff_zonekey"].notna()
        & (merged["pickup_zonekey"] != merged["dropoff_zonekey"])
    )
    count = int(mask.sum())
    return pd.DataFrame({"cross_zone_trip_count": [count]})
