"""Pinned Apache SpatialBench q2 for GeoPandas."""

import geopandas as gpd
import pandas as pd
from pandas import DataFrame
from shapely import wkb


def q2(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q2 (GeoPandas): Count trips starting within Coconino County zone.

    Finds the first zone row where z_name == 'Coconino County' and counts trips whose
    pickup point intersects that polygon. Returns single-row DataFrame with
    trip_count_in_coconino_county.
    """
    trip_df = pd.read_parquet(data_paths["trip"])
    zone_df = pd.read_parquet(data_paths["zone"])
    target = zone_df[zone_df["z_name"] == "Coconino County"].head(1)
    if target.empty:
        return pd.DataFrame({"trip_count_in_coconino_county": [0]})
    poly = wkb.loads(target.iloc[0]["z_boundary"])
    trip_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    # Ensure intersects is called on a GeoSeries, not a Series
    pickup_geoms = gpd.GeoSeries(trip_df["pickup_geom"], crs="EPSG:4326")
    count = int(pickup_geoms.intersects(poly).sum())
    return pd.DataFrame({"trip_count_in_coconino_county": [count]})
