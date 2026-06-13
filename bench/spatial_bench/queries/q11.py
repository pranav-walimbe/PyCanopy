"""Q11: Count trips that start and end in different zones.

PyCanopy: within-join trip pickups and dropoffs against zones separately, join on
trip, and count differing zone pairs. Mirrors the reference, which forms the
pickup-zone x dropoff-zone cross product per trip (zones are tiered and overlap)
and counts pairs whose zones differ.

The pickup and dropoff joins are streamed separately via collect_batched and zipped:
both probe sides derive from the same trip table (equal height, same morsel size), so
their i-th morsels cover the same trips. A trip's pickup and dropoff matches therefore
land in the same paired morsel, the per-trip cross product is local to it, and the
summed count is exact.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import polars as pl

from bench.spatial_bench import check
from pycanopy import wkb_points_to_xy

id = "q11"
title = "Count trips that cross between different zones"


def _zones_of(joined, zone_col) -> pl.DataFrame:
    """Return (t_tripkey, zone_col) for every zone containing each joined point."""
    return joined.select(["t_tripkey", "z_zonekey"]).rename({"z_zonekey": zone_col})


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", ["t_tripkey", "t_pickuploc", "t_dropoffloc"])
    zone = tables.table("zone", ["z_zonekey", "z_boundary"])
    sf = tables.polygon_frame(zone, "z_boundary")

    px, py = wkb_points_to_xy(trip["t_pickuploc"])
    dx, dy = wkb_points_to_xy(trip["t_dropoffloc"])
    keys = trip.select("t_tripkey")
    pickup_df = keys.with_columns(pl.Series("px", px), pl.Series("py", py))
    dropoff_df = keys.with_columns(pl.Series("dx", dx), pl.Series("dy", dy))

    pickup_batches = sf.lazy().within_join(pickup_df, "px", "py").collect_batched()
    dropoff_batches = sf.lazy().within_join(dropoff_df, "dx", "dy").collect_batched()

    count = 0
    for pickup_joined, dropoff_joined in zip(pickup_batches, dropoff_batches, strict=True):
        pickup = _zones_of(pickup_joined, "pickup_zone")
        dropoff = _zones_of(dropoff_joined, "dropoff_zone")
        merged = pickup.join(dropoff, on="t_tripkey", how="inner")
        count += merged.filter(pl.col("pickup_zone") != pl.col("dropoff_zone")).height

    return pl.DataFrame({"cross_zone_trip_count": [count]})


def reference(paths) -> pd.DataFrame:
    trip_df = pd.read_parquet(paths["trip"], columns=["t_tripkey", "t_pickuploc", "t_dropoffloc"])
    zone_df = pd.read_parquet(paths["zone"], columns=["z_zonekey", "z_boundary"])
    zone_df["zone_geom"] = gpd.GeoSeries.from_wkb(zone_df["z_boundary"], crs="EPSG:4326")

    def zone_join(wkb_col, key_name):
        df = trip_df[["t_tripkey", wkb_col]].copy()
        df["geom"] = gpd.GeoSeries.from_wkb(df[wkb_col], crs="EPSG:4326")
        points = gpd.GeoDataFrame(df, geometry="geom", crs="EPSG:4326")
        zones = gpd.GeoDataFrame(
            zone_df.rename(columns={"z_zonekey": key_name}), geometry="zone_geom", crs="EPSG:4326"
        )
        return gpd.sjoin(points, zones[[key_name, "zone_geom"]], how="left", predicate="within")

    pickup_join = zone_join("t_pickuploc", "pickup_zonekey")
    dropoff_join = zone_join("t_dropoffloc", "dropoff_zonekey")
    merged = pickup_join[["t_tripkey", "pickup_zonekey"]].merge(
        dropoff_join[["t_tripkey", "dropoff_zonekey"]], on="t_tripkey", how="inner"
    )
    mask = (
        merged["pickup_zonekey"].notna()
        & merged["dropoff_zonekey"].notna()
        & (merged["pickup_zonekey"] != merged["dropoff_zonekey"])
    )
    return pd.DataFrame({"cross_zone_trip_count": [int(mask.sum())]})


def validate(pc_df, ref_df) -> tuple[bool, str]:
    return check.scalar(
        pc_df["cross_zone_trip_count"][0], int(ref_df["cross_zone_trip_count"].iloc[0])
    )
