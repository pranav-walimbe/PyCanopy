"""Q10: Per-zone trip statistics, retaining zones with no trips.

PyCanopy: within-join trip pickups against zones and aggregate, then left-join the
aggregates back onto all zones so empty zones survive with num_trips = 0. The
reference uses a right sjoin and fills zeros.

The trip points are streamed through the zone index in batches and reduced to
per-zone partial aggregates each batch, so the join intermediate is bounded by the
batch size rather than trips times zone-overlap. Counts and sums are additive, so
combining the per-batch partials yields the exact single-pass result.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import polars as pl

from bench.spatial_bench import check
from bench.spatial_bench.data import wkb_points_to_xy

id = "q10"
title = "Per-zone trip stats (zones with zero trips retained)"

_TRIP_COLS = ["t_tripkey", "t_pickuploc", "t_pickuptime", "t_dropofftime", "t_distance"]
# Trip points streamed through the spatial join this many at a time.
_BATCH = 1_000_000


def pycanopy(tables) -> pl.DataFrame:
    zone = tables.table("zone", ["z_zonekey", "z_name", "z_boundary"])
    sf = tables.polygon_frame(zone, "z_boundary")

    trip = tables.table("trip", _TRIP_COLS)
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    qdf = trip.with_columns(
        pl.Series("qx", qx),
        pl.Series("qy", qy),
        duration_seconds=(pl.col("t_dropofftime") - pl.col("t_pickuptime")).dt.total_seconds(),
    ).select(["qx", "qy", "t_distance", "duration_seconds"])

    # Each batch reduces to per-zone partial sums and counts, which combine additively.
    partials = []
    for chunk in qdf.iter_slices(_BATCH):
        joined = sf.lazy().within_join(chunk, "qx", "qy").collect()
        partials.append(
            joined.group_by(["z_zonekey", "z_name"]).agg(
                sum_duration=pl.col("duration_seconds").sum(),
                sum_distance=pl.col("t_distance").sum(),
                num_trips=pl.len(),
            )
        )

    agg = (
        pl.concat(partials)
        .group_by(["z_zonekey", "z_name"])
        .agg(
            sum_duration=pl.col("sum_duration").sum(),
            sum_distance=pl.col("sum_distance").sum(),
            num_trips=pl.col("num_trips").sum(),
        )
        .with_columns(
            avg_duration=pl.col("sum_duration") / pl.col("num_trips"),
            avg_distance=pl.col("sum_distance") / pl.col("num_trips"),
        )
        .select(["z_zonekey", "z_name", "avg_duration", "avg_distance", "num_trips"])
    )

    all_zones = zone.select(["z_zonekey", "z_name"])
    result = (
        all_zones.join(agg, on=["z_zonekey", "z_name"], how="left")
        .with_columns(num_trips=pl.col("num_trips").fill_null(0))
        .rename({"z_name": "pickup_zone"})
    )
    return result.sort(["avg_duration", "z_zonekey"], descending=[True, False], nulls_last=True)


def reference(paths) -> pd.DataFrame:
    trip_df = pd.read_parquet(paths["trip"], columns=_TRIP_COLS)
    trip_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    pickups = gpd.GeoDataFrame(trip_df, geometry="pickup_geom", crs="EPSG:4326")

    zone_df = pd.read_parquet(paths["zone"], columns=["z_zonekey", "z_name", "z_boundary"])
    zone_df["zone_geom"] = gpd.GeoSeries.from_wkb(zone_df["z_boundary"], crs="EPSG:4326")
    zones = gpd.GeoDataFrame(zone_df, geometry="zone_geom", crs="EPSG:4326")

    result = (
        gpd.sjoin(pickups, zones, how="right", predicate="within")
        .assign(
            duration_seconds=lambda d: (d["t_dropofftime"] - d["t_pickuptime"]).dt.total_seconds()
        )
        .groupby(["z_zonekey", "z_name"], dropna=False)
        .agg(
            avg_duration=("duration_seconds", "mean"),
            avg_distance=("t_distance", "mean"),
            num_trips=("t_tripkey", "count"),
        )
        .reset_index()
        .assign(num_trips=lambda d: d["num_trips"].fillna(0).astype(int))
        .sort_values(by=["avg_duration", "z_zonekey"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    return result


def validate(pc_df, ref_df) -> tuple[bool, str]:
    pc_map = {r["z_zonekey"]: r["num_trips"] for r in pc_df.iter_rows(named=True)}
    ref_map = dict(zip(ref_df["z_zonekey"], ref_df["num_trips"], strict=False))
    return check.grouped(pc_map, ref_map)
