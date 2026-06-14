"""Q10: Per-zone trip statistics, retaining zones with no trips.

PyCanopy: within-join trip pickups against zones and aggregate, then left-join the
aggregates back onto all zones so empty zones survive with num_trips = 0.

The library streams the trip points through the zone index in morsels via
collect_batched; each morsel is reduced to per-zone partial aggregates here, so the
join intermediate is bounded by the morsel size rather than trips times zone-overlap.
Counts and sums are additive, so combining the per-morsel partials yields the exact
single-pass result.
"""

from __future__ import annotations

import polars as pl

from pycanopy import wkb_points_to_xy

id = "q10"
title = "Per-zone trip stats (zones with zero trips retained)"

_TRIP_COLS = ["t_tripkey", "t_pickuploc", "t_pickuptime", "t_dropofftime", "t_distance"]

# avg_duration is an interval in SedonaDB (Timedelta) vs float seconds here, so it is
# left out of the value check; num_trips and avg_distance are compared.
compare = {"keys": ["z_zonekey"], "values": ["num_trips", "avg_distance"]}


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

    # Each morsel reduces to per-zone partial sums and counts, which combine additively.
    partials = [
        joined.group_by(["z_zonekey", "z_name"]).agg(
            sum_duration=pl.col("duration_seconds").sum(),
            sum_distance=pl.col("t_distance").sum(),
            num_trips=pl.len(),
        )
        for joined in sf.lazy().within_join(qdf, "qx", "qy").collect_batched()
    ]

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
