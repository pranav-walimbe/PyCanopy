"""Q10: Per-zone trip statistics, retaining zones with no trips.

PyCanopy: a fused aggregate-join of trip pickups against zones, then a left-join of
the aggregates back onto all zones so empty zones survive with num_trips = 0.

The aggregate-join streams the trip points through the zone index in morsels and
reduces each morsel to per-zone partial counts and means, so the join pair frame is
never materialised; the partials combine into the exact single-pass result.
"""

from __future__ import annotations

import polars as pl

import pycanopy as pc
from pycanopy import wkb_points_to_xy

id = "q10"
title = "Per-zone trip stats (zones with zero trips retained)"

_TRIP_COLS = ["t_pickuploc", "t_pickuptime", "t_dropofftime", "t_distance"]

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

    agg = (
        sf.lazy()
        .within_join(qdf, "qx", "qy")
        .group_by(["z_zonekey", "z_name"])
        .agg(
            avg_duration=pc.agg.mean("duration_seconds"),
            avg_distance=pc.agg.mean("t_distance"),
            num_trips=pc.agg.count(),
        )
    )

    all_zones = zone.select(["z_zonekey", "z_name"])
    result = (
        all_zones.join(agg, on=["z_zonekey", "z_name"], how="left")
        .with_columns(num_trips=pl.col("num_trips").fill_null(0))
        .rename({"z_name": "pickup_zone"})
    )
    return result.sort(["avg_duration", "z_zonekey"], descending=[True, False], nulls_last=True)
