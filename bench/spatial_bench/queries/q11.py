"""Q11: Count trips that start and end in different zones.

PyCanopy: within-join trip pickups and dropoffs against zones separately, join on
trip, and count differing zone pairs (zones are tiered and overlap, so a trip can
fall in several).

The pickup and dropoff joins are streamed separately via collect_batched and zipped:
both probe sides derive from the same trip table (equal height, same morsel size), so
their i-th morsels cover the same trips. A trip's pickup and dropoff matches therefore
land in the same paired morsel, the per-trip cross product is local to it, and the
summed count is exact.
"""

from __future__ import annotations

import polars as pl

from pycanopy import wkb_points_to_xy

id = "q11"
title = "Count trips that cross between different zones"

compare = {"keys": [], "values": ["cross_zone_trip_count"]}


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

    # Only t_tripkey and z_zonekey are needed downstream, so push a select into each join
    # to gather just those two columns rather than the full zone boundary geometry.
    pickup_batches = (
        sf.lazy()
        .within_join(pickup_df, "px", "py")
        .select(["t_tripkey", "z_zonekey"])
        .collect_batched()
    )
    dropoff_batches = (
        sf.lazy()
        .within_join(dropoff_df, "dx", "dy")
        .select(["t_tripkey", "z_zonekey"])
        .collect_batched()
    )

    count = 0
    for pickup_joined, dropoff_joined in zip(pickup_batches, dropoff_batches, strict=True):
        pickup = _zones_of(pickup_joined, "pickup_zone")
        dropoff = _zones_of(dropoff_joined, "dropoff_zone")
        merged = pickup.join(dropoff, on="t_tripkey", how="inner")
        count += merged.filter(pl.col("pickup_zone") != pl.col("dropoff_zone")).height

    return pl.DataFrame({"cross_zone_trip_count": [count]})
