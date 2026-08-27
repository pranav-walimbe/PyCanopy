# Quick Start

## Installation

```bash
pip install pycanopy
```

Pre-built wheels for Linux, macOS, and Windows. No Rust toolchain required.

## Point dataset

```python
import polars as pl
from pycanopy import SpatialFrame

cities = pl.read_parquet("cities.parquet")
city_sf = SpatialFrame(cities, x_col="lon", y_col="lat")

# Bounding-box query combined with a scalar predicate
result = (
    city_sf.lazy()
    .filter(pl.col("population") > 100_000)
    .range_query(min_x=-10.0, min_y=35.0, max_x=40.0, max_y=70.0)
    .collect()
)

# k-nearest neighbors
nearest = city_sf.lazy().knn(x=2.35, y=48.85, k=5).collect()

# kNN join: for each query point, the 3 nearest dataset rows
query_df = pl.DataFrame({"qx": [2.35, 13.4], "qy": [48.85, 52.5]})
result = city_sf.lazy().knn_join(query_df, x_col="qx", y_col="qy", k=3).collect()
```

## Parquet and GeoParquet

Keep a Parquet source deferred until the query is collected:

```python
trip_sf = SpatialFrame.scan_parquet(
    "trips.parquet",
    geometry_col="geometry",
    geometry_kind="point",
)

result = (
    trip_sf.lazy()
    .filter(pl.col("fare") > 20)
    .within_distance_of_point(cx=-111.7610, cy=34.8697, distance=0.45)
    .select("trip_id", "fare")
    .collect()
)
```

For GeoParquet, `geometry_col` and `geometry_kind` are inferred from file metadata when omitted.
Local paths, globs, path lists, and cloud URLs accepted by `polars.scan_parquet` are supported.

## Polygon dataset

```python
from shapely.geometry import box
from pycanopy import SpatialFrame

polygons = [box(i, 0, i + 0.9, 0.9) for i in range(100_000)]
zones = pl.DataFrame(
    {
        "zone_id": list(range(100_000)),
        "zone_name": [f"zone-{i}" for i in range(100_000)],
        "geom": polygons,
    }
)
zone_sf = SpatialFrame.from_polygons(zones, geometry_col="geom")

# Which polygons contain this point?
containing = zone_sf.lazy().contains(x=5.5, y=0.5).collect()

# For each query point, find the polygons that contain it
query_df = pl.DataFrame({"qx": [5.5, 12.3], "qy": [0.5, 0.5]})
result = zone_sf.lazy().within_join(query_df, x_col="qx", y_col="qy").collect()
```

## Aggregate over a join

```python
import pycanopy as pc

trips = pl.DataFrame(
    {
        "trip_id": [1, 2, 3],
        "lon": [5.5, 12.3, 42.1],
        "lat": [0.5, 0.5, 0.5],
        "fare": [12.0, 18.0, 9.0],
    }
)

# Count trips per zone and average fare, reduced over a streamed join
# The full pair frame is never materialized
stats = (
    zone_sf.lazy()
    .within_join(trips, x_col="lon", y_col="lat")
    .group_by(["zone_id", "zone_name"])
    .agg(trip_count=pc.agg.count(), avg_fare=pc.agg.mean("fare"))
)
```

## Inspecting the plan

```python
lf = (
    city_sf.lazy()
    .range_query(min_x=-10.0, min_y=35.0, max_x=40.0, max_y=70.0)
    .filter(pl.col("population") > 100_000)
)

print(lf.explain())
# RANGE_QUERY [(-10, 35) → (40, 70)]
# FROM
#   FILTER [(col("population")) > (dyn int: 100000)]
#   FROM
#     DF [N=100,000; path: EXPR]
```

The optimizer flipped the declaration order so the scalar filter runs first. The spatial operation then masks the rows that survived that filter.

## Streaming large results

```python
# Stream join morsels directly to Parquet
zone_sf.lazy().polygon_knn_join(trips, "lon", "lat", k=5).sink_parquet("nearest.parquet")

# Expose the join as a Polars LazyFrame for downstream operations
(
    zone_sf.lazy()
    .polygon_knn_join(trips, "lon", "lat", k=5)
    .select(["trip_id", "zone_id", "distance_to_polygon"])
    .lazy_source()
    .sort("distance_to_polygon")
    .sink_parquet("nearest_sorted.parquet")
)
```

## Index mode

```python
# "auto" (default): build index when the cost model justifies it
# "eager": always build, "none": always scan
city_sf = SpatialFrame(cities, x_col="lon", y_col="lat", index_mode="auto")
```

## Coordinate reference system

```python
# Default is "planar": Euclidean distance in the coordinates' own units
city_sf = SpatialFrame(cities, x_col="lon", y_col="lat")

# "geographic": read lon/lat degrees and measure haversine distance in meters
city_sf = SpatialFrame(
    cities,
    x_col="lon",
    y_col="lat",
    coordinate_system="geographic",
)

# distance is now meters, so this is a true 50 km radius
near = city_sf.lazy().within_distance_of_point(
    cx=-111.7610,
    cy=34.8697,
    distance=50_000,
).collect()
```
