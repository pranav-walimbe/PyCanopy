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

df = pl.read_parquet("cities.parquet")
sf = SpatialFrame(df, x_col="lon", y_col="lat")

# Bounding-box query combined with a scalar predicate
result = (
    sf.lazy()
    .filter(pl.col("population") > 100_000)
    .range_query(min_x=-10.0, min_y=35.0, max_x=40.0, max_y=70.0)
    .collect()
)

# k-nearest neighbours
nearest = sf.lazy().knn(x=2.35, y=48.85, k=5).collect()

# kNN join: for each query point, the 3 nearest dataset rows
query_df = pl.DataFrame({"qx": [2.35, 13.4], "qy": [48.85, 52.5]})
result = sf.lazy().knn_join(query_df, x_col="qx", y_col="qy", k=3).collect()
```

## Polygon dataset

```python
from shapely.geometry import box
from pycanopy import SpatialFrame

polygons = [box(i, 0, i + 0.9, 0.9) for i in range(100_000)]
df = pl.DataFrame({"id": list(range(100_000)), "geom": polygons})
sf = SpatialFrame.from_polygons(df, geometry_col="geom")

# Which polygons contain this point?
containing = sf.lazy().contains(x=5.5, y=0.5).collect()

# For each query point, find the polygons that contain it
query_df = pl.DataFrame({"qx": [5.5, 12.3], "qy": [0.5, 0.5]})
result = sf.lazy().within_join(query_df, x_col="qx", y_col="qy").collect()
```

## Aggregate over a join

```python
import pycanopy as pc

# Count trips per zone and average fare, reduced over a streamed join
# The full pair frame is never materialised
stats = (
    zones.lazy()
    .within_join(trips, x_col="lon", y_col="lat")
    .group_by(["zone_id", "zone_name"])
    .agg(trip_count=pc.agg.count(), avg_fare=pc.agg.mean("fare"))
)
```

## Inspecting the plan

```python
lf = (
    sf.lazy()
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

The optimizer flipped the declaration order: scalar filter runs first, spatial query runs on the smaller survivor set.

## Streaming large results

```python
# Stream to Parquet in bounded memory
sf.lazy().polygon_knn_join(trips, "lon", "lat", k=5).sink_parquet("nearest.parquet")

# Fuse join + sort + sink into a single spilling Polars pipeline
(
    sf.lazy()
    .polygon_knn_join(trips, "lon", "lat", k=5)
    .select(["trip_id", "building_id", "distance_to_polygon"])
    .lazy_source()
    .sort("distance_to_polygon")
    .sink_parquet("nearest_sorted.parquet")
)
```

## Index mode

```python
# "auto" (default): build index when the cost model justifies it
# "eager": always build, "none": always scan
sf = SpatialFrame(df, x_col="lon", y_col="lat", index_mode="auto")
```

## Live updates via delta buffer

```python
import numpy as np

# Append new points, visible to queries immediately with no index rebuild
sf.engine.append_delta(np.array([2.5]), np.array([48.9]))
result = sf.lazy().range_query(-10.0, 35.0, 40.0, 70.0).collect()

# Force a flush manually if needed
sf.engine.flush()
```
