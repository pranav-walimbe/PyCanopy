# Operations Reference

## Point datasets

| Operation | Call | Returns |
|:----------|:-----|:--------|
| Range query | `.range_query(min_x, min_y, max_x, max_y)` | Rows inside the bounding box |
| Range filter | `.range_filter(min_x, min_y, max_x, max_y)` | New SpatialFrame with only rows inside the bounding box |
| k-nearest neighbours | `.knn(x, y, k)` | The `k` rows nearest a point |
| kNN join | `.knn_join(df, x_col, y_col, k)` | The `k` nearest rows for each query point |
| Within-distance join | `.within_distance_join(df, x_col, y_col, distance)` | Rows within `distance` of each query point |
| Convex-hull area | `SpatialFrame.convex_hull_area(xs, ys)` | Area of the convex hull of a point set |
| Batch convex-hull area | `Engine.group_convex_hull_areas(xs_series, ys_series)` | Convex hull area for each group |
| WKB point distance | `wkb_point_distance(series_a, series_b)` | Euclidean distance between two WKB point columns |

## Polygon datasets

| Operation | Call | Returns |
|:----------|:-----|:--------|
| Point in polygon | `.contains(x, y)` | Polygons that contain the point |
| MBR range | `.range_query(min_x, min_y, max_x, max_y)` | Polygons whose bounding box meets the query box |
| Range filter | `.range_filter(min_x, min_y, max_x, max_y)` | New SpatialFrame with only polygons intersecting the bounding box |
| Within join | `.within_join(df, x_col, y_col)` | Polygons that contain each query point |
| Point-to-polygon distance join | `.polygon_within_distance_join(df, x_col, y_col, distance)` | Polygons within `distance` of each query point |
| Point-to-polygon kNN join | `.polygon_knn_join(df, x_col, y_col, k)` | The `k` nearest polygons for each query point |
| Intersects self-join | `.intersects_pairs(key_col=None)` | Intersecting polygon pairs with overlap area and IoU |
| Area | `.polygon_areas()` | Area of each polygon |
| Points near a polygon | `.points_within_distance_of_polygon(polygon, distance)` | Points within `distance` of a single polygon |

## Reductions and streaming

Compose with any join operation.

| Operation | Call | Returns |
|:----------|:-----|:--------|
| Aggregate-join | `.group_by(keys).agg(pc.agg.count/sum/mean/min/max(...))` | One row per group, reduced over the join with no pair frame |
| Projection pushdown | `.select(cols)` | Narrows both join sides before the gather |
| Stream in batches | `.collect_batched()` | An iterator of result morsels, bounded memory |
| Stream to Parquet | `.sink_parquet(path)` | Writes the result to disk in bounded memory |
| Out-of-core pipeline | `.lazy_source()` | A Polars source that fuses join + sort + sink, spilling to disk |

## Examples

### Range query and kNN

```python
import polars as pl
from pycanopy import SpatialFrame

df = pl.read_parquet("cities.parquet")
sf = SpatialFrame(df, x_col="lon", y_col="lat")

# Bounding-box filter combined with a scalar predicate
result = (
    sf.lazy()
    .filter(pl.col("population") > 100_000)
    .range_query(min_x=-10.0, min_y=35.0, max_x=40.0, max_y=70.0)
    .collect()
)

# k-nearest neighbours
nearest = sf.lazy().knn(x=2.35, y=48.85, k=5).collect()
```

### Inspecting the plan

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

### Aggregate over a join

```python
import pycanopy as pc

stats = (
    zones.lazy()
    .within_join(trips, x_col="lon", y_col="lat")
    .group_by(["zone_id", "zone_name"])
    .agg(trip_count=pc.agg.count(), avg_fare=pc.agg.mean("fare"))
)
```

### Out-of-core joins

```python
# Stream straight to Parquet
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

### Polygon dataset

```python
from shapely.geometry import box
from pycanopy import SpatialFrame

polygons = [box(i, 0, i + 0.9, 0.9) for i in range(100_000)]
df = pl.DataFrame({"id": list(range(100_000)), "geom": polygons})
sf = SpatialFrame.from_polygons(df, geometry_col="geom")

containing = sf.lazy().contains(x=5.5, y=0.5).collect()
intersecting = sf.lazy().range_query(0.0, 0.0, 10.0, 1.0).collect()
```

### Delta buffer (live updates)

```python
import numpy as np

# Append new points visible to queries immediately, no index rebuild yet
sf.engine.append_delta(np.array([2.5]), np.array([48.9]))

result = sf.lazy().range_query(-10.0, 35.0, 40.0, 70.0).collect()

# Force a flush manually if needed
sf.engine.flush()
```

### Index mode

```python
# "auto" (default) builds when the cost model justifies it
# "eager" always builds, "none" always scans
sf = SpatialFrame(df, x_col="lon", y_col="lat", index_mode="auto")
```

### Branching from a shared base

```python
from pycanopy import SpatialFrame, SpatialLazyFrame

base = sf.lazy().filter(pl.col("population") > 100_000).range_query(-10.0, 35.0, 40.0, 70.0)

major = base.filter(pl.col("population") > 1_000_000)
minor = base.filter(pl.col("population") <= 1_000_000)

# collect_all detects the shared prefix and executes both branches in a single pass
results = SpatialLazyFrame.collect_all([major, minor])
df_major, df_minor = results
```
