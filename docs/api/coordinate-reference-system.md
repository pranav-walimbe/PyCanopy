# Coordinate Reference System

A `SpatialFrame` measures **planar** distance by default: plain Euclidean distance on `x_col`/`y_col` in your coordinates' own units. Passing `coordinate_system="geographic"` reads `x_col`/`y_col` as WGS84 lon/lat degrees and measures great-circle (haversine) distance in meters. The setting is fixed at construction and defaults to `planar`.

```python
from pycanopy import SpatialFrame

planar = SpatialFrame(df, x_col="x", y_col="y")  # Euclidean, coordinate units
geographic = SpatialFrame(df, x_col="lon", y_col="lat", coordinate_system="geographic")  # haversine meters
```

| Planar or geographic | Planar only |
|----------------------|-------------|
| `within_distance_of_point` | `knn` |
| `within_distance_join` | `knn_join` |
| `Engine.radius_query` | `polygon_within_distance_join` |
| `point_distance` | `points_within_distance_of_polygon` |
| `distance_to_point` | `wkb_point_distance` |

## Distance utilities

These module-level helpers compute distances over raw coordinate arrays without a frame. Each accepts anything that coerces to a float64 array, including Polars columns, and runs in one parallel pass. `point_distance` and `distance_to_point` take a `coordinate_system` of `"planar"` or `"geographic"`.

```python
from pycanopy import point_distance, distance_to_point

d = point_distance(df["lon_a"], df["lat_a"], df["lon_b"], df["lat_b"], coordinate_system="geographic")
d = distance_to_point(df["lon"], df["lat"], -111.7610, 34.8697, coordinate_system="geographic")
```

::: pycanopy.point_distance

::: pycanopy.distance_to_point

::: pycanopy.wkb_point_distance

::: pycanopy.wkb_points_to_xy
