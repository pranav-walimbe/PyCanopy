# Planar and Geographic Distances

PyCanopy supports planar coordinates and WGS84 longitude/latitude. The default `planar` model uses
Euclidean distance in the coordinates' existing units. The `geographic` model reads coordinates as
longitude/latitude degrees and returns haversine distance in meters. PyCanopy does not store CRS
metadata or reproject coordinates, so inputs must already use the declared coordinate system.

```python
from pycanopy import SpatialFrame

planar = SpatialFrame(df, x_col="x", y_col="y")
geographic = SpatialFrame(
    df,
    x_col="lon",
    y_col="lat",
    coordinate_system="geographic",
)
```

| Planar or geographic | Planar only |
|----------------------|-------------|
| `within_distance_of_point` | `knn` |
| `within_distance_join` | `knn_join` |
| `Engine.radius_query` | `polygon_within_distance_join` |
| `point_distance` | `points_within_distance_of_polygon` |
| `distance_to_point` | `wkb_point_distance` |

Declaring geographic coordinates outside the valid longitude/latitude range emits
`PyCanopyCoordinateWarning`.

::: pycanopy.PyCanopyCoordinateWarning

## Point-to-point distances

Compute one distance for each pair of rows. Inputs may be Polars columns, NumPy arrays, or other
numeric sequences of equal length:

```python
from pycanopy import point_distance

distances_m = point_distance(
    df["start_lon"],
    df["start_lat"],
    df["end_lon"],
    df["end_lat"],
    coordinate_system="geographic",
)
```

::: pycanopy.point_distance

## Distance to one fixed point

```python
from pycanopy import distance_to_point

distances_m = distance_to_point(
    df["lon"],
    df["lat"],
    cx=-111.7610,
    cy=34.8697,
    coordinate_system="geographic",
)
```

::: pycanopy.distance_to_point

## Distances between WKB point columns

`wkb_point_distance()` compares two WKB point columns row by row using planar Euclidean distance:

```python
from pycanopy import wkb_point_distance

distances = wkb_point_distance(df["pickup_wkb"], df["dropoff_wkb"])
```

::: pycanopy.wkb_point_distance

## Decode WKB points

`wkb_points_to_xy()` returns contiguous NumPy coordinate arrays that can be used directly or added
back to a Polars DataFrame:

```python
import polars as pl
from pycanopy import wkb_points_to_xy

xs, ys = wkb_points_to_xy(df["geometry"])
points = df.with_columns(pl.Series("x", xs), pl.Series("y", ys))
```

::: pycanopy.wkb_points_to_xy
