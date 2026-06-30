# Getting Started

PyCanopy is a declarative spatial query layer for Polars with a Rust core.

## Installation

```bash
pip install pycanopy
```

Pre-built wheels for Linux, macOS, and Windows. No Rust toolchain required.

## Quick start

```python
import polars as pl
from pycanopy import SpatialFrame

sf = SpatialFrame(pl.read_parquet("cities.parquet"), x_col="lon", y_col="lat")

result = (
    sf.lazy()
    .filter(pl.col("population") > 100_000)
    .range_query(-10.0, 35.0, 40.0, 70.0)
    .collect()
)
```

## Why PyCanopy

The only spatial engine with a Polars-native API, cost-model-driven index selection, and a full spatial query planner.

|  | PyCanopy | GeoPandas | DuckDB | SedonaDB | Spatial Polars |
|:--|:--------:|:---------:|:------:|:--------:|:--------------:|
| Polars-native, no SQL or conversion             | ✓ | ✗ | ✗ (SQL) | ✗ (SQL) | ✓ |
| Spatial query planner (reorder, fuse, pushdown) | ✓ | ✗ | ✗ | ✓ (SQL) | ✗ |
| Index vs scan decided by cost model             | ✓ | ✗ | ✗ | ✗ | ✗ |
| Adaptive index (KD-tree / R-tree / grid)        | ✓ | ✗ STRtree | ✗ R-tree | ✗ Quadtree | ✗ STRtree / KDTree |

## Accepted input formats

| Format | Example |
|:-------|:--------|
| numpy `(N, 2)` array | `np.array([[x, y], ...])` |
| GeoArrow PyArrow array | `pa.StructArray` or `FixedSizeList<2>` |
| geopandas `GeoSeries` | `gdf.geometry` |
| shapely Points / Polygons / MultiPolygons | `[Point(x, y), ...]` |
| list of `(x, y)` tuples | `[(x, y), ...]` |
| Separate coordinate sequences | `Engine.from_coords(xs, ys)` |
| WKB point column (Binary) | `SpatialFrame.from_wkb_points(df, "geom")` |
| WKB polygon column (Binary) | `SpatialFrame.from_wkb_polygons(df, "geom")` |
