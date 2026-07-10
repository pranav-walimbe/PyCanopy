# PyCanopy

A declarative spatial query layer for Polars. Rust core, Python API.

## What is PyCanopy

PyCanopy brings spatial queries (range, kNN, joins, polygon containment) into the Polars ecosystem without leaving Python. You declare operations in any order; the query planner reorders, fuses, and pushes them down before execution. The index type (KD-tree, R-tree, grid, or brute force) is selected automatically by a cost model calibrated to your hardware.

## Why PyCanopy

|  | PyCanopy | GeoPandas | DuckDB | SedonaDB | Spatial Polars |
|:--|:--------:|:---------:|:------:|:--------:|:--------------:|
| Polars-native, no SQL or conversion             | ✓ | ✗ | ✗ (SQL) | ✗ (SQL) | ✓ |
| Spatial query planner (reorder, fuse, pushdown) | ✓ | ✗ | ✗ | ✓ (SQL) | ✗ |
| Index vs scan decided by cost model             | ✓ | ✗ | ✗ | ✗ | ✗ |
| Adaptive index (KD-tree / R-tree / grid)        | ✓ | ✗ | ✗ | ✗ | ✗ |

## Benchmarks

[Apache SpatialBench](https://github.com/apache/sedona-spatialbench) is the industry-standard single-node spatial query benchmark, maintained by the Apache Sedona project. Results below are from a single `m7i.2xlarge` (8 vCPU, 32 GB), the same instance type used in the published baseline.

PyCanopy wins a total of 11/24 testcases and lands within 5% of winning 14/24 testcases (there is some variance among benchmark runs).

**SF1** (~6M trips)

![Apache SpatialBench SF1](assets/spatialbench_sf1_auto.png)

**SF10** (~60M trips)

![Apache SpatialBench SF10](assets/spatialbench_sf10_auto.png)

Full results tables with per-query times are on the [Benchmarks](benchmarks.md) page.

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
