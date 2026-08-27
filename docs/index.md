# PyCanopy

A declarative spatial query layer for Polars. Rust core, Python API.

## What is PyCanopy

PyCanopy adds spatial filters, nearest-neighbour queries, joins, and grouped aggregations to Polars
DataFrames. Its lazy query planner combines Polars expressions with native Rust spatial kernels and
uses workload statistics to choose an execution strategy and spatial index.

## Why PyCanopy

| Capability | PyCanopy | GeoPandas | DuckDB | SedonaDB | Spatial Polars |
|:--|:--:|:--:|:--:|:--:|:--:|
| Uses Polars DataFrames directly | ✓ | ✗ | ✗ | ✗ | ✓ |
| Spatial-aware query planning | ✓ | ✗ | ✓ | ✓ | ✗ |
| Automatically accelerates spatial joins with an index | ✓ | ✓ | ✓ | ✓ | ✗ |
| Explicit cost-based choice between scanning and building an index | ✓ | ✗ | ✗ | ✗ | ✗ |
| Selects among multiple spatial index types by workload | ✓ | ✗ | ✗ | ✗ | ✗ |

## Benchmarks

[Apache SpatialBench](https://github.com/apache/sedona-spatialbench) is the industry-standard single-node spatial query benchmark, maintained by the Apache Sedona project. Results below are from a single `m7i.2xlarge` (8 vCPU, 32 GB), the same instance type used in the published baseline.

PyCanopy is fastest on 11/24 testcases and lands within 5% of the fastest time on 14/24 testcases (there is some variance among benchmark runs).

**SF1** (~6M trips)

![Apache SpatialBench SF1](assets/spatialbench_sf1_auto.png)

**SF10** (~60M trips)

![Apache SpatialBench SF10](assets/spatialbench_sf10_auto.png)

Full results tables with per-query times are on the [Benchmarks](benchmarks.md) page.

## Data sources

| Source | Entry point |
|:-------|:------------|
| Point coordinate columns in a Polars `DataFrame` | `SpatialFrame(df, x_col="x", y_col="y")` |
| Point WKB in a Polars `DataFrame` | `SpatialFrame.from_wkb_points(df, "geometry")` |
| Polygon or MultiPolygon WKB in a Polars `DataFrame` | `SpatialFrame.from_wkb_polygons(df, "geometry")` |
| Shapely or GeoArrow polygon geometry | `SpatialFrame.from_polygons(df, "geometry")` |
| Point or polygon WKB in a Polars `LazyFrame` | `SpatialFrame.from_lazy(lf, "geometry", "point")` |
| Local, cloud, or GeoParquet data | `SpatialFrame.scan_parquet(path)` |

The lower-level `Engine` also accepts NumPy arrays, coordinate sequences, GeoArrow arrays,
GeoPandas geometry, and Shapely geometry directly.
