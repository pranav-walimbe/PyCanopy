# PyCanopy

A declarative spatial query layer for Polars. Rust core, Python API.

## What is PyCanopy

PyCanopy brings fast spatial queries (range, kNN, joins, polygon containment) into the Polars ecosystem without leaving Python. You declare the operations you want and the engine decides the most efficient way to execute the query and whether it should build one of its available spatial indices.

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

PyCanopy is fastest on 11/24 testcases, including one tie, and lands within 5% of the fastest time on 14/24 testcases (there is some variance among benchmark runs).

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
