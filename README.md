<p align="center">
  <img src="assets/pycanopy_logo3.png" alt="PyCanopy" width="800"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/pycanopy/"><img src="https://img.shields.io/pypi/v/pycanopy" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/pycanopy/"><img src="https://img.shields.io/pypi/pyversions/pycanopy" alt="Python versions"/></a>
  <a href="https://github.com/pranav-walimbe/pycanopy/actions/workflows/CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/pranav-walimbe/pycanopy/CI.yml?branch=main&label=tests" alt="CI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

<p align="center">A spatial query layer for Polars. Rust core, Python API.</p>

---

> [!NOTE]
> Up to **155x** on range queries · up to **1,949x** on kNN · up to **1,521x** on polygon contains · up to **8,522x** on within joins (vs GeoPandas) · [Full benchmarks](#benchmarks)

---

## Installation

```bash
pip install pycanopy
```

> Pre-built wheels for Linux, macOS, and Windows. No Rust toolchain required.

```python
import polars as pl
from pycanopy import SpatialFrame

sf = SpatialFrame(pl.read_parquet("cities.parquet"), x_col="lon", y_col="lat")
result = sf.lazy().filter(pl.col("population") > 100_000).range_query(-10.0, 35.0, 40.0, 70.0).collect()
```

---

## Why PyCanopy

Polars has no native spatial support. The standard alternatives each require a tradeoff:

- **GeoPandas** applies linear scans by default; STRtree requires explicit `.sindex` opt-in and is the only index type available
- **DuckDB spatial** has a mature R-tree and good performance, but requires leaving Polars for SQL and explicit index creation

PyCanopy stays native to Polars and adds a query optimizer on top. The optimizer decides execution order, decides whether an index is worth building (and which kind) from a cost model, and fuses consecutive spatial predicates where possible.

|                              | PyCanopy      | GeoPandas        | GeoPolars          | DuckDB spatial      |
|:-----------------------------|:-------------:|:----------------:|:------------------:|:-------------------:|
| Works natively in Polars     | ✓             | ✗                | ✓                  | ✗ (SQL + convert)   |
| Lazy / declarative API       | ✓             | ✗                | via Polars         | SQL only            |
| Auto index selection         | ✓             | ✗ (STR only)     | ✗ (R-tree, manual) | ✗ (R-tree, opt-in)  |
| Cost-based index vs scan     | ✓             | ✗                | ✗                  | ✗                   |
| KNN join built-in            | ✓             | ✗                | ✗                  | ✗ (O(N) scan)       |
| Spatial operation ordering   | ✓             | ✗                | ✗                  | ✗                   |
| Spatial predicate fusion     | ✓             | ✗                | ✗                  | ✗                   |
| Live point ingestion         | ✓             | ✗                | ✗                  | ✗                   |

---

## Operations

Spatial operations chain off `sf.lazy()` and mix freely with Polars' own `.filter(expr)`. The optimizer orders the whole chain before anything runs.

**Point datasets**

| Operation              | Call                                                  | Returns                                          |
|:-----------------------|:------------------------------------------------------|:-------------------------------------------------|
| Range query            | `.range_query(min_x, min_y, max_x, max_y)`            | Rows inside the bounding box                      |
| k-nearest neighbours   | `.knn(x, y, k)`                                        | The `k` rows nearest a point                      |
| kNN join               | `.knn_join(df, x_col, y_col, k)`                       | The `k` nearest rows for each query point         |
| Within-distance join   | `.within_distance_join(df, x_col, y_col, distance)`   | Rows within `distance` of each query point        |
| Convex-hull area       | `SpatialFrame.convex_hull_area(xs, ys)`               | Area of the convex hull of a point set            |

**Polygon datasets**

| Operation                     | Call                                                          | Returns                                                 |
|:------------------------------|:-------------------------------------------------------------|:--------------------------------------------------------|
| Point in polygon              | `.contains(x, y)`                                            | Polygons that contain the point                         |
| MBR range                     | `.range_query(min_x, min_y, max_x, max_y)`                  | Polygons whose bounding box meets the query box         |
| Within join                   | `.within_join(df, x_col, y_col)`                            | Polygons that contain each query point                  |
| Point→polygon distance join   | `.polygon_within_distance_join(df, x_col, y_col, distance)` | Polygons within `distance` of each query point          |
| Point→polygon kNN join        | `.polygon_knn_join(df, x_col, y_col, k)`                    | The `k` nearest polygons for each query point           |
| Intersects self-join          | `.intersects_pairs()`                                       | Intersecting polygon pairs with overlap area and IoU    |
| Area                          | `.polygon_areas()`                                          | Area of each polygon                                    |
| Points near a polygon         | `.points_within_distance_of_polygon(polygon, distance)`     | Points within `distance` of a single polygon            |

Joins and aggregations that return tables (`.intersects_pairs`, `.polygon_areas`, `.points_within_distance_of_polygon`) are called directly on the `SpatialFrame`; the filtering operations chain off `.lazy()`.

---

## Usage

### Point dataset: range and KNN

```python
import polars as pl
from pycanopy import SpatialFrame

df = pl.read_parquet("cities.parquet")
sf = SpatialFrame(df, x_col="lon", y_col="lat")

# Bounding-box filter combined with a scalar predicate.
# Optimizer places the scalar filter first, then runs the range query
# on the reduced row set.
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
# Declare ops in any order — explain() shows what the optimizer will actually run.
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

print(lf.explain(optimized=False))
# FILTER [(col("population")) > (dyn int: 100000)]
# FROM
#   RANGE_QUERY [(-10, 35) → (40, 70)]
#   FROM
#     DF [N=100,000]
```

The optimizer flipped the declaration order: the scalar filter runs first on all rows, then the spatial query runs on the smaller survivor set. `explain(optimized=False)` shows declaration order for comparison. Follows Polars' FROM-chain convention — bottom runs first, top is the outermost result.

<details>
<summary>More examples — KNN join, polygon contains, within-distance join, branching, delta buffer</summary>

### Chaining multiple spatial predicates

```python
# Two range predicates are fused into a single index build on large datasets.
result = (
    sf.lazy()
    .range_query(0.0, 0.0, 50.0, 50.0)
    .range_query(10.0, 10.0, 40.0, 40.0)
    .collect()
)
```

### KNN join

```python
query_df = pl.DataFrame({"qx": [2.35, 13.4], "qy": [48.85, 52.5]})

# For each row in query_df, find the 3 nearest rows in sf.
result = sf.lazy().knn_join(query_df, x_col="qx", y_col="qy", k=3).collect()
```

### Polygon dataset: contains and range

```python
from shapely.geometry import box
from pycanopy import SpatialFrame

polygons = [box(i, 0, i + 0.9, 0.9) for i in range(100_000)]
df = pl.DataFrame({"id": list(range(100_000)), "geom": polygons})
sf = SpatialFrame.from_polygons(df, geometry_col="geom")

# Which polygons contain this point?
containing = sf.lazy().contains(x=5.5, y=0.5).collect()

# Which polygon MBRs intersect this bbox?
intersecting = sf.lazy().range_query(0.0, 0.0, 10.0, 1.0).collect()
```

### Polygon holes

```python
from shapely.geometry import Polygon

# Interior rings (holes) are fully supported.
outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
hole  = [(2, 2), (8, 2),  (8, 8),  (2, 8)]
donut = Polygon(outer, [hole])

sf = SpatialFrame.from_polygons(pl.DataFrame({"id": [0], "geom": [donut]}), geometry_col="geom")

# Point inside the hole is NOT contained.
sf.lazy().contains(x=5.0, y=5.0).collect()   # empty

# Point outside the hole but inside the outer ring IS contained.
sf.lazy().contains(x=1.0, y=1.0).collect()   # returns the polygon row
```

### Within join

```python
# For each query point, find which polygons in sf contain it.
query_df = pl.DataFrame({"qx": [5.5, 12.3], "qy": [0.5, 0.5]})
result = sf.lazy().within_join(query_df, x_col="qx", y_col="qy").collect()
```

### Within-distance join

```python
# For each query point, find all sf points within 50 km.
query_df = pl.DataFrame({"qx": [2.35, 13.4], "qy": [48.85, 52.5]})
result = sf.lazy().within_distance_join(query_df, x_col="qx", y_col="qy", distance=50.0).collect()
```

### Point-to-polygon joins

```python
# (polygon SpatialFrame) For each query point, the polygons within a distance
# of it — measured to the polygon boundary, zero when the point is inside.
query_df = pl.DataFrame({"qx": [5.5, 12.3], "qy": [0.5, 0.5]})
near = sf.lazy().polygon_within_distance_join(query_df, x_col="qx", y_col="qy", distance=2.0).collect()

# For each query point, its k nearest polygons (adds a distance_to_polygon column).
nearest = sf.lazy().polygon_knn_join(query_df, x_col="qx", y_col="qy", k=3).collect()
```

### Polygon aggregations

```python
# Area of every polygon (appends an 'area' column).
areas = sf.polygon_areas()

# All intersecting polygon pairs, with overlap area and IoU.
overlaps = sf.intersects_pairs()

# (point SpatialFrame) rows whose point lies within a distance of one polygon.
from shapely.geometry import box
pts = point_sf.points_within_distance_of_polygon(box(0.0, 0.0, 1.0, 1.0), distance=0.5)
```

### Branching from a shared base

```python
from pycanopy import SpatialFrame, SpatialLazyFrame

# Expensive filter applied once; two queries branch from the result.
base = sf.lazy().filter(pl.col("population") > 100_000).range_query(-10.0, 35.0, 40.0, 70.0)

major = base.filter(pl.col("population") > 1_000_000)
minor = base.filter(pl.col("population") <= 1_000_000)

# collect_all detects the shared prefix, caches it in Polars,
# and executes both branches in a single pass.
results = SpatialLazyFrame.collect_all([major, minor])
df_major, df_minor = results
```

### Live updates via delta buffer

```python
# Append new points -- visible to queries immediately, no index rebuild yet.
import numpy as np
sf.engine.append_delta(np.array([2.5]), np.array([48.9]))

# Queries probe the main index and scan the delta in parallel.
result = sf.lazy().range_query(-10.0, 35.0, 40.0, 70.0).collect()

# The buffer flushes automatically when accumulated query cost exceeds
# the estimated index rebuild cost, or when it exceeds 10% of N.
# Force a flush manually if needed.
sf.engine.flush()
```

</details>

---

## Benchmarks

Apple M-series used for benchmarking. **Warm** = cached index, second call. **Index build** = one-time cost, amortised across queries. Naive baseline is GeoPandas. Datasets are mocked from random uniform distribution.

### Single-query operations

| Operation              |       N | Index build |    Warm |   Naive | Speedup    | Idx mem |
|:-----------------------|--------:|------------:|--------:|--------:|-----------:|--------:|
| Range query (points)   | 100,000 |      1.3 ms |   29 µs |  4.4 ms |   **155x** | 783 KB  |
| kNN k=10               | 100,000 |      9.3 ms |    3 µs |  5.4 ms | **1,949x** | 1.9 MB  |
| Polygon contains       | 100,000 |      6.2 ms |    5 µs |  7.0 ms | **1,521x** | 3.7 MB  |
| Polygon range          | 100,000 |      5.6 ms |    8 µs |  3.3 ms |   **391x** | 3.7 MB  |
| kNN join k=5           |  10,000 |      7.3 ms |  2.1 ms |   5.4 s | **2,601x** | 180 KB  |
| Within-distance join   |  10,000 |      0.5 ms | 12.6 ms |   1.3 s |   **102x** | —       |
| Within join (polygons) |  10,000 |      1.6 ms | 0.52 ms |   4.4 s | **8,522x** | 354 KB  |

---

## How It Works

PyCanopy plans a query in two layers, then hands the result to Polars to run.

### Query flow

```
  sf.lazy().filter(...).range_query(...).knn_join(...).collect()
                            |
            +---------------+----------------+
            |   Logical plan (whole chain)   |
            |   order ops . fuse predicates  |
            |   . pick join side . EXPR/IO   |
            +---------------+----------------+
                            |
            +---------------+----------------+
            |   Access path (per operation)  |
            |   index or scan, and which —   |
            |   a cost model decides         |
            +---------------+----------------+
                            |
            +---------------+----------------+
            |   Polars runs the emitted ops  |
            +---------------+----------------+
                            |
                      pl.DataFrame
```

### Logical planning

Decisions about the shape of the query, made over the whole chain before any data is touched:

- **Predicate pushdown** — scalar filters are placed before spatial ones and sorted cheapest-first (each filter's cost is estimated by walking its Polars expression tree). They cost little and shrink the row count before any index is touched.
- **Fusion** — consecutive spatial predicates on large datasets are merged into a single index build and one pass over the data.
- **Join side** — for symmetric joins (`within_join`, `within_distance_join`) the planner indexes the smaller side when it is under half the size of the other. `knn_join` is asymmetric and always indexes the engine side.
- **Execution path** — a very selective filter slices the prebuilt index directly (IO path); otherwise the work is emitted as Polars expressions that filter first and build a small index on the survivors (EXPR path).

### Cost model — index or scan?

For each spatial operation the planner decides whether to use an index at all. Building one isn't free — a tree costs about `N log N` up front — so it only pays off if the index is queried enough times. The planner compares two estimates, where `N` is the dataset size and `Q` is the number of query points:

```
  scan    =  Q × N                              every row, for every query point
  index   =  N log N  (build, once)  +  Q × log N  (probe each query point)
```

Building wins once `Q` climbs past roughly `log N`: a one-off lookup (`Q = 1`) just scans, while a join with thousands of probes builds the index and reuses it across all of them. **Selectivity** — the fraction of rows a predicate keeps, estimated from the data extent and a 32×32 histogram — sharpens the call: a predicate that keeps most rows skips the index outright, since a tree that prunes nothing is slower than a plain scan.

`index_mode` (set per frame) chooses how that estimate is applied:

- **`eager`** (default) — always build the selected index.
- **`auto`** — build only when the estimate beats a scan for this query's `Q`.
- **`none`** — always scan, never build.

### Index management

Indexes are built lazily — nothing at load time. The dataset stats (extent, point distribution, a 32×32 histogram) are computed once up front, drive the choice at the first query, and the chosen index is cached for every query after. The build policy is the per-frame `index_mode` described above (`SpatialFrame(..., index_mode=...)`, default `eager`).

When a non-brute index is built, its kind comes from:

| Condition                                     | Index        |
|:----------------------------------------------|:-------------|
| N < 500, selectivity > 50%, or k/N > 10%     | Brute force  |
| Point range, uniform distribution             | Uniform grid |
| Point range, clustered distribution           | KD-tree      |
| Point KNN or contains                         | KD-tree      |
| Polygons, any query                           | R-tree       |

All index types share the same underlying coordinate arrays with no duplication.

### Why Rust

The hot paths need packed immutable index structures, zero-copy array slices at the Python boundary, and loop-level parallelism. C++ would require a separate FFI layer and loses the native Polars plugin integration that PyO3/Maturin provides for free.

---

## Accepted input formats

| Format                             | Example                                    |
|:-----------------------------------|:-------------------------------------------|
| numpy `(N, 2)` array               | `np.array([[x, y], ...])`                  |
| GeoArrow PyArrow array             | `pa.StructArray` or `FixedSizeList<2>`     |
| geopandas `GeoSeries`              | `gdf.geometry`                             |
| list of shapely Points or Polygons | `[Point(x, y), ...]`                       |
| list of `(x, y)` tuples            | `[(x, y), ...]`                            |
| Separate coordinate sequences      | `Engine.from_coords(xs, ys)`               |

---

## License

MIT
