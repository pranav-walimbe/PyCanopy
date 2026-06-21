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
> **State of the art on [Apache SpatialBench](https://github.com/apache/sedona-spatialbench):** fastest on 7 of 12 queries at both SF1 and SF10, beating SedonaDB, DuckDB, GeoPandas, and Spatial Polars on every heavy spatial join.

Apache SpatialBench is the standard single-node spatial-analytics benchmark: 12 queries over millions of trips and zones. PyCanopy runs the whole suite in Polars-like syntax, never dropping to SQL or a separate engine.

<p align="center">
  <img src="assets/spatialbench_sf1_auto.png" alt="PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF1" width="100%"/>
</p>
<p align="center"><sub>Apache SpatialBench SF1 · lower is better · bars past the cap truncated with their value · TIMEOUT / ERROR annotated</sub></p>

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

Every spatial option for a Polars user asks you to give something up:

- **GeoPandas** is eager and pandas-based. Its one index (STRtree) is opt-in, and a join larger than memory simply fails.
- **DuckDB spatial** is fast and out-of-core, but you leave Polars for SQL and create the R-tree index by hand.
- **SedonaDB** is a capable spatial engine, but it is a separate SQL engine rather than a Polars-native API.
- **Spatial Polars** provides polars-like syntax but lacks sophisticated optimizations + indexing.

PyCanopy's principle is to stay inside Polars and add a real query planner. You declare spatial ops in any order. It reorders them, fuses adjacent predicates, pushes projections into joins, and uses a cost model to decide per query whether to build an index at all (and which kind).

How the options compare:

|                                                   | PyCanopy | GeoPandas      | DuckDB spatial | SedonaDB | Spatial Polars |
|:--------------------------------------------------|:--------:|:--------------:|:--------------:|:--------:|:--------------:|
| Polars-native, no SQL or conversion               | ✓        | ✗              | ✗ (SQL)        | ✗ (SQL)  | ✓              |
| Spatial query planner (reorder, fuse, push)       | ✓        | ✗              | ✗              | ✓        | ✗              |
| Cost-based index vs scan, per query               | ✓        | ✗              | ✗              | ✗        | ✗              |
| Multi-index types, auto-selected per query        | ✓        | ✗ (STRtree)    | ✗ (R-tree)     | ✗ (R-tree) | ✗ (STRtree)  |
| kNN join                                          | ✓        | ✓ (nearest)    | ✗              | ✓        | ✓ (centroid)   |
| Aggregate-join without materialising pairs        | ✓        | ✗              | ✗              | ✗        | ✗              |
| Larger-than-RAM joins                             | ✓        | ✗              | ✓              | ✓        | ✗              |

---

## Operations

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
| Point-to-polygon distance join   | `.polygon_within_distance_join(df, x_col, y_col, distance)` | Polygons within `distance` of each query point          |
| Point-to-polygon kNN join        | `.polygon_knn_join(df, x_col, y_col, k)`                    | The `k` nearest polygons for each query point           |
| Intersects self-join          | `.intersects_pairs()`                                       | Intersecting polygon pairs with overlap area and IoU    |
| Area                          | `.polygon_areas()`                                          | Area of each polygon                                    |
| Points near a polygon         | `.points_within_distance_of_polygon(polygon, distance)`     | Points within `distance` of a single polygon            |

**Reductions and streaming** (compose with any join)

| Operation              | Call                                                       | Returns                                                       |
|:-----------------------|:-----------------------------------------------------------|:-------------------------------------------------------------|
| Aggregate-join         | `.group_by(keys).agg(pc.agg.count/sum/mean/min/max(...))`  | One row per group, reduced over the join with no pair frame   |
| Projection pushdown    | `.select(cols)`                                            | Narrows both join sides before the gather                     |
| Stream in batches      | `.collect_batched()`                                       | An iterator of result morsels, bounded memory                |
| Stream to Parquet      | `.sink_parquet(path)`                                      | Writes the result to disk in bounded memory                  |
| Out-of-core pipeline   | `.lazy_source()`                                           | A Polars source that fuses join + sort + sink, spilling to disk |

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
# Declare ops in any order. explain() shows what the optimizer will actually run.
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

The optimizer flipped the declaration order. The scalar filter runs first on all rows, then the spatial query runs on the smaller survivor set. Plans follow Polars' FROM-chain convention, so the bottom runs first and the top is the final result.

### Aggregate over a join

```python
import pycanopy as pc

# Count trips per zone and average their fare, reduced over a streamed
# point-in-polygon join. The full pair frame is never materialised: each
# morsel reduces to per-group partials that combine into the final result.
stats = (
    zones.lazy()
    .within_join(trips, x_col="lon", y_col="lat")
    .group_by(["zone_id", "zone_name"])
    .agg(trip_count=pc.agg.count(), avg_fare=pc.agg.mean("fare"))
)
```

### Out-of-core joins (larger than RAM)

```python
# A join whose result exceeds memory: stream it straight to Parquet,
# bounded to one morsel at a time.
sf.lazy().polygon_knn_join(trips, "lon", "lat", k=5).sink_parquet("nearest.parquet")

# Or fuse the join with a sort and sink into a single spilling Polars
# pipeline, so even an ordered result larger than RAM never materialises.
(
    sf.lazy()
    .polygon_knn_join(trips, "lon", "lat", k=5)
    .select(["trip_id", "building_id", "distance_to_polygon"])
    .lazy_source()
    .sort("distance_to_polygon")
    .sink_parquet("nearest_sorted.parquet")
)
```

<details>
<summary>More examples: point and polygon joins, aggregations, branching, delta buffer, index modes</summary>

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
# of it. Distance is to the polygon boundary, and zero when the point is inside.
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

### Convex-hull area

```python
import numpy as np

# Area of the convex hull of a standalone point set (no frame needed).
area = SpatialFrame.convex_hull_area(np.array([0.0, 1.0, 0.5]), np.array([0.0, 0.0, 1.0]))
```

### Index mode

```python
# Fixed per frame. "auto" lets the cost model choose index vs scan per query;
# "none" always scans; "eager" (default) always builds the selected index.
sf = SpatialFrame(df, x_col="lon", y_col="lat", index_mode="auto")
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

### Apache SpatialBench

Run on a single `m7i.2xlarge` (8 vCPU, 32 GB), the same hardware used by [Apache SpatialBench](https://github.com/apache/sedona-spatialbench). PyCanopy is measured live with `index_mode="auto"`; SedonaDB 0.3.0, DuckDB 1.5.4, GeoPandas 1.1.3, and Spatial Polars 0.3.0 are snapshot numbers from [run #152](https://github.com/apache/sedona-spatialbench/actions/runs/27864209643) (2026-06-20).

**SF1** (~6M trips). PyCanopy wins 7 of 12, taking all heavy spatial joins (q5, q7–q12). DuckDB wins q1–q4; SedonaDB wins q6.

<p align="center">
  <img src="assets/spatialbench_sf1_auto.png" alt="PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF1" width="100%"/>
</p>
<p align="center"><sub>Apache SpatialBench SF1 · lower is better · linear axis, bars past the cap truncated with their value · TIMEOUT / ERROR annotated</sub></p>

**SF10** (~60M trips). PyCanopy wins 7 of 12 and is the only engine to finish all 12. SedonaDB errors on q12, DuckDB OOMs on q7/q11 and times out on q10/q12, GeoPandas and Spatial Polars OOM on most queries. q12 returns a result larger than 32 GB; PyCanopy streams the join and spills the sort to disk.

<p align="center">
  <img src="assets/spatialbench_sf10_auto.png" alt="PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF10" width="100%"/>
</p>
<p align="center"><sub>Apache SpatialBench SF10 · lower is better · linear axis, bars past the cap truncated with their value · TIMEOUT / ERROR annotated</sub></p>

### Per-operation vs GeoPandas

Apple M-series. **Cold** = fresh engine, index build included. **Warm** = cached index, second call. **GeoPandas** is the naive baseline (no spatial index). Uniform random data.

| Operation                          |       N |    Cold |    Warm | GeoPandas |   Speedup |
|:-----------------------------------|--------:|--------:|--------:|----------:|----------:|
| Range query (points)               | 100,000 |  2.7 ms |   31 µs |    5.5 ms |   **177×** |
| kNN k=10                           | 100,000 |  9.5 ms |    5 µs |    7.4 ms | **1,452×** |
| Contains (polygons)                | 100,000 |  5.1 ms |    3 µs |    5.2 ms | **1,922×** |
| Range (polygons)                   | 100,000 |  5.5 ms |    7 µs |    3.2 ms |   **470×** |
| kNN join k=5                       |  10,000 |  8.8 ms |  2.8 ms |    5.6 s  | **1,958×** |
| Within-distance join               |  10,000 | 15.7 ms | 13.9 ms |    3.7 s  |   **266×** |
| Within join (polygons)             |   5,000 |  3.3 ms | 0.73 ms |    1.2 s  | **1,663×** |
| Point→polygon kNN join k=5         |   5,000 |  7.4 ms |  4.4 ms |    6.3 s  | **1,414×** |
| Point→polygon within-distance join |   5,000 |  6.9 ms |  6.5 ms |    5.5 s  |   **858×** |
| Intersects self-join               |   5,000 |  3.0 ms | 0.89 ms |   0.84 s  |   **943×** |

---

## How It Works

PyCanopy plans a query in two layers, then hands the result to Polars to run.

### Query flow

```
  sf.lazy().filter(...).range_query(...).knn_join(...).collect()
                            |
            +---------------+----------------+
            |   Logical plan (whole chain)   |
            |   order ops, fuse predicates,  |
            |   pick join side, EXPR vs IO   |
            +---------------+----------------+
                            |
            +---------------+----------------+
            |   Access path (per operation)  |
            |   index or scan, and which     |
            |   kind: a cost model decides   |
            +---------------+----------------+
                            |
            +---------------+----------------+
            |   Polars runs the emitted ops  |
            +---------------+----------------+
                            |
                      pl.DataFrame
```

### Logical planning

Decisions about the whole chain, made before any data is touched:

- **Predicate pushdown:** scalar filters run first, cheapest first (cost from the Polars expression tree), shrinking the row count cheaply.
- **Fusion:** consecutive spatial predicates merge into one index build and pass.
- **Join side:** symmetric joins (`within_join`, `within_distance_join`) index the smaller side. `knn_join` indexes the engine side.
- **Projection pushdown:** a terminal `.select()` pushes into the join, gathering only the requested columns, not the full width.
- **Execution path:** very selective filters slice the prebuilt index directly (IO path). Otherwise filters run first and a small index builds on the survivors (EXPR path).

### Cost model: index or scan?

Building an index costs about `N log N`, so it only pays off if queried enough times. For each operation the planner compares two estimates (`N` is the dataset size, `Q` the number of query points):

```
  scan   =  Q * N                          every row, for every query point
  index  =  N log N  (build once)  +  Q * log N   (probe per query point)
```

Building wins once `Q` passes roughly `log N`. A one-off lookup scans; a join with many probes builds the index and reuses it. Selectivity refines this: if a predicate keeps most rows, the planner skips the index, since a tree that prunes nothing loses to a plain scan.

`index_mode`, set per frame, picks how the estimate is used:

- **`eager`** (default): always build the selected index.
- **`auto`**: build only when the estimate beats a scan for this `Q`.
- **`none`**: always scan.

### Index management

Indexes build lazily, never at load time. Dataset stats (extent, distribution, a 32x32 histogram) are computed once up front and drive the first query's choice, after which the index is cached for all later queries. When a non-brute index is built, its kind comes from:

| Condition                                     | Index        |
|:----------------------------------------------|:-------------|
| N < 500, selectivity > 50%, or k/N > 10%     | Brute force  |
| Point range, uniform distribution             | Uniform grid |
| Point range, clustered distribution           | KD-tree      |
| Point KNN or contains                         | KD-tree      |
| Polygons, any query                           | R-tree       |

All index types share the same coordinate arrays with no duplication.

### Why Rust

The hot paths need packed immutable index structures, zero-copy array slices at the Python boundary, and loop-level parallelism. C++ would require a separate FFI layer and loses the native Polars plugin integration that PyO3/Maturin provides for free.

---

## Accepted input formats

| Format                             | Example                                    |
|:-----------------------------------|:-------------------------------------------|
| numpy `(N, 2)` array               | `np.array([[x, y], ...])`                  |
| GeoArrow PyArrow array             | `pa.StructArray` or `FixedSizeList<2>`     |
| geopandas `GeoSeries`              | `gdf.geometry`                             |
| shapely Points / Polygons / MultiPolygons | `[Point(x, y), ...]`                |
| list of `(x, y)` tuples            | `[(x, y), ...]`                            |
| Separate coordinate sequences      | `Engine.from_coords(xs, ys)`               |
| WKB point column (Binary)          | `SpatialFrame.from_wkb_points(df, "geom")` |
| WKB polygon column (Binary)        | `SpatialFrame.from_wkb_polygons(df, "geom")` |

---

## Acknowledgements

Some works that inspired this project:

- [Polars](https://github.com/pola-rs/polars): a columnar DataFrame engine that PyCanopy builds on
- [geo-index](https://github.com/georust/geo-index): provides packed, immutable, zero-copy KD-tree and R-tree structures used
- [Spatial Polars](https://github.com/ATL2001/spatial_polars): an earlier effort to bring spatial functionality to Polars
- [Apache Sedona](https://sedona.apache.org): state-of-the-art spatial SQL engine + benchmark for evals

---

## License

MIT
