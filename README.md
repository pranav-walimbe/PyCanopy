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

> [!TIP]
> Range query **173x** faster than GeoPandas · KNN join **1,429x** · Polygon within join **6,901x** · [Benchmark details](#benchmarks)

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

Polars has no native spatial query support. Getting bounding-box filters, k-nearest neighbours, or point-in-polygon tests typically means converting to GeoPandas, managing an index manually, or scanning every row in Python. GeoPandas applies linear scans by default for containment and range tests; its STRtree requires explicit opt-in via `.sindex` and is the only available index type regardless of data distribution. KNN has no built-in path at all.

PyCanopy adds a declarative lazy query layer directly on Polars DataFrames. You describe the operations you want, and PyCanopy decides which index to build, in what order to run each operation, and what to hand off to Polars to execute.

|                             | PyCanopy      | GeoPandas        | Manual STRtree |
|:----------------------------|:-------------:|:----------------:|:--------------:|
| Works natively in Polars    | ✓             | ✗                | ✗              |
| Lazy / declarative API      | ✓             | ✗                | ✗              |
| Auto index selection        | ✓             | ✗ (STR only)     | ✗              |
| KNN join built-in           | ✓             | ✗                | ✗              |
| Delta buffer (live append)  | ✓             | ✗                | ✗              |

- **Lazy planning** -- declare ops, the optimizer reorders and fuses them before any index is built
- **Auto index selection** -- KD-tree, R-tree, uniform grid, or brute force chosen per query
- **Native Polars** -- results are `pl.DataFrame`, no round-trip through GeoPandas
- **Rust hot paths** -- zero-copy at the Python boundary, loop-level parallelism via Rayon
- **Delta buffer** -- append new points and query immediately without rebuilding the index

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

<details>
<summary>More examples -- KNN join, polygon contains, within-distance join, branching, delta buffer</summary>

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

All measurements on Apple M-series, uniform random data. **Warm** = second call with cached index. **Index build** = cold minus warm (one-time cost amortised across queries). Naive baseline is GeoPandas.

### Single-query ops (N=100,000)

| Operation          |   Index build |    Warm | GeoPandas |   Speedup |
|:-------------------|--------------:|--------:|----------:|----------:|
| Range query        |        1.2 ms |   30 µs |    5.1 ms |  **173x** |
| kNN k=10           |        9.9 ms |    7 µs |    6.3 ms |  **881x** |
| Polygon contains   |        5.8 ms |    5 µs |    7.2 ms | **1382x** |
| Polygon range      |        5.4 ms |    9 µs |    3.5 ms |  **407x** |

### Batch joins (N=Q=10,000)

| Operation             |   Index build |    Warm | Naive loop |   Speedup |
|:----------------------|--------------:|--------:|-----------:|----------:|
| kNN join k=5          |        7.6 ms |  4.1 ms |     5.83 s | **1429x** |
| Within-distance join  |        3.7 ms | 13.7 ms |     1.48 s |  **108x** |
| Within join (polygon) |        2.1 ms |  0.7 ms |     4.99 s | **6901x** |

### Chained lazy queries (N=100,000)

Each row is a multi-predicate chain run through the optimizer. GeoPandas applies all predicates manually with no lazy planning.

| Chain                                        |   Index build |    Warm | GeoPandas |  Speedup |
|:---------------------------------------------|--------------:|--------:|----------:|---------:|
| `circ_scalar + range³`                       |        2.4 ms | 0.19 ms |    9.4 ms |  **49x** |
| `3x scalar + range² + scalar`                |        0.9 ms | 0.22 ms |    6.0 ms |  **28x** |
| `range² + 3x scalar` (reordered)             |        0.9 ms | 0.20 ms |    5.7 ms |  **29x** |
| `circ_scalar + range + scalar + range²`      |        0.8 ms | 0.17 ms |    8.0 ms |  **47x** |
| `range⁴ 10% (fusion)`                       |        1.1 ms | 0.93 ms |   13.1 ms |  **14x** |

---

## How It Works

### Query flow

```
  sf.lazy().filter(...).range_query(...).knn_join(...).collect()
                            |
               +------------+------------+
               |     SpatialOptimizer    |
               |  * reorder ops by cost  |
               |  * fuse spatial preds   |
               |  * select index type    |
               |  * spatial join order   |
               +------------+------------+
                            |
               +------------+------------+
               |      Polars executes    |
               |  scalar filters first   |
               |  then spatial queries   |
               +------------+------------+
                            |
                      pl.DataFrame
```

### Optimizer decisions

- **Predicate pushdown:** scalar predicates are placed before spatial ones and sorted cheapest-first using AST cost estimation. They cost nothing extra and shrink the row count before any index is touched.
- **Fusion:** consecutive spatial predicates on large datasets are merged into a single index build and one pass over the data.
- **Index type:** selected per query based on geometry type, data distribution, and selectivity.
- **Join order:** for symmetric joins (`within_join`, `within_distance_join`), the optimizer indexes the smaller side when it is less than half the size of the other. `knn_join` is asymmetric and always indexes the engine side.

### Index management

Indexes are built lazily. Nothing is constructed at load time; stats (extent, point distribution, a 32x32 histogram) are computed eagerly and drive selection at the first query. The selected index is then cached for all subsequent queries.

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
