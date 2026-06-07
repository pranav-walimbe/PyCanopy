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
> Up to **155x** on range queries · up to **1,949x** on kNN · up to **1,521x** on polygon contains · up to **8,522x** on within joins · [Full benchmarks](#benchmarks)

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

Polars has no native spatial query support. Getting bounding-box filters, k-nearest neighbours, or point-in-polygon tests require us to rely on other solutions (e.g. GeoPandas, DuckDB, etc). The alternatives each leave something on the table.

GeoPandas applies linear scans by default; its STRtree requires explicit opt-in via `.sindex` and is the only available index type regardless of data distribution.

GeoPolars (currently alpha) is a Polars plugin with Polars-native expressions and lazy evaluation, and it ships an R*-tree index, but the index is manually managed, only one type is available, and Polars' general-purpose optimizer applies no spatial query planning / ordering.

DuckDB spatial has a mature R-tree, but it requires the user to interface with it using SQL (less intuitive than Polars), the index must be created explicitly, one index type is available (not optimal for all queries), and also does not perform spatial query planning / ordering.

PyCanopy adds a declarative lazy query layer directly on Polars DataFrames. You describe the operations you want, and PyCanopy decides which index to build, in what order to run each operation, and delegates non-spatial operations to Polars. It is designed for in-memory workloads at the moment.

|                              | PyCanopy      | GeoPandas        | GeoPolars (alpha)  | DuckDB spatial      |
|:-----------------------------|:-------------:|:----------------:|:------------------:|:-------------------:|
| Works natively in Polars     | ✓             | ✗                | ✓                  | ✗ (SQL + convert)   |
| Lazy / declarative API       | ✓             | ✗                | via Polars         | SQL only            |
| Auto index selection         | ✓             | ✗ (STR only)     | ✗ (R-tree, manual) | ✗ (R-tree, opt-in)  |
| KNN join built-in            | ✓             | ✗                | ✗                  | ✗ (O(N) scan)       |
| Spatial operation ordering   | ✓             | ✗                | ✗                  | ✗                   |
| Spatial Predicate fusion     | ✓             | ✗                | ✗                  | ✗                   |
| Zero-copy Python boundary    | ✓             | ✗                | ✓                  | ✗                   |

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

### Inspecting the optimizer plan

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

Follows Polars' FROM-chain convention: bottom = runs first, top = outermost result. In the optimized plan, FILTER appears below RANGE_QUERY — the scalar filter runs first on raw data, and RANGE_QUERY receives the already-filtered subset. `explain(optimized=False)` shows declaration order for comparison.

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

Apple M-series used for benchmarking. **Warm** = cached index, second call. **Index build** = one-time cost, amortised across queries. Uniform distribution; clustered note below.

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

### Chained lazy queries (N = 100,000, uniform)

The optimizer reorders scalars before spatial ops regardless of declared order, and fuses consecutive wide spatial predicates into one index pass.

| Chain                                        | Optimizer action | Index build |    Warm | GeoPandas | Speedup  |
|:---------------------------------------------|:-----------------|------------:|--------:|----------:|---------:|
| circ\_scalar → range³                        | scalar first     |      2.5 ms | 0.19 ms |    9.2 ms |  **50x** |
| range² → 3× scalar (spatial declared first)  | scalars first    |      1.0 ms | 0.23 ms |    6.0 ms |  **26x** |
| range⁴ at 10% selectivity                   | fused            |      1.0 ms | 0.92 ms |     13 ms |  **14x** |
| wide\_scalar (95%) → tight\_range (1%)       | scalar first     |      4.1 ms | 0.30 ms |    3.1 ms |  **11x** |
| circ\_scalar + diag\_scalar → kNN k=50       | scalar first     |       15 ms | 1.25 ms |    3.6 ms |   **3x** |

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
