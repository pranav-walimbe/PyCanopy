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

## Background

Polars has no native spatial query support. Getting bounding-box filters, k-nearest neighbours, or point-in-polygon tests on a Polars DataFrame typically means converting to GeoPandas, managing an index manually, or scanning every row in Python.

GeoPandas applies linear scans by default for containment and range tests; its STRtree requires explicit opt-in via `.sindex` and is the only available index type regardless of data distribution. KNN has no built-in path at all and requires a separate library.

PyCanopy adds a declarative lazy query layer directly on Polars DataFrames. You describe the spatial operations you want, and PyCanopy decides which index to build, in what order to run each operation, and what to hand off to Polars to execute.

---

## Installation

```bash
pip install pycanopy
```

> Pre-built wheels for Linux, macOS, and Windows. No Rust toolchain required.

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
# Append new points — visible to queries immediately, no index rebuild yet.
import numpy as np
sf.engine.append_delta(np.array([2.5]), np.array([48.9]))

# Queries probe the main index and scan the delta in parallel.
result = sf.lazy().range_query(-10.0, 35.0, 40.0, 70.0).collect()

# The buffer flushes automatically when accumulated query cost exceeds
# the estimated index rebuild cost, or when it exceeds 10% of N.
# Force a flush manually if needed.
sf.engine.flush()
```

---

## Accepted input formats

| Format | Example |
|---|---|
| numpy `(N, 2)` array | `np.array([[x, y], ...])` |
| GeoArrow PyArrow array | `pa.StructArray` or `FixedSizeList<2>` |
| geopandas `GeoSeries` | `gdf.geometry` |
| list of shapely Points or Polygons | `[Point(x, y), ...]` |
| list of `(x, y)` tuples | `[(x, y), ...]` |
| Separate coordinate sequences | `Engine.from_coords(xs, ys)` |

---

## Benchmarks

All measurements on Apple M-series, uniform random data. **Warm** = second call with cached index. **Index build** = cold minus warm (one-time cost amortised across queries). Naive baseline is GeoPandas.

### Single-query ops (N=100,000)

| Operation | Index build | Warm | GeoPandas | Speedup |
|---|---|---|---|---|
| Range query | 9 ms | 177 µs | 5.68 ms | **32×** |
| kNN k=10 | 73 ms | 22 µs | 6.35 ms | **289×** |
| Polygon contains | 127 ms | 20 µs | 6.54 ms | **326×** |
| Polygon range | 129 ms | 333 µs | 4.31 ms | **13×** |

### Batch joins (N=Q=10,000)

| Operation | Index build | Warm | Naive loop | Speedup |
|---|---|---|---|---|
| kNN join k=5 | 17 ms | 16.7 ms | 6.11 s | **366×** |
| Within-distance join | 2 ms | 67.8 ms | 1.60 s | **24×** |
| Within join (polygon) | 19 ms | 10.1 ms | 4.68 s | **463×** |

### Sample Chained lazy queries (N=100,000)

Each row is a multi-predicate chain run through the optimizer. GeoPandas applies all predicates manually with no lazy planning.

| Chain | Index build | Warm | GeoPandas | Speedup |
|---|---|---|---|---|
| `circ_scalar → range³` | 19 ms | 1.03 ms | 9.31 ms | **9×** |
| `3× scalar → range² → scalar` | 8 ms | 0.70 ms | 5.74 ms | **8×** |
| `range² → 3× scalar` (reordered) | 7 ms | 0.56 ms | 5.71 ms | **10×** |
| `circ_scalar → range → scalar → range²` | 7 ms | 0.78 ms | 8.20 ms | **11×** |

---

## How It Works

### Query Flow

```
  sf.lazy().filter(...).range_query(...).knn_join(...).collect()
                              │
                 ┌────────────▼────────────┐
                 │     SpatialOptimizer    │
                 │  • reorder ops by cost  │
                 │  • fuse spatial preds   │
                 │  • select index type    │
                 │  • spatial join order   │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │      Polars executes    │
                 │  scalar filters first   │
                 │  then spatial queries   │
                 └────────────┬────────────┘
                              │
                        pl.DataFrame
```

### Implementation Details

**Optimizer decisions**

- **Predicate Pushdown:** scalar predicates are placed before spatial ones. They cost nothing extra and shrink the row count before any index is touched.
- **Fusion:** consecutive spatial predicates on large datasets are merged into a single index build and one pass over the data.
- **Index type:** selected per query based on geometry type, data distribution, and selectivity (see Index Management below).
- **Spatial Join Order:** for symmetric joins (`within_join`, `within_distance_join`), the optimizer indexes the smaller side when it is less than half the size of the other, minimizing index build cost. `knn_join` is asymmetric and always indexes the engine side.

**Index Management**

Indexes are built lazily. Nothing is constructed at load time; stats (extent, point distribution, a 32x32 histogram) are computed eagerly and drive selection at the first query. The selected index is then cached for all subsequent queries.

| Condition | Index |
|---|---|
| N < 500, selectivity > 50%, or k/N > 10% | Brute force |
| Point range, uniform distribution | Uniform grid |
| Point range, clustered distribution | KD-tree |
| Point KNN or contains | KD-tree |
| Polygons, any query | R-tree |

All index types share the same underlying coordinate arrays with no duplication.

**Why Rust**

The hot paths need packed immutable index structures, zero-copy array slices at the Python boundary, and loop-level parallelism. C++ would require a separate FFI layer and loses the native Polars plugin integration that PyO3/Maturin provides for free.

---

## License

MIT
