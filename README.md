<p align="center">
  <img src="assets/pycanopy_logo3.png" alt="PyCanopy" width="800"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/pycanopy/"><img src="https://badge.fury.io/py/pycanopy.svg" alt="PyPI version"/></a>
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
- **Fan-out caching:** `collect_all` detects when multiple `SpatialLazyFrame` instances were branched from the same base by comparing plan node identity. The shared prefix is emitted once with a Polars `.cache()` barrier, and all branch suffixes execute in a single `pl.collect_all()` call.

**Index Management**

Indexes are built lazily. Nothing is constructed at load time; stats (extent, point distribution, a 32x32 histogram) are computed eagerly and drive selection at the first query. The selected index is then cached for all subsequent queries.

| Condition | Index |
|---|---|
| N < 500 or selectivity > 50% | Brute force |
| Point KNN | KD-tree |
| Point range + uniform distribution | Uniform grid |
| Point range + clustered distribution | KD-tree |
| Polygons (any query) | R-tree |

All index types share the same underlying coordinate arrays with no duplication.

**Why Rust**

The hot paths need packed immutable index structures, zero-copy array slices at the Python boundary, and loop-level parallelism. C++ would require a separate FFI layer and loses the native Polars plugin integration that PyO3/Maturin provides for free.

---

## License

MIT
