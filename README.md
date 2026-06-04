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

## How It Works

PyCanopy sits as a planning and execution layer between your code and Polars. You declare what you want spatially; PyCanopy decides how to run it, then hands a Polars LazyFrame chain back for Polars to execute and return a native DataFrame.

```
  declare                      plan                       hand to Polars
  ───────────    ──────────────────────────────────    ──────────────────────────

  SpatialFrame   SpatialLazyFrame   SpatialOptimizer   scalar filter exprs
  ─────────────  ────────────────   ────────────────   (Polars runs these first,
  xs, ys         .filter()          reorder by cost     cheaply, reducing rows)
  stats cached   .range_query()     fuse predicates
  index lazy     .knn_join()        pick index type    spatial map_batches exprs
                 ...                pick EXPR or IO     (is_elementwise=False tells
                                                         Polars to treat these as
                                    SpatialExecutor      column-level barriers and
                                    emits Polars chain   not reorder past them)
                                    dispatches Rust
                                    batch joins         join results via rayon,
                                                        concat'd into LazyFrame

                                                        pl.DataFrame  <── .collect()
```

The optimizer makes four decisions before any data moves:

**Ordering.** Scalar Polars predicates are placed before spatial operations. Running them first costs nothing extra and shrinks the row count before any index is touched.

**Fusion.** Consecutive spatial predicates on a large dataset are merged so the index is built once and all predicates are evaluated in a single pass over the data.

**Index type.** Chosen per query from KD-tree (point KNN), uniform grid (uniform-distribution range), R-tree (polygons), or brute force (small N or full-scan selectivity).

**Execution path.** The EXPR path emits spatial filters as Polars `map_batches` expressions. Polars sees `is_elementwise=False` and treats each expression as a barrier, running scalar predicates above it first and passing the reduced row set to the spatial closure. The IO path is used for tight spatial filters: the pre-built Engine index is queried directly, the DataFrame is sliced to the small candidate set, and scalar filters run on that slice.

Scalar predicates, sorting, projection, and `collect()` are always handled by Polars. PyCanopy does not re-implement any of that.

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

## Index selection

PyCanopy inspects the dataset at load time and picks automatically:

| Condition | Index |
|---|---|
| N < 500 or selectivity > 50% | Brute force |
| Points + kNN | KD-tree |
| Points + uniform distribution + range | Uniform grid |
| Points + clustered distribution + range | KD-tree |
| Polygons or mixed geometries | R-tree |

---

## License

MIT
