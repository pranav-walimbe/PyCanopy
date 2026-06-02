<p align="center">
  <img src="assets/pycanopy_logo3.png" alt="PyCanopy" width="800"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/pycanopy/"><img src="https://badge.fury.io/py/pycanopy.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/pycanopy/"><img src="https://img.shields.io/pypi/pyversions/pycanopy" alt="Python versions"/></a>
  <a href="https://github.com/pranavwalimbe/pycanopy/actions/workflows/CI.yml"><img src="https://github.com/pranavwalimbe/pycanopy/actions/workflows/CI.yml/badge.svg" alt="CI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

<p align="center">A geospatial query engine for optimized in-memory queries. Rust core, Python API.</p>

---

## Background

GeoPandas spatial queries (kNN, bounding-box range, point-in-polygon) default to linear scans, and even optimized baselines like the GeoPandas STRtree or scipy KDTree require manual index selection and still carry Python-level overhead.

PyCanopy is a dedicated spatial query engine that inspects your dataset at load time (size, geometry type, spatial distribution) and automatically picks the fastest index (KD-tree, R-tree, uniform grid, or brute force).

### Preliminary Performance Comparison

Measured on 1 million geometries against GeoPandas STRtree / scipy KDTree. Three scenarios used: uniform distribution, clustered distribution queried in a dense region, and clustered distribution queried in a sparse region.

**Points**

| Query | Scenario | PyCanopy | GeoPandas sindex | Speedup |
|---|---|---|---|---|
| kNN k=10 | Uniform | 0.03 ms | 0.57 ms scipy KDTree | **17x** |
| kNN k=10 | Clustered, dense | 0.03 ms | 0.13 ms scipy KDTree | **4x** |
| kNN k=10 | Clustered, sparse | 0.01 ms | 0.18 ms scipy KDTree | **14x** |
| Range 1% bbox | Uniform | 1.27 ms | 6.50 ms STRtree | **5x** |
| Range 1% bbox | Clustered, dense | 1.41 ms | 25.61 ms STRtree | **18x** |
| Range 1% bbox | Clustered, sparse | 0.01 ms | 0.12 ms STRtree | **24x** |

**Polygons**

| Query | Scenario | PyCanopy | GeoPandas sindex | Speedup |
|---|---|---|---|---|
| Range 1% bbox | Uniform | 0.63 ms | 4.67 ms STRtree | **7x** |
| Range 1% bbox | Clustered, dense | 8.40 ms | 22.73 ms STRtree | **3x** |
| Range 1% bbox | Clustered, sparse | 0.00 ms | 0.09 ms STRtree | **37x** |
| Contains | Uniform | 0.01 ms | 0.04 ms STRtree | **4x** |
| Contains | Clustered, dense | 0.04 ms | 0.08 ms STRtree | **2x** |
| Contains | Clustered, sparse | 0.01 ms | 0.03 ms STRtree | **5x** |

---

## Installation

```bash
pip install pycanopy
```

> Pre-built wheels for Linux, macOS, and Windows. No Rust toolchain required.

---

## Quick start

```python
import numpy as np
from pycanopy import Engine

# Point dataset
coords = np.random.uniform(0, 100, size=(500_000, 2))
engine = Engine(coords)

nearest = engine.knn(x=42.0, y=37.0, k=10)
in_box  = engine.range_query(min_x=10.0, min_y=10.0, max_x=50.0, max_y=50.0)

# Polygon dataset
from shapely.geometry import box
polygons = [box(i, 0, i + 0.9, 0.9) for i in range(500_000)]
poly_engine = Engine.from_polygons(polygons)

intersecting = poly_engine.range_query(0.0, 0.0, 10.0, 1.0)
containing   = poly_engine.contains(x=5.5, y=0.5)
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

## Development setup

Requires Python ≥ 3.9 and a Rust toolchain ([rustup.rs](https://rustup.rs)).

```bash
git clone https://github.com/pranavwalimbe/pycanopy
cd pycanopy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" maturin shapely geopandas
maturin develop
pytest
cargo test
```

---

## License

MIT
