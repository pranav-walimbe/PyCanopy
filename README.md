<p align="center">
  <img src="assets/pycanopy_logo2.png" alt="PyCanopy" width="800"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/pycanopy/"><img src="https://badge.fury.io/py/pycanopy.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/pycanopy/"><img src="https://img.shields.io/pypi/pyversions/pycanopy" alt="Python versions"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

A geospatial query engine for optimized in-memory queries. Rust core, Python API.

---

## Background

GeoPandas spatial queries (kNN, bounding-box range, point-in-polygon) default to linear scans, and even optimized baselines like the GeoPandas STRtree or scipy KDTree require manual index selection and still carry Python-level overhead.

PyCanopy is a dedicated spatial query engine that inspects your dataset at load time (size, geometry type, spatial distribution) and automatically picks the fastest index (KD-tree, R-tree, uniform grid, or brute force).

### Preliminary Performance Comparison

Cross-compared query speed on mock dataset with 1 million geometries

| Query | PyCanopy | GeoPandas sindex | Speedup |
|---|---|---|---|
| kNN k=10 (points) | 0.03 ms | 0.63 ms scipy KDTree | **18x** |
| Range 1% bbox (points) | 1.26 ms | 4.80 ms STRtree | **4x** |
| Range 1% bbox (polygons) | 0.60 ms | 4.48 ms STRtree | **7.5x** |
| Contains (polygons) | 0.01 ms | 0.05 ms STRtree | **3.4x** |

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
