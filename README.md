<p align="center">
  <img src="assets/pycanopy_logo3.png" alt="PyCanopy" width="800"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/pycanopy/"><img src="https://badge.fury.io/py/pycanopy.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/pycanopy/"><img src="https://img.shields.io/pypi/pyversions/pycanopy" alt="Python versions"/></a>
  <a href="https://github.com/pranav-walimbe/pycanopy/actions/workflows/CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/pranav-walimbe/pycanopy/CI.yml?branch=main&label=tests" alt="CI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

<p align="center">A geospatial query engine for optimized in-memory queries. Rust core, Python API.</p>

---

## Background

GeoPandas spatial queries (kNN, bounding-box range, point-in-polygon) default to linear scans, and even optimized baselines like the GeoPandas STRtree or scipy KDTree require manual index selection and still carry Python-level overhead.

PyCanopy is a dedicated spatial query engine that inspects your dataset at load time (size, geometry type, spatial distribution) and automatically picks the fastest index (KD-tree, R-tree, uniform grid, or brute force).

### Preliminary Performance Comparison

Measured on 1 million geometries. Naive = GeoPandas full scan, sindex = GeoPandas STRtree / scipy KDTree (pre-built). Two scenarios: uniform distribution and clustered data queried in a sparse region. †early exit via spatial histogram (index is not traversed).

**Points**

| Query | Scenario | PyCanopy | GeoPandas naive | GeoPandas sindex |
|---|---|---|---|---|
| kNN k=10 | Uniform | 0.02 ms | 44 ms **(2200x)** | 0.81 ms **(41x)** |
| kNN k=10 | Clustered, sparse | 0.01 ms | 33 ms **(3300x)** | 0.08 ms **(8x)** |
| Range 1% bbox | Uniform | 1.12 ms | 333 ms **(297x)** | 4.96 ms **(4x)** |
| Range 1% bbox | Clustered, sparse† | 0.004 ms | 319 ms **(79750x)** | 0.11 ms **(28x)** |

**Polygons**

| Query | Scenario | PyCanopy | GeoPandas naive | GeoPandas sindex |
|---|---|---|---|---|
| Range 1% bbox | Uniform | 0.59 ms | 34 ms **(58x)** | 4.52 ms **(8x)** |
| Range 1% bbox | Clustered, sparse† | 0.002 ms | 27 ms **(13500x)** | 0.04 ms **(20x)** |
| Contains | Uniform | 0.01 ms | 41 ms **(4100x)** | 0.05 ms **(5x)** |
| Contains | Clustered, sparse† | 0.005 ms | 41 ms **(8200x)** | 0.03 ms **(6x)** |

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
git clone https://github.com/pranav-walimbe/pycanopy
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
