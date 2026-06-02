<p align="center">
  <img src="assets/pycanopy_logo2.png" alt="PyCanopy" width="800"/>
</p>

# PyCanopy

<p align="center">
  <a href="https://pypi.org/project/pycanopy/"><img src="https://badge.fury.io/py/pycanopy.svg" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/pycanopy/"><img src="https://img.shields.io/pypi/pyversions/pycanopy" alt="Python versions"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

A geospatial query engine with automatic index selection. Rust core, Python API.

> **Early development.** The API is stable for point and polygon datasets but the project is actively evolving. Contributions and feedback welcome.

---

## Background

GeoPandas is excellent for data manipulation, but spatial queries (finding the k nearest points, all geometries within a bounding box, which polygons contain a location) default to full O(N) scans. For large datasets this gets slow fast.

PyCanopy is a dedicated spatial query engine that sits alongside your GeoPandas workflow. It inspects your dataset at load time (size, geometry type, spatial distribution) and automatically picks the fastest index (KD-tree, R-tree, uniform grid, or brute force) without you having to think about it. The core is written in Rust and coordinates cross the Python/Rust boundary as zero-copy numpy buffers.

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
