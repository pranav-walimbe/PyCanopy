<p align="center">
  <img src="assets/pycanopy_logo2.png" alt="PyCanopy" width="400"/>
</p>

# PyCanopy

A geospatial query engine with automatic index selection. Written in Rust, callable through python.

PyCanopy accepts point datasets and answers spatial queries (k-nearest neighbours, bounding-box range, point-in-polygon) by automatically choosing the fastest index — R-tree, KD-tree, uniform grid, or brute force — based on dataset size, geometry kind, and spatial distribution.

## Installation

```bash
pip install pycanopy
```

> **Note:** Pre-built wheels are provided for Linux, macOS, and Windows. No Rust toolchain required.

## Quick start

```python
import numpy as np
from pycanopy import Engine

coords = np.random.uniform(0, 100, size=(50_000, 2))
engine = Engine(coords)

indices = engine.knn(x=42.0, y=37.0, k=10)
indices = engine.range_query(min_x=10.0, min_y=10.0, max_x=50.0, max_y=50.0)
indices = engine.contains(x=25.0, y=25.0)
```

## Accepted input formats

| Format | Example |
|---|---|
| numpy `(N, 2)` array | `np.array([[x, y], ...])` |
| GeoArrow PyArrow array | `pa.StructArray` or `FixedSizeList<2>` |
| geopandas `GeoSeries` | `gdf.geometry` |
| list of shapely Points | `[Point(x, y), ...]` |
| list of `(x, y)` tuples | `[(x, y), ...]` |
| Separate coordinate lists | `Engine.from_coords(xs, ys)` |

## How index selection works

| Condition | Index chosen |
|---|---|
| N < 500 or selectivity > 50% | Brute force |
| Points + kNN | KD-tree |
| Points + uniform + range | Uniform grid |
| Points + clustered + range | KD-tree |
| Polygons or mixed geometries | R-tree |

## Development setup

Requires Python >= 3.9 and a Rust toolchain ([rustup.rs](https://rustup.rs)).

```bash
git clone https://github.com/pranavwalimbe/pycanopy
cd pycanopy
uv sync --group dev
uv run maturin develop
uv run pytest
cargo test
```

## License

MIT
