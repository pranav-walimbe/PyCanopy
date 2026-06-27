# Ops Benchmark

Per-primitive benchmark for PyCanopy's batch spatial join operations. Each operation is
measured cold (SpatialFrame construction + index build + query) and warm (index cached,
query only) against the best available indexed Python baseline.

## Operations

| Operation | N | Competitor |
|-----------|---|-----------|
| knn\_join k=5 | 100 K × 100 K points | cKDTree (scipy) |
| within\_distance\_join | 100 K × 100 K points | STRtree (shapely) |
| polygon\_knn\_join k=5 | 100 K × 100 K polygons | cKDTree on centroids (scipy) |
| within\_join | 100 K × 100 K polygons | STRtree (shapely) |
| polygon\_within\_distance\_join | 100 K × 100 K polygons | STRtree (shapely) |
| intersects self-join | 100 K polygons | STRtree (shapely) |

Data is uniformly random over `[0, 1]²`. Polygons are axis-aligned boxes of size 0.005.

GeoPandas STRtree does not support batch kNN with k > 1, so kNN joins use scipy cKDTree
(the standard go-to for that case). All other joins use STRtree.

## Columns

| Column | Meaning |
|--------|---------|
| `cold ms` | PyCanopy: SpatialFrame construction + index build + query |
| `warm ms` | PyCanopy: query only (index cached) |
| `gp index` | Competitor index used for this operation |
| `gp cold ms` | Competitor: index construction + query |
| `gp ms` | Competitor: query only (index pre-built) |
| `speedup` | `gp ms / warm ms` (warm vs warm) |

Both sides are timed identically: cold includes index construction, warm uses the pre-built index.

## Running

```
uv run python -m bench.ops
```

Results are written to `assets/ops.txt`.
