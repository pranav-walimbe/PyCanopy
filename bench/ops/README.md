# Ops Calibration Benchmark

Measures warm probe time per `(index_kind, query_kind)` at multiple N values and derives
the `per_ns` constants used in `CostFactors` (`src/planner/calibration.rs`).

## What it measures

| Constant | Index | Query kind | Dataset |
|----------|-------|------------|---------|
| `kdtree_knn_ns` | KDTree | kNN | clustered points |
| `kdtree_range_ns` | KDTree | range | clustered points |
| `rtree_knn_ns` | RTree | kNN | polygons |
| `rtree_range_ns` | RTree | range | polygons |
| `grid_range_ns` | Grid | range | uniform points |
| `scan_ns_per_item` | BruteForce | kNN | uniform points |
| `build_ns_per_item` | KDTree | — | clustered points |

Brute-force scan is only measured up to `--brute-max-n` (default 100,000) since it grows
as `Q×N` and becomes impractical at large N.

All data is generated over `[0, 1]²`. Range queries use a `0.1 × 0.1` bbox (selectivity ≈ 0.01).
kNN uses k=5. Each timing is the median of `--runs` repetitions.

## How constants are derived

Each measured warm time `T` (ms) is inverted through the cost model formula:

| Constant | Formula |
|----------|---------|
| `kd_knn_ns`, `rt_knn_ns` | `T × 1e6 / (Q × (log₂N + k))` |
| `kd_range_ns`, `rt_range_ns` | `T × 1e6 / (Q × (log₂N + sel×N))` |
| `grid_range_ns` | `T × 1e6 / (Q × sel×N)` |
| `scan_ns_per_item` | `T × 1e6 / (Q × N)` |
| `build_ns_per_item` | `T × 1e6 / (N × log₂N)` |

## Running

```
uv run python -m bench.ops
```

Optional flags:

```
--sizes N [N ...]     dataset sizes to sweep (default: 10000 100000 500000 1000000)
--queries Q           queries per timing call (default: 500)
--runs R              repetitions per measurement, median taken (default: 3)
--brute-max-n N       skip brute-force above this N (default: 100000)
```
