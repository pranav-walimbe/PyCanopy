# Ops Calibration Benchmark

Measures the 9 `CostFactors` ns/op constants used by the query planner's cost model
(`src/planner/cost.rs`, `src/planner/calibration.rs`).

## Method

- Each dataset size generates synthetic data with numpy (uniform or clustered points) or
  `shapely.box` (polygons), then wraps it in a `SpatialFrame`/`Engine`.
- Build cost times only `engine.build_index()` on a fresh engine. Probe cost times the repeated queries against a pre-built index.
- The constant is `time / workload_term`, where the term is read directly off the cost
  model formula in `cost.rs` (e.g. `Q * N` for `scan_ns_per_item`).
- This ratio is computed at multiple dataset sizes and we return the median as the suggested value. 


## What it measures

| Constant | Op timed | Dataset | Term |
|---|---|---|---|
| `scan_ns_per_item` | brute-force kNN, no index | uniform points | `Q * N` |
| `grid_build_ns_per_item` | build | uniform points | `N` |
| `kdtree_build_ns_per_item` | build | clustered points | `N * log2(N)` |
| `rtree_build_ns_per_item` | build | polygons | `N * log2(N)` |
| `grid_range_ns` | range probe | uniform points | true hit total |
| `kdtree_range_ns` | range probe | clustered points | `Q * log2(N)` + true hit total |
| `rtree_range_ns` | range probe | polygons | `Q * log2(N)` + true hit total |
| `kdtree_knn_ns` | kNN probe | clustered points | `Q * (log2(N) + k)` |
| `rtree_knn_ns` | kNN probe | polygons | `Q * (log2(N) + k)` |

## Running

```
uv run python -m bench.ops
```

Flags:

```
--runs R   timing repetitions per measurement, the minimum is taken (default: 3)
--seed S   RNG seed for data and query generation (default: 42)
```

## Example Output

```
Suggested CostFactors (copy into src/planner/calibration.rs):

Brute Force
    scan_ns_per_item:            100.80,

Points
    grid_build_ns_per_item:      84.74,
    kdtree_build_ns_per_item:    4.36,
    grid_range_ns:               211.38,
    kdtree_range_ns:             81.19,
    kdtree_knn_ns:               148.71,

Polygons
    rtree_build_ns_per_item:     74.17,
    rtree_range_ns:              176.16,
    rtree_knn_ns:                1299.75,

elapsed: 20.3 s   peak RSS: 268.9 MiB
```