# Ops Calibration Benchmark

Measures the 9 `CostFactors` ns/op constants used by the query planner's cost model
(`src/planner/cost.rs`, `src/planner/calibration.rs`).

## Method

- Each dataset size generates synthetic data with numpy (uniform or clustered points) or
  `shapely.box` (polygons), then wraps it in a `SpatialFrame`/`Engine`.
- Build cost times only `engine.build_index()` on a fresh engine. Probe cost times the repeated queries against a pre-built index.
- The constant is `time / workload_term`, where the term is read directly off the cost
  model formula in `cost.rs` (e.g. `Q * N` for `knn_scan_ns_per_item`).
- This ratio is computed at multiple dataset sizes and we return the median as the suggested value. 


## What it measures

| Constant | Op timed | Dataset | Term |
|---|---|---|---|
| `knn_scan_ns_per_item` | brute-force kNN, no index | uniform points | `Q * N` |
| `bbox_scan_ns_per_item` | brute-force range, no index | uniform points | `Q * N` |
| `grid_build_ns_per_item` | build | uniform points | `N` |
| `kdtree_build_ns_per_item` | build | clustered points | `N * log2(N)` |
| `rtree_build_ns_per_item` | build | polygons | `N * log2(N)` |
| `grid_range_ns` | range probe | uniform points | materialized row count |
| `kdtree_range_ns` | range probe | clustered points | `Q * log2(N)` + materialized row count |
| `rtree_range_ns` | range probe | polygons | `Q * log2(N)` + materialized row count |
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
    knn_scan_ns_per_item:        3.07,
    bbox_scan_ns_per_item:       0.59,

Points
    grid_build_ns_per_item:      10.33,
    kdtree_build_ns_per_item:    0.47,
    grid_range_ns:               34.40,
    kdtree_range_ns:             17.56,
    kdtree_knn_ns:               24.92,

Polygons
    rtree_build_ns_per_item:     60.08,
    rtree_range_ns:              24.69,
    rtree_knn_ns:                132.17,

elapsed: 2.9 s   peak RSS: 281.2 MiB
```