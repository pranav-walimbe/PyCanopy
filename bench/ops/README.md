# Ops Calibration Benchmark

Fits the ten `CostFactors` values the query planner uses (`src/planner/cost.rs`) and writes them to
the bundled profile at `python/pycanopy/cost_profiles/default.json`.

The Python Engine wrapper reads that profile once per process and applies it to each Rust Engine;
Rust-only users and tests read the same file embedded at build time, so no constant is duplicated
in source. Hardware detection, alternate-profile selection, and a user-facing tuning API are not
part of this harness yet.

## Running

Must run against an optimized build, since the fitted constants describe release-mode timings.

```bash
# Build --release, calibrate, and replace default.json
make tune-engine

# Same measurements and fitting, leaves default.json untouched
uv run python -m bench.ops --dry-run
```

```text
--runs R         probe samples per configuration (default: 5)
--build-runs R   fresh build samples per size (default: 3)
--seed S         synthetic data and query seed (default: 42)
--dry-run        print fitted constants without updating default.json
```

The rendered report also lands in `assets/ops.txt`. The JSON holds only the ten fitted constants.

## Method

Build and probe durations come from the Engine's own Rust metrics, not Python stopwatches. Builds
use an internal calibration hook to construct Grid, KD-tree, or R-tree explicitly on a fresh
Engine, excluding SpatialFrame construction and statistics collection. Probe indexes are built and
warmed before five measured samples, and each configuration contributes its median Engine time.

Each factor is fit across its applicable configurations using the exact planner workload term and a
least-squares line through the origin: `elapsed_ns = factor * term`. The bbox scan uses a tiny box
around a known input point so histogram statistics cannot return early ahead of the brute-force
scan.

## Calibration matrix

| Dimension | Values |
|---|---|
| Point sizes | 10K, 100K, 1M |
| Polygon sizes | 10K, 50K, 100K |
| Distributions | uniform points / Grid, clustered points / KD-tree, uniform polygons / R-tree |
| Range selectivity | 1% at every size; 0.1% and 10% at the middle size |
| kNN `k` | 5 at every size; 1 and 50 at the middle size |
| Queries per probe sweep | 100 |
| Probe samples | 5, after one warm-up |
| Fresh build samples | 3 |

## Workload terms

| Constant | Operation | Term |
|---|---|---|
| `knn_scan_ns_per_item` | brute-force point kNN | `Q * N` |
| `bbox_scan_ns_per_item` | brute-force point range | `N` |
| `grid_build_ns_per_item` | Grid build | `N` |
| `kdtree_build_ns_per_item` | KD-tree build | `N * log2(N)` |
| `rtree_build_ns_per_item` | R-tree build | `N * log2(N)` |
| `grid_range_ns` | Grid range | actual output rows |
| `kdtree_range_ns` | KD-tree range | `Q * log2(N) + actual output rows` |
| `rtree_range_ns` | R-tree range | `Q * log2(N) + actual output rows` |
| `kdtree_knn_ns` | KD-tree kNN | `Q * (log2(N) + k)` |
| `rtree_knn_ns` | R-tree kNN | `Q * (log2(N) + k)` |
