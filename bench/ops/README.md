# Ops Calibration Benchmark

Calibrates the ten `CostFactors` values used by the query planner (`src/planner/cost.rs`).
The canonical bundled profile is `python/pycanopy/cost_profiles/default.json`. Normal runs replace
that file atomically; `--dry-run` performs the same measurements and fitting without changing it.

The Python Engine wrapper reads the bundled profile once per process and applies it to each Rust
Engine. Rust-only users and tests use the same file embedded at build time, so the values are not
duplicated in source code. Hardware detection, alternate-profile selection, and a user-facing
tuning API are not part of this harness yet.

## Method

- Build and probe durations come from the Engine's Rust metrics, not Python stopwatches.
- Builds use an internal calibration hook to explicitly construct Grid, KD-tree, or R-tree on a
  fresh Engine. SpatialFrame construction and statistics collection are excluded.
- Probe indexes are built and warmed before five measured samples. Each configuration contributes
  its median Engine time.
- Each factor is fit across its applicable configurations using the exact planner workload term
  and a least-squares line through the origin: `elapsed_ns = factor * term`.
- The bbox scan uses a tiny box around a known input point, ensuring histogram statistics cannot
  return early before the brute-force scan.

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

## Running

Build the optimized extension, calibrate, and update `default.json`:

```bash
make tune-engine
```

Preview without changing `default.json`:

```bash
uv run python -m bench.ops --dry-run
```

Options:

```text
--runs R         probe samples per configuration (default: 5)
--build-runs R   fresh build samples per size (default: 3)
--seed S         synthetic data and query seed (default: 42)
--dry-run        print results without replacing default.json
```

The rendered report is also written to `assets/ops.txt`. The loadable JSON contains only the ten
fitted constants.
