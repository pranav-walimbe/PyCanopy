# Ops Calibration Benchmark

## What It Is

This benchmark calibrates the ten `CostFactors` values used by the native query planner in
`src/planner/cost.rs`. It writes the fitted values to
`python/pycanopy/cost_profiles/default.json`, which is shared by the Python wrapper and Rust core.

## What It Runs

- Builds Grid, KD-tree, and R-tree indexes over synthetic point and polygon datasets
- Measures range and k-nearest-neighbour probes across several sizes and selectivities
- Uses native Engine timings rather than Python wall-clock timings
- Fits each planner factor to its workload term with least-squares regression
- Writes a readable report to `assets/ops.txt`

The calibration must run against an optimized build because its output describes release-mode
performance.

## Usage

Build the release extension, run the calibration, and update the bundled profile:

```bash
make tune-engine
```

Run the same calibration without changing the profile:

```bash
uv run python -m bench.ops --dry-run
```

Useful options:

```text
--runs R         probe samples per configuration (default: 5)
--build-runs R   fresh build samples per size (default: 3)
--seed S         synthetic data and query seed (default: 42)
--dry-run        print fitted constants without updating default.json
```

## File Organization

```text
bench/ops/
├── __main__.py  # benchmark matrix, measurements, fitting, and report generation
├── utils.py     # synthetic point and polygon fixtures
└── README.md
```

Related outputs and planner code:

```text
assets/ops.txt                              # rendered calibration report
python/pycanopy/cost_profiles/default.json  # bundled fitted factors
src/planner/cost.rs                         # cost formulas using the factors
```
