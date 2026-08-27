# SpatialBench

## What It Is

This harness runs the [Apache SpatialBench](https://github.com/apache/sedona-spatialbench)
single-node workload on PyCanopy, DuckDB, SedonaDB, and GeoPandas. It provisions isolated EC2
instances, runs and verifies each query, and combines the results into a chart and text report.

## What It Runs

- The 12 upstream SpatialBench queries at SF1 or SF10
- One isolated process per query with remote Parquet reads included in wall time
- Multiple engines concurrently on identical `m7i.2xlarge` instances
- An optional PyCanopy profile comparing the current branch with the latest release

Ordinary runs produce a grouped PNG and text report. Profile mode runs SF1 once per query and
writes the combined wall, RSS, native Engine metrics, and verification results to
`assets/profile.txt`.

## Usage

The launcher requires AWS credentials with EC2, SSM, `iam:PassRole`, and results-bucket access.
The benchmark dependencies are installed through the `bench` uv group.

Run all engines at SF1 or SF10:

```bash
make sf1
make sf10
```

Limit a run to selected engines:

```bash
make sf1 engines="pycanopy duckdb"
```

Profile the current PyCanopy branch against the latest release:

```bash
make profile
```

Use the module directly for query selection or repetition control:

```bash
uv run --group bench python -m bench.spatial_bench \
  --scale-factor 1 --engine pycanopy --query q12 --n 3
```

Profile mode takes no additional flags. On-box code is cloned from `repository_branch`, so changes
to the harness or queries must be pushed before launching a run.

## Harness Constraints

- Workload commit `b9221a9c4b02b10db20611d79b4019d2b3c4b68e`, dataset `v0.1.0`, and reference answers are pinned
- Inputs are read directly from the public `s3://pycanopy-bench-data` mirror in `us-west-2`
- Each engine runs on an ephemeral 8-vCPU, 32-GB `m7i.2xlarge` with no cross-query cache reuse
- Ordinary runs install current PyPI releases and record exact versions; profile mode builds the configured branch and installs the latest PyCanopy release beside it
- Every repetition runs in a fresh subprocess with a 1,200-second limit; incomplete repetition sets are invalid
- An OOM fails that engine-query and resumes any remaining queries for the engine on a fresh instance
- Results must match the committed upstream answers, including row order and schema-controlled numeric tolerances
- PyCanopy queries are handwritten against its public API, while DuckDB and SedonaDB run pinned upstream SQL; q11 and q12 intentionally use PyCanopy's public morsel APIs

## File Organization

```text
bench/spatial_bench/
├── __main__.py       # local launcher, monitoring, and result collection
├── config.py         # workload, dataset, engine, and AWS configuration
├── bootstrap.sh      # EC2 setup and suite entry point
├── run_query.py      # isolated execution of one query
├── driver_utils.py   # timed and profiled engine runs
├── profiler_utils.py # RSS sampling, native metrics, and verification
├── report_utils.py   # transport, text, chart, and profile reports
├── queries/          # q01-q12 implementations grouped by engine
├── engines/          # engine-specific execution adapters
└── answers/          # pinned upstream answers and comparison metadata
```
