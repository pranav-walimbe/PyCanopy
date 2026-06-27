# SpatialBench

PyCanopy measured against the [Apache SpatialBench](https://github.com/apache/sedona-spatialbench)
query suite: 12 spatial queries over a NYC taxi + building dataset at scale factors SF1 and SF10.
Published baseline numbers (SedonaDB, DuckDB, GeoPandas) come from `docs/single-node-benchmarks.md`
in that repository, measured on the same hardware.

## Methodology

**Hardware:** Ephemeral **m7i.2xlarge** (8 vCPU, 32 GB RAM, `us-west-2`) launched for each run,
matching the published baseline hardware exactly.

**Cold S3 reads:** Data is read directly from the public SpatialBench S3 bucket inside the timed
window. Each query subprocess starts with a cold page cache.

**Process isolation:** Each query runs in a fresh Python subprocess (`_runner.py`) so no state,
cached data, or compiled code leaks between queries.

**Repetitions:** Each query runs `--n` times (default 3) in separate subprocesses. The reported
time is the average. Each subprocess has a 1200-second timeout matching the published baseline.

**Verification (`--profile`):** A separate diagnostic mode that runs SF1 once per query through an
instrumented path. It attributes wall time and peak RSS to per-stage buckets (fetch / build / query
/ collect), verifies each result against the SedonaDB oracle row-by-row, and writes a summary to
`assets/profile.txt`.

## Usage

Requires AWS credentials with EC2 + S3 permissions. See `config.yaml` for bucket and instance
configuration.

```
# Standard benchmark run (all queries)
python -m bench.spatial_bench --scale-factor {1,10} [--index-eager|--index-auto|--index-none] [--n N]

# Run only specific queries (useful for debugging individual queries)
python -m bench.spatial_bench --scale-factor 1 --query q12
python -m bench.spatial_bench --scale-factor 1 --query q4 q10 q11

# Per-stage profiling + verification (SF1 only)
python -m bench.spatial_bench --profile
```

The launcher spins up an EC2 instance, polls S3 for progress, downloads the result chart PNG (or
`profile.txt`), and terminates the instance when done.

## IAM setup

The EC2 instance uses an instance role (no keys injected). The role needs:

- `s3:GetObject` on the SpatialBench data bucket
- `s3:PutObject` / `s3:GetObject` on the results bucket specified in `config.yaml`

See `config.yaml` for bucket names and the `spatial_bench/README.md` IAM section for the full
policy.

## Directory layout

```
bench/spatial_bench/
├── __main__.py      # local launcher: spin up EC2, poll S3, download chart, terminate
├── _onbox.py        # on-box suite driver: loops over queries, calls _runner.py
├── _runner.py       # per-query subprocess entry point (one fresh interpreter per run)
├── _profile.py      # profile mode: per-stage timing + RSS + oracle verification
├── utils.py         # measure_query, write_chart, verify_outputs, published baselines
├── sedona_sql.py    # authoritative SedonaDB SQL for all 12 queries (oracle source of truth)
├── config.yaml      # fixed infra config: bucket, instance type, repo URL + branch
├── bootstrap.sh     # EC2 user-data: install deps, clone repo, build PyCanopy, run suite
└── queries/
    ├── __init__.py  # registers all query modules in _BY_ID
    ├── q01.py       # one module per query: pycanopy() implementation + compare spec
    ├── ...
    └── q12.py
```
