# SpatialBench Test Harness

## Benchmark

PyCanopy is measured against the [Apache SpatialBench](https://github.com/apache/sedona-spatialbench)
query suite: 12 spatial queries over a NYC taxi + zone dataset at scale factors SF1 and SF10.
Published baseline numbers (SedonaDB, DuckDB, GeoPandas) come from
`docs/single-node-benchmarks.md` in that repository, measured on the same m7i.2xlarge hardware / constraints.

## Methodology

**Hardware:** Each run uses an ephemeral **m7i.2xlarge** (8 vCPU, 32 GB RAM, `us-west-2`)
launched for the benchmark which matches the published baseline hardware.

**Cold S3 Reads** Data is read directly from the public S3 bucket
(`s3://wherobots-examples/...`) inside the timed window. 

**Per-query Process Isolation:** Each query runs in a **fresh Python subprocess**
(`python -m bench.spatial_bench._runner`) to make timings accurate.

**Repetitions:** Each query is executed **3 times** in separate subprocesses. The reported
time is the **average of all three runs**. Each subprocess has a **1200-second per-query timeout**, matching the published baseline timeout.

**Output Verification:** Users can optionally specify a --verify flag on a benchmark run to trigger output comparison between Apache Sedona and PyCanopy. Note: outputs can be very large to materialize so this is only recommended for SF1.

## Run Benchmark

Requires AWS credentials with EC2 + S3 permissions. IAM setup and bucket configuration are
in `config.yaml`.

```
python -m bench.spatial_bench --scale-factor {1,10} [--index-eager|--index-auto|--index-none] [--verify]
```

The launcher spins up an EC2 box, polls S3 for completion, downloads the chart PNG, and
terminates the instance.

## Directory Layout

```
bench/spatial_bench/
├── __main__.py      # local EC2 launcher (spin up, poll, download chart, terminate)
├── _onbox.py        # on-box suite driver (called by bootstrap.sh, loops over queries)
├── _runner.py       # per-query subprocess entry point (isolated interpreter per query)
├── utils.py         # measure_query, write_chart, verify_outputs, PUBLISHED baselines
├── sedona_sql.py    # SedonaDB SQL for each query (used by the oracle verifier)
├── config.yaml      # fixed infra config (bucket, instance type, repo branch)
├── bootstrap.sh     # EC2 user-data script (installs deps, clones repo, calls _onbox.py)
└── queries/
    ├── q01.py             # one file per SpatialBench query (pycanopy() + compare spec)
    ├── ...
    ├── q12.py
    └── __init__.py        # registers queries by id in _BY_ID
```