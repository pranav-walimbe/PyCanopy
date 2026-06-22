# Benchmarks

PyCanopy ships two benchmark suites. Both render their output into `assets/`.

| Suite | What it measures | Data |
| --- | --- | --- |
| [`ops/`](#ops-benchmark) | Every spatial primitive, cold and warm, vs a naive GeoPandas baseline | Self-generated |
| [`spatial_bench/`](#spatialbench) | PyCanopy on [Apache SpatialBench](https://sedona.apache.org/spatialbench/) vs the published SedonaDB, DuckDB, and GeoPandas numbers | Official parquet |

## Ops benchmark

```bash
python -m bench.ops
```

Each op runs on a fresh engine (cold, with index build) and again warm, on uniformly
random data with eager indexing. The run prints one line per op and writes a summary
table to `assets/ops.txt`.

## SpatialBench

Benchmarking runs **only** on an EC2 `m7i.2xlarge` to stay comparable to the published
baseline. Each query runs in an isolated subprocess and reads directly from S3 during
the timed window, matching the published methodology. The chart lands in `assets/`.

Run on the box after deploying `bootstrap.sh`:

```bash
python -m bench.spatial_bench --scale-factor 1
```

**Flags**

| Flag | Effect |
| --- | --- |
| `--scale-factor {1,10}` | Scale factor to benchmark (required) |
| `--index-eager` | Build an index whenever a kind is selected (default is cost-based auto) |
| `--verify` | Run the live SedonaDB output check per query |

Writes `assets/spatialbench_sf{N}_auto.png` (or `_eager` when `--index-eager` is passed).

### AWS setup

The box needs an IAM instance profile with permission to read the public SpatialBench
data and write results to your result bucket. Minimal policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::YOUR_RESULT_BUCKET/spatialbench-runs/*" },
    { "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::YOUR_RESULT_BUCKET" }
  ]
}
```

The `@@NAME@@` placeholders in `bootstrap.sh` must be substituted before deployment
(region, result bucket, instance profile, repo URL/branch, scale factor, runtime cap).
