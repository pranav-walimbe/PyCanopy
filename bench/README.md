# Benchmarks

PyCanopy ships two benchmark suites. Both render their output into `assets/`.

| Suite | What it measures | Data |
| --- | --- | --- |
| [`ops/`](#ops-benchmark) | Every spatial primitive, cold and warm, vs a naive GeoPandas baseline | Self-generated |
| [`spatial_bench/`](#spatialbench) | PyCanopy on [Apache SpatialBench](https://sedona.apache.org/spatialbench/) vs the published SedonaDB, DuckDB, and GeoPandas numbers | Official parquet |

## Ops benchmark

```bash
python -m bench.ops.run                       # uniform data
python -m bench.ops.run --distribution clustered
```

Each op runs on a fresh engine (cold, with index build) and again warm. The run prints
one line per op and writes a summary table to `assets/`.

## SpatialBench

Benchmarking runs **only** on an ephemeral EC2 box matched to the published
`m7i.2xlarge` hardware, never locally, so the numbers stay comparable to the published
baseline. The only thing that comes back is a comparison chart PNG in `assets/`.

```bash
uv sync --group bench                         # boto3, geopandas, etc.
# edit spatial_bench/config.yaml: set result_bucket and instance_profile
python -m bench.spatial_bench --scale 1
```

The box builds PyCanopy, runs every query, uploads the chart, and self-terminates. The
launcher polls S3, streams progress, downloads the PNG, and terminates the box.

**Flags**

| Flag | Effect |
| --- | --- |
| `--scale {1,10}` | Scale factor to benchmark (required) |
| `--index-eager` (default) | Build an index whenever a kind is selected |
| `--index-auto` | Build an index only when the cost model beats a scan |
| `--index-none` | Brute-force every query |
| `--no-verify` | Skip the live SedonaDB output check |

The scale factor, index mode, and verification are CLI flags. The fixed infrastructure
(bucket, instance, data location, repo) lives in `config.yaml`. Writes
`assets/spatialbench_sf{N}.png`.

### AWS setup

1. **Credentials.** `aws configure` (or `AWS_PROFILE` / SSO). boto3 reads your standard
   config. Nothing is stored in the repo or copied onto the box.
2. **Instance profile.** Name it in `config.yaml`. The box assumes this role to read the
   public data and write results. Minimal policy:

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

3. **Runner permissions.** Whoever runs `python -m bench.spatial_bench` needs
   `ec2:RunInstances`, `ec2:DescribeInstances`, `ec2:TerminateInstances`,
   `ssm:GetParameter`, `iam:PassRole`, and read access to the result bucket.

The box always terminates: shutdown-on-terminate, a `max_runtime_min` watchdog, and a
`finally` in the launcher that kills it even on interrupt.
