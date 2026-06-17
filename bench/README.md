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

The heavy run happens on an ephemeral EC2 box, matched to the published `m7i.2xlarge`
hardware. The only thing that comes back is a comparison chart PNG in `assets/`.

```bash
pip install --group bench                     # boto3, geopandas, etc.
# edit spatial_bench/config.yaml: set result_bucket and instance_profile
python -m bench.spatial_bench.aws_run
```

The box builds PyCanopy, runs every query, uploads the chart, and self-terminates.
Locally, `aws_run.py` polls S3, streams progress, downloads the PNG, and terminates the
box.

**Flags**

| Flag | Effect |
| --- | --- |
| `--index-eager` (default) | Build an index whenever a kind is selected |
| `--index-auto` | Build an index only when the cost model beats a scan |
| `--index-none` | Brute-force every query |
| `--no-verify` | Skip the live SedonaDB output check |

Scale factor and data location live in `config.yaml`.

### Run one scale factor locally

No AWS needed. Same flags, plus `--queries q1 q4 ...` for a subset.

```bash
python -m bench.spatial_bench.utils --data-dir <local|s3://...> --scale-factor 1
```

Writes `assets/spatialbench_sf1.png`. This is the exact module the EC2 box runs, so
local and cloud runs render identically.

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

3. **Runner permissions.** Whoever runs `aws_run.py` needs `ec2:RunInstances`,
   `ec2:DescribeInstances`, `ec2:TerminateInstances`, `ssm:GetParameter`, `iam:PassRole`,
   and read access to the result bucket.

The box always terminates: shutdown-on-terminate, a `max_runtime_min` watchdog, and a
`finally` in `aws_run.py` that kills it even on interrupt.
