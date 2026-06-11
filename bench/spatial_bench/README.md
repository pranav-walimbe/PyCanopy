# SpatialBench on PyCanopy

Runs PyCanopy against [Apache SpatialBench](https://sedona.apache.org/spatialbench/)
on the official parquet data and compares to the published SedonaDB / DuckDB /
GeoPandas numbers. The heavy run happens on an ephemeral EC2 box; the only
artifact that comes back is a small results JSON, which renders into tables and
charts anywhere.

## One-command AWS run

```bash
pip install --group bench          # boto3, geopandas, etc.
# edit config.yaml: set result_bucket and instance_profile
python -m bench.spatial_bench.aws_run
```

That launches an `m7i.2xlarge` in `us-west-2` (to match the published hardware),
which builds PyCanopy, copies the data, runs every query, uploads the result,
and terminates itself. `aws_run.py` polls for the result, downloads it, and
prints the report. Nothing heavy runs locally.

## Prerequisites

1. **Local AWS credentials** — `aws configure` (or `AWS_PROFILE` / SSO). Never
   stored in the repo; boto3 reads your standard config.
2. **An IAM instance profile** named in `config.yaml`. The box assumes this role
   to read the public data and write results (no keys are copied onto the box).
   Minimal policy for the role:

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

   The official data bucket is public, so the role needs no permission for it
   (the box falls back to an anonymous read).
3. **Permissions for whoever runs `aws_run.py`** — `ec2:RunInstances`,
   `ec2:DescribeInstances`, `ec2:TerminateInstances`, `ssm:GetParameter`,
   `iam:PassRole` (for the instance profile), and read access to the result bucket.

## Safety

The box always terminates: `InstanceInitiatedShutdownBehavior=terminate` plus a
self-`shutdown`, a watchdog that force-shuts-down after `max_runtime_min`, and a
`finally` in `aws_run.py` that terminates the instance even if interrupted.

## Local dev (no AWS)

Run a single scale factor against local or `s3://` parquet, then render — useful
for checking query correctness against the GeoPandas oracle at small scale:

```bash
python -m bench.spatial_bench.measure --data-dir <local|s3://...> --scale-factor 1 \
    --output results/sf1.json
python -m bench.spatial_bench.report --results results/sf1.json
```

The results JSON is the only handoff between `measure` and `report`, so the same
JSON from an AWS run renders identically here.
