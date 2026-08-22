# PyCanopy Remaining Work

This file tracks actionable work remaining after the source, documentation, and SpatialBench
cloud-harness audit. Preserve unrelated user worktree changes when addressing these items.

## Performance work

### Prioritize I/O and late materialization

Fetch dominates q1, q2, q3, and q6 at SF1, limiting the value of further spatial-kernel tuning for
those queries. q4 reads every trip WKB value before retaining only the top 1,000 tips.

Investigate row-aware or two-phase late materialization for q4 and avoid retaining geometry
columns that the final query does not return.

### Push eligible filters into lazy sources

Filters written on the Polars `LazyFrame` before `SpatialFrame.from_lazy` can reach the source,
but PyCanopy `.filter(...)` operations currently run after lazy ingestion. PyCanopy projects the
filter's required columns correctly, then reads and decodes geometry for every source row before
applying it.

Later, identify target-side scalar filters that can safely run before the first spatial operation
and insert them into the source `LazyFrame`. Preserve declaration semantics, source row alignment,
and query-versus-target column ownership around joins. This is an I/O optimization, not a
correctness blocker.

### Fuse q11 endpoint lookup

q11 creates pickup and dropoff polygon-pair frames, joins them in Polars, and then counts trips
whose endpoints belong to different zones.

Add a dual-endpoint polygon lookup and fused cross-zone counter to avoid both pair frames, their
gathers, and the intermediate hash join.

## SpatialBench validity and measurement

### Record provenance and robust statistics

The harness records the pinned workload revision, dataset version, and every raw sample. Add git
SHA and dirty state, dataset checksum, AMI ID, kernel, CPU, EBS settings, dependency and compiler
versions, thread counts, source region, and timestamps. Avoid an unversioned latest AMI for
comparable runs.

Replace arithmetic means as the primary summary with medians plus dispersion or confidence
intervals. Do not silently accept partial sample sets.

### Rebuild cross-engine comparisons

Historical SedonaDB, DuckDB, and GeoPandas results used obsolete query definitions and have been
removed from current charts. Rerun every compared engine against the pinned workload, dataset, and
infrastructure before restoring comparative claims.

## Deferred measurement and tuning

- Evaluate planner regret by timing every available index for each calibration shape and comparing
  the planner choice with the fastest measured choice.
- Add hardware detection, alternate cost-profile selection, and a user-facing tuning API.

## Documentation cleanup

- Reconcile q12 timings in `docs/benchmarks.md` with the selected workload version.
- Update aggregation documentation to distinguish direct Rust fusion from morsel reduction.
- Correct quickstart language that says eager mode always builds immediately.
- Identify the workload and dataset version in every public benchmark claim.

## Recommended order

1. Complete benchmark provenance and rerun comparison engines on the pinned workload.
2. Add late materialization for q4 and remove unnecessary retained geometry columns.
3. Fuse q11 dual-endpoint zone lookup.

## Deferred source ergonomics

Add GeoParquet-aware discovery to `SpatialFrame.scan_parquet` so `geometry_col` and
`geometry_kind` can be inferred from metadata. Preserve explicit overrides, require a choice when
multiple geometry columns exist, validate metadata across files, and report unsupported geometry
types clearly. Keep `from_lazy` explicit because a general Polars `LazyFrame` may not retain source
GeoParquet metadata.
