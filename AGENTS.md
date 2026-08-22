# PyCanopy Remaining Work

This file tracks actionable work remaining after the source, documentation, and SpatialBench
cloud-harness audit. Preserve unrelated user worktree changes when addressing these items.

## Performance work

### Evaluate deeper row-aware geometry reads

The q4 query now ranks trips from scalar columns before scanning pickup WKB for the selected keys.
Measure whether its second Parquet scan still reads substantial geometry data because selected
rows span many pages. If that scan remains material, consider a source-execution feature that
carries stable selected row identities into geometry ingestion and reads only supported source
pages or row ranges. This would be a lazy-source integration change, not a spatial-kernel change;
do not add it without benchmark evidence that the query-level approach is insufficient.

## SpatialBench validity and measurement

### Add multi-engine harness support

Allow one run to select multiple engines and render their results together while keeping the
pinned workload, dataset, infrastructure, and output format consistent across engines.

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

1. Add multi-engine harness support.
2. Complete the documentation cleanup.
3. Rerun comparison engines on the pinned workload.
4. Evaluate whether q4 needs deeper row-aware source reads.
5. Add GeoParquet metadata discovery.
6. Evaluate planner regret.
7. Add hardware profiles and a user-facing tuning API.

## Deferred source ergonomics

Add GeoParquet-aware discovery to `SpatialFrame.scan_parquet` so `geometry_col` and
`geometry_kind` can be inferred from metadata. Preserve explicit overrides, require a choice when
multiple geometry columns exist, validate metadata across files, and report unsupported geometry
types clearly. Keep `from_lazy` explicit because a general Polars `LazyFrame` may not retain source
GeoParquet metadata.
