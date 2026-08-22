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

1. Fuse q11 dual-endpoint zone lookup.
2. Push eligible filters into lazy sources.
3. Add multi-engine harness support.
4. Complete the documentation cleanup.
5. Rerun comparison engines on the pinned workload.
6. Evaluate whether q4 needs deeper row-aware source reads.
7. Add GeoParquet metadata discovery.
8. Evaluate planner regret.
9. Add hardware profiles and a user-facing tuning API.

## Deferred source ergonomics

Add GeoParquet-aware discovery to `SpatialFrame.scan_parquet` so `geometry_col` and
`geometry_kind` can be inferred from metadata. Preserve explicit overrides, require a choice when
multiple geometry columns exist, validate metadata across files, and report unsupported geometry
types clearly. Keep `from_lazy` explicit because a general Polars `LazyFrame` may not retain source
GeoParquet metadata.
