# Query Planning

Operations on a `SpatialLazyFrame` build an immutable plan. At collection time, the
`SpatialOptimizer` rewrites the plan and selects an execution path for the
`SpatialExecutor`.

## Optimization features

### Selectivity and cost estimation

Spatial-filter selectivity is estimated from the dataset extent and query shape. Scalar
Polars filters receive heuristic costs based on expression structure, not column
statistics. These estimates guide ordering and execution-path selection.

### Predicate pushdown and ordering

Within a reorderable section, scalar filters run before spatial filters. Scalars are
ordered by estimated expression cost and spatial filters by selectivity. kNN operations
and joins are barriers; filters do not move across them.

### Filter fusion

On datasets with at least 500 rows, eligible `range_query` and `contains` filters may be
combined. Rust queries each predicate and intersects their sorted result indices. Filters
estimated to retain less than 5% of rows remain separate.

### Join orientation

`within_join` and `within_distance_join` may reverse the indexed and probe sides based on
their relative sizes. Other joins retain their declared orientation.

### Projection pushdown

A terminal `.select(...)` is pushed into spatial joins. Each side is narrowed before rows
are gathered, while preserving columns required by post-join filters.

## Execution-path selection

After rewriting the plan, PyCanopy selects one execution path for the query:

- **IO path:** For sufficiently selective `range_query`, `contains`, or fused spatial
  filters, the executor queries the spatial engine directly and slices the source
  DataFrame using the returned row indices.
- **EXPR path:** For broader filters, scalar Polars expressions run first and the spatial
  operation evaluates the surviving original row indices through Polars' `map_batches`
  API. kNN and spatial join plans also use this path.

This chooses how the plan interacts with Polars. The Rust cost model separately chooses
whether to scan, reuse an index, or build a new one.
