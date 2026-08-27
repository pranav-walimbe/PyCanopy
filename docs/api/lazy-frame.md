# SpatialLazyFrame

`SpatialLazyFrame` is an immutable plan builder. The optimizer can reorder compatible filters,
fuse spatial predicates, push work into deferred sources, and choose how supported joins execute.
Joins and k-nearest-neighbour operations are plan barriers, and `limit()` is terminal except for a
final `select()` or `count()`.

Execute a plan with `collect()`, `count()`, `collect_batched()`, `sink_parquet()`, or a grouped
aggregation. `lazy_source()` instead exposes its streamed output to a downstream Polars lazy plan.

`SpatialGroupBy` is returned by `.group_by()` and holds the keys for a fused aggregate-join.

::: pycanopy.SpatialLazyFrame

::: pycanopy.SpatialGroupBy
