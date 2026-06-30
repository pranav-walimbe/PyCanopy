# SpatialLazyFrame

`SpatialLazyFrame` is an immutable plan builder. Operations can be declared in any order — the optimizer reorders, fuses, and pushes them down before execution. Nothing runs until `.collect()` is called.

`SpatialGroupBy` is returned by `.group_by()` and holds the keys for a fused aggregate-join.

::: pycanopy.SpatialLazyFrame

::: pycanopy.SpatialGroupBy
