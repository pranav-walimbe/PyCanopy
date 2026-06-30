# SpatialFrame

`SpatialFrame` is the entry point for all spatial queries. It owns a materialized Polars DataFrame, the spatial index engine, and cached column metadata. All declarative query planning begins with `.lazy()`.

::: pycanopy.SpatialFrame
