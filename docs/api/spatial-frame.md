# SpatialFrame

`SpatialFrame` is the entry point for all spatial queries. It can own a materialized Polars
DataFrame and spatial engine or defer both behind a Polars `LazyFrame`. All declarative query
planning begins with `.lazy()`.

## Lazy sources

Use `from_lazy` with any Polars lazy source or transformation:

```python
zones = SpatialFrame.from_lazy(
    pl.scan_parquet("s3://bucket/zones/*.parquet").filter(pl.col("state") == "CA"),
    geometry_col="geometry",
    geometry_kind="polygon",
)
```

For direct Parquet access, `scan_parquet` forwards scan options to Polars:

```python
zones = SpatialFrame.scan_parquet(
    "s3://bucket/zones/*.parquet",
    geometry_col="geometry",
    geometry_kind="polygon",
    storage_options={"skip_signature": "true"},
)
```

Point WKB sources use the same APIs:

```python
events = SpatialFrame.scan_parquet(
    "s3://bucket/events/*.parquet",
    geometry_col="geometry",
    geometry_kind="point",
    coordinate_system="geographic",
    ingest_batch_size=32_768,
)
```

No source rows are collected and no spatial engine is built until query collection. A terminal
PyCanopy `.select(...)` is pushed into the source scan. WKB is always read to construct native
geometry, but it remains in the tabular result only when geometry is selected. Without an explicit
select, all source columns, including geometry, are returned.

Projected input is consumed in ordered batches. Each WKB batch is decoded into native point or
polygon buffers and released before the next batch. The completed engine still holds the full
native geometry dataset, while selected attributes remain as chunked Polars columns. Blocking
operations in the supplied `LazyFrame`, such as a global sort, may still materialize upstream.

`ingest_batch_size` defaults to 32,768 rows. Smaller values reduce temporary geometry memory.
Larger values may improve throughput for inexpensive geometries.

::: pycanopy.SpatialFrame
