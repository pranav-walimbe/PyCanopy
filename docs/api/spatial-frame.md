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

No source rows are collected and no spatial engine is built until query collection. A terminal
PyCanopy `.select(...)` is pushed into the source scan. WKB is always read to construct native
geometry, but it remains in the tabular result only when geometry is selected. Without an explicit
select, all source columns, including geometry, are returned.

The projected source is currently materialized before WKB decoding. Output joins can still stream
in morsels, but input geometry ingestion is not yet batch-bounded.

::: pycanopy.SpatialFrame
