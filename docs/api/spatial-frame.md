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
PyCanopy `.select(...)` narrows the source scan. A safe leading sequence of scalar filters and an
optional limit are also pushed into the lazy source before geometry is decoded. WKB is always read
for the rows used to construct native geometry, but it remains in the tabular result only when
selected. Without an explicit select, all source columns, including geometry, are returned.

Projected input is decoded in ordered batches into native point or polygon buffers. A WKB batch can
be released after decoding unless the geometry column is part of the requested output. The
completed engine still holds the full native geometry dataset, while selected attributes remain as
chunked Polars columns. Blocking operations in the supplied `LazyFrame`, such as a global sort, may
still materialize upstream.

`ingest_batch_size` defaults to 32,768 rows. Smaller values reduce temporary geometry memory.
Larger values may improve throughput for inexpensive geometries.

## Inspecting GeoParquet metadata

`scan_parquet()` calls `infer_geoparquet_geometry()` automatically when the geometry column or kind
is omitted. The utility is also available directly when you only need to inspect a source:

```python
from pycanopy import infer_geoparquet_geometry

geometry_col, geometry_kind = infer_geoparquet_geometry("zones.parquet")
```

::: pycanopy.infer_geoparquet_geometry

::: pycanopy.SpatialFrame
