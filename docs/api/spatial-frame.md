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

At collection, eligible filters, limits, and projections are pushed into the lazy source before WKB
is decoded in ordered batches. Geometry is read for every surviving row but is returned as a column
only when selected. Without `.select()`, all source columns are returned.

The complete native geometry dataset is built before spatial execution. `ingest_batch_size`
controls decode batch size and defaults to 32,768 rows. Blocking operations already present in the
supplied `LazyFrame` may still materialize their input.

## Inspecting GeoParquet metadata

`scan_parquet()` calls `infer_geoparquet_geometry()` automatically when the geometry column or kind
is omitted. The utility is also available directly when you only need to inspect a source:

```python
from pycanopy import infer_geoparquet_geometry

geometry_col, geometry_kind = infer_geoparquet_geometry("zones.parquet")
```

::: pycanopy.infer_geoparquet_geometry

::: pycanopy.SpatialFrame
