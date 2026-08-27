# Data Ingestion and Ownership

`SpatialFrame` has two source modes. Both produce the same execution state: aligned Polars
attributes plus native Rust geometry.

```mermaid
flowchart TB
    subgraph Materialized["Materialized frame"]
        MDF["Polars DataFrame"] --> DEC1["decode or copy geometry once"]
    end

    subgraph Deferred["Deferred frame"]
        SRC["Polars LazyFrame or Parquet scan"] --> PUSH["safe filters, limit, projection"]
        PUSH --> BATCH["ordered WKB decode batches"]
    end

    DEC1 --> STATE["execution state"]
    BATCH --> STATE
    STATE --> ATTR["Polars attribute columns"]
    STATE --> GEO["Rust point or polygon buffers"]
```

## Materialized sources

- Coordinate columns build a point engine from contiguous x/y arrays
- Point WKB is decoded into x/y arrays and retained in the DataFrame unless the caller drops it
- Polygon WKB is decoded into native coordinate and offset arrays
- Shapely and GeoArrow polygon inputs are normalized into the same native representation
- The caller's original DataFrame is never modified

## Deferred sources

- `from_lazy()` accepts a Polars `LazyFrame` containing Binary WKB
- `scan_parquet()` creates the same lazy source from local paths, globs, path lists, or cloud URLs
- GeoParquet metadata can supply the geometry column and point/polygon kind
- Nothing is decoded and no engine exists until a terminal operation prepares the plan

```mermaid
flowchart LR
    PLAN["leading plan nodes"] --> SAFE{"safe source prefix?"}
    SAFE -- "scalar filter" --> FILTER["Polars filter pushdown"]
    FILTER --> LIMIT["optional terminal limit"]
    SAFE -- "spatial op or barrier" --> KEEP["leave in spatial plan"]
    LIMIT --> PROJECT["required source columns"]
    KEEP --> PROJECT
    PROJECT --> DECODE["collect batches and decode WKB"]
```

## What is pushed into the source

- A leading sequence of supported scalar expressions
- An optional `limit()` immediately after that sequence
- Columns required by filters, joins, and the terminal `select()` or `count()`
- Geometry for every row that reaches native materialization

Unsupported, derived, or geometry-dependent expressions remain in the spatial plan. Pushdown is
conservative: failing to recognize an expression changes performance, not results.

## Geometry representation

```mermaid
flowchart TB
    P["Points"] --> XY["x: Arc&lt;[f64]&gt;<br/>y: Arc&lt;[f64]&gt;"]
    G["Polygons and MultiPolygons"] --> C["flat x/y coordinates"]
    G --> RO["ring offsets"]
    G --> PO["polygon offsets"]
    G --> PM["part-to-polygon mapping"]
```

- Flat arrays avoid one Python or Rust object per geometry
- Offset arrays represent variable-length rings and polygon parts
- Point coordinate buffers can be shared by scan and grid paths
- Derived polygon subsets copy compact native geometry rather than decoding WKB again

## Memory boundary

!!! important
    A deferred frame is deferred ingestion, not a permanently streaming spatial engine. Collection
    builds the complete native geometry dataset before spatial execution.

- Batch decoding avoids requiring one large Python geometry-object array
- Selected attribute columns remain in Polars
- WKB batches can be released after decode unless geometry is selected in the output
- A blocking operation already present in the supplied Polars `LazyFrame` may materialize its own
  input before PyCanopy receives batches
