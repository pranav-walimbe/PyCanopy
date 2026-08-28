# Data Ingestion

Every materialized `SpatialFrame` contains two aligned parts: a Polars DataFrame for attributes and
a native `Engine` for geometry. PyCanopy can build that state from an existing DataFrame or defer
the work until a query reaches a terminal operation.

![Materialized and deferred ingestion paths converge on aligned Polars attributes and native geometry](../assets/diagrams/ingestion-paths.png)

## Materialized input

The eager constructors convert geometry when the frame is created:

- Point coordinate columns become contiguous `float64` x/y arrays in the native engine
- Standard little-endian point WKB is decoded with NumPy buffer views; unusual variants fall back
  to Shapely
- Polygon WKB is decoded into native coordinate and offset arrays, with a Shapely fallback for
  unsupported variants
- Shapely and GeoArrow polygon inputs are normalized into the same native representation
- PyCanopy constructs a new DataFrame rather than modifying the caller's object

An eager WKB constructor keeps the caller's geometry column in the DataFrame. Dropping that column
is an explicit DataFrame operation.

## Deferred input

`SpatialFrame.from_lazy()` stores a Polars `LazyFrame` whose geometry column contains Binary WKB.
`SpatialFrame.scan_parquet()` creates the same source from local paths, globs, path lists, or cloud
URLs. GeoParquet metadata can supply the geometry column and geometry kind.

At a terminal operation, PyCanopy:

1. identifies a safe leading scalar-filter prefix and optional source limit;
2. determines which source columns the remaining plan needs;
3. asks Polars for ordered batches containing those columns and geometry;
4. decodes each geometry batch into native buffers; and
5. assembles the retained attributes in matching row order.

Spatial optimization and execution begin only after the complete native engine exists.

## Source pushdown

PyCanopy pushes only operations whose meaning it can preserve before geometry materialization:

- A leading sequence of supported scalar expressions
- An optional `limit()` immediately after that sequence
- Columns required by filters, joins, and the terminal `select()` or `count()`
- Geometry for every row that reaches materialization

Unsupported, derived, or geometry-dependent expressions remain in the spatial plan. If PyCanopy
does not recognize an expression, it performs the work later; the result does not change.

## Native geometry layout

![Points and polygons use contiguous coordinate and offset buffers](../assets/diagrams/geometry-layout.png)

- Point x/y coordinates live in Rust-owned `Arc<[f64]>` buffers
- Read-only NumPy and Polars views can share those point buffers
- Polygon coordinates are flat arrays indexed by ring, polygon, and part offsets
- Native polygon subsets copy compact geometry buffers instead of decoding WKB again

## Memory boundary

!!! important
    Deferred ingestion does not create a permanently streaming spatial engine. The full native
    geometry dataset exists before spatial execution begins.

`collect_batches()` lets Polars collect the source in chunks and lets PyCanopy release each WKB
batch after decoding. Retained attribute batches remain until PyCanopy assembles the aligned
DataFrame. If the query selects the WKB geometry column, that column is retained like any other
attribute. A blocking operation already present in the supplied `LazyFrame` may materialize its
own input before PyCanopy receives a batch.
