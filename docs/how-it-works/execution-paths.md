# Physical Execution

The executor turns an optimized plan into Polars operations and native spatial calls. Plan shape
and estimated selectivity determine how rows cross that boundary.

![The executor routes joins through morsels and spatial filters through either a Polars expression or direct index path](../assets/diagrams/execution-routing.png)

## Expression path

Use Polars to reduce rows before native spatial evaluation:

- Add a temporary original-row index
- Run scalar Polars expressions first
- Pass surviving row indices to Rust through `map_batches`
- Evaluate the point spatial predicate over those candidates
- Return an aligned Boolean mask to Polars

This path avoids building a temporary spatial index over the filtered subset. It is a Python
callback inside a Polars lazy expression, not a native Polars expression plugin.

## Direct index path

Query the complete native engine first when the spatial predicate is selective:

- Resolve each spatial predicate to compact row-index lists
- Intersect multiple hit lists before gathering attributes
- Gather matching `SpatialFrame` rows once
- Run remaining scalar filters over that smaller DataFrame

Polygon filters use this path because native polygon geometry is not represented as Polars
coordinate columns. kNN without a preceding scalar filter also queries the complete engine.

This decision only controls the Polars integration path. The native cost model separately chooses
whether a kernel scans, reuses an index, or builds one.

## From kernel to result

The native and tabular stages exchange compact indices rather than full rows:

![Native access and exact refinement produce compact indices that drive narrow Polars gathers and post-join work](../assets/diagrams/kernel-result-pipeline.png)

### Native spatial work

- Bounds and indexes discard geometries that cannot match
- Point-in-polygon uses prepared polygon bands for exact containment
- Polygon distance and kNN refine bounding-box candidates with exact geometry distance
- Eligible grouped kernels update aggregate state instead of emitting every pair
- Long-running Rust sections release the Python GIL and use Rayon where parallelism helps

### Join assembly

- Kernels return query-side and target-side row indices
- Projection planning narrows both gathers before rows are copied
- Conflicting target column names receive a `right_` prefix
- Post-join Polars nodes run over the assembled morsel

## Packed in-memory structures

- `geo-index` KD-trees and R-trees keep traversal data in contiguous packed buffers
- Point coordinates live in Rust-owned `Arc<[f64]>` arrays shared by scan and grid paths
- KD-trees share coordinates for exact refinement while owning packed traversal data
- Polygon rings and parts use flat coordinate arrays plus offsets instead of nested objects

These layouts reduce pointer chasing and improve cache locality during traversal.

## Boundary crossings and copies

- Coordinate constructors normalize inputs to contiguous NumPy arrays and copy once into
  Rust-owned storage
- Point and polygon WKB are decoded from Arrow buffers without one Python geometry object per row
- Native kernels return arrays of indices, masks, distances, or aggregate state
- Engine coordinate buffers can be exposed as read-only NumPy and Polars views without another
  coordinate copy

“Zero-copy” applies to those shared buffers, not the entire query. Row gathers, index builds, input
normalization, and final result construction can still allocate.
