# Spatial Query Planning

## Operations become plan nodes

Calling a method on `SpatialLazyFrame` does not run that operation. It creates a typed node and
returns a new frame containing the previous plan plus that node.

For example, a filter, spatial query, and projection become a plan like:

```text
ScalarNode → RangeNode → SelectNode
```

- Each node stores only what its operation needs, such as a Polars expression, query bounds, join
  inputs, or selected columns
- Nodes initially appear in declaration order; optimization begins when a terminal operation
  collects, streams, counts, or sinks the query

| Node category | Examples | Planning role |
|:--------------|:---------|:--------------|
| Scalar filter | `filter()` | Polars expression that may move within a reorderable section |
| Spatial filter | `range_query()`, `contains()` | Row-subset operation ordered by estimated selectivity |
| Nearest neighbour | `knn()` | Changes result shape and forms a planning barrier |
| Spatial join | `within_join()`, `knn_join()` | Combines two inputs and forms a planning barrier |
| Terminal shape | `select()`, `count()` | Defines which attributes the result must retain |
| Row bound | `limit()` | Preserves result order and forms a barrier |

## From a logical plan to execution

![A logical plan moves through source preparation and optimizer passes to become an execution-ordered physical plan](../assets/diagrams/query-plan-transformation.png)

- Materialized frames already have their Polars attributes, native geometry, and dataset statistics,
  so they skip source preparation
- Deferred frames first push a safe leading tabular prefix into Polars and materialize the required
  source rows
- The optimizer then works from the complete engine statistics and emits an execution-ordered plan
- Native access-path planning later decides whether each spatial operation should scan, reuse an
  index, or build a new one

## Ordering and barriers

The optimizer divides a plan at operations whose semantics depend on order. It may reorder filters
inside each resulting section, but never moves them across kNN, joins, self-joins, or limits.

- Scalar filters run before spatial filters within a reorderable section
- Scalar expressions use a small structural cost derived from their Polars expression tree
- Range and radius filters estimate selectivity from the query area and dataset extent
- Containment and kNN nodes use expected result-size estimates
- Spatial filters are ordered from narrower to broader expected results
- These ordering heuristics are separate from the calibrated native index cost model

## Spatial filter fusion

Compatible range and containment filters can share one Rust boundary crossing:

![Filter fusion and projection pushdown reduce boundary crossings and gathered columns](../assets/diagrams/query-plan-rewrites.png)

- Fusion applies only to eligible consecutive range and containment filters
- Very selective filters stay separate because reducing rows early is more valuable than fusion
- Small datasets skip fusion because planning and intersection overhead would outweigh the saved
  boundary crossings

## Projection planning

A terminal `select()` or `count()` tells the planner which attributes must survive execution.

- Join gathers retain requested output columns plus columns required by later filters
- Deferred sources read only source-side columns needed by the remaining plan
- `count()` retains no output attributes, though filters and geometry may still require source data

## Join orientation

- Symmetric point joins may flip when probing from the smaller side avoids unnecessary work
- Supported point-to-polygon distance kernels compare the forward plan with building a grid over the
  point side
- Existing indexes count as already built when native planning compares orientations
- Joins without a safe reverse kernel preserve their declared orientation

## Execution-path selection

After optimization, the planner chooses how spatial results reconnect to Polars rows:

- Joins and kNN use the expression path that carries row indices through the Polars plan
- Highly selective spatial filters can use the direct I/O path, querying the full native engine and
  gathering only matching Polars rows
- Broader filters use the expression path so Polars can apply scalar work before spatial evaluation

This choice controls Polars integration. The native cost model independently chooses the spatial
access path used inside the Rust engine.
