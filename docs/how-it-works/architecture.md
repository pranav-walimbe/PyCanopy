# Architecture Overview

PyCanopy is a spatial query layer around Polars rather than a replacement DataFrame engine.

```mermaid
flowchart TB
    Q["1. Declare a spatial query<br/>with the Polars-like Python API"]
    P["2. Build and optimize<br/>an immutable spatial plan"]
    E["3. Execute together<br/>Polars handles tables · Rust handles spatial work"]
    O["4. Return a Polars DataFrame<br/>or stream results to a sink"]
    Q --> P --> E --> O
```

## Design boundaries

- **Polars owns tabular work:** file scans, scalar expressions, column projection, row gathering,
  joins by key, sorting, and final output
- **Rust owns spatial work:** native geometry, dataset statistics, spatial indexes, candidate
  traversal, exact geometry predicates, and parallel kernels
- **Python connects the two:** immutable plan nodes, optimization passes, execution routing, and
  morsel orchestration
- **Spatial indexes are optional:** the engine can scan, reuse an index, or build one for the
  current workload

## One query through the system

```python
result = (
    zones.lazy()
    .within_join(trips, x_col="lon", y_col="lat")
    .filter(pl.col("fare") > 20)
    .select("trip_id", "zone_id")
    .collect()
)
```

```mermaid
flowchart TB
    PLAN["Declared plan<br/>join → fare filter → projection"]
    OPT["Optimized plan<br/>preserve join barrier · narrow gathered columns"]
    JOIN["Rust spatial join<br/>return query and target row indices"]
    RESULT["Polars<br/>gather columns · filter fare · build result"]
    PLAN --> OPT --> JOIN --> RESULT
```

## Design philosophy

- Keep user data in Polars instead of introducing a second table abstraction
- Cross the Python/Rust boundary with contiguous arrays and compact row indices
- Build an index only when its expected reuse or probe savings repay its build cost
- Bound large join intermediates without claiming every query is out of core
- Keep query-specific planning in the library rather than requiring handwritten execution plans

The following pages trace each layer in more detail, from source ingestion through result
materialization.
