# Architecture Overview

PyCanopy is a spatial query layer around Polars rather than a replacement DataFrame engine.

![PyCanopy architecture showing how its planner routes tabular work to Polars and spatial work to the Rust engine](../assets/diagrams/architecture-overview.png)

## Architecture Philosophy

- **Delegate tabular work to Polars:** rely on Polars for scans, scalar expressions, projections,
  row gathering, tabular joins, and final output
- **Abstract away complexity:** expose a DataFrame-oriented API while planning execution paths,
  indexes, join direction, and spatial kernels under the hood
- **Bound peak memory:** design ingestion, joins, and aggregation around batches, morsels, compact
  row indices, and narrow intermediates
- **Centralize cost-based decisions:** use one calibrated native model to compare scans, existing
  indexes, new indexes, index kinds, and spatial join direction
- **Favor contiguous representations:** keep coordinates, packed indexes, and kernel inputs in
  cache-efficient memory layouts with minimal copying
