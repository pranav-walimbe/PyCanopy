# Engine

`Engine` is the low-level native geometry and spatial-index layer behind `SpatialFrame`. Most users
should build queries through `SpatialFrame.lazy()`. Use `Engine` directly for geometry queries
without DataFrame row gathering or for its standalone geometry computations.

For coordinate systems and the standalone distance utilities, see [Coordinate Reference System](coordinate-reference-system.md).

::: pycanopy.Engine
    options:
      filters:
        - "!^_"
        - "!^batch_"
        - "!^fused"
        - "!^intersect"
        - "!^range_mask"
        - "!^contains_mask"
        - "!^knn_mask"
        - "!^polygon_pairs"
