# Engine

`Engine` is the low-level spatial index that backs every `SpatialFrame`. Most users interact with it only through `SpatialFrame.engine` for delta buffer operations or batch convex-hull computations. It can also be used directly when you need index access without the DataFrame layer.

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
