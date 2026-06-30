# Engine & Utilities

`Engine` is the low-level spatial index that backs every `SpatialFrame`. Most users interact with it only through `SpatialFrame.engine` for delta buffer operations or batch convex-hull computations. It can also be used directly when you need index access without the DataFrame layer.

`wkb_point_distance` and `wkb_points_to_xy` are standalone utility functions for working with WKB-encoded geometry columns.

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

::: pycanopy.wkb_point_distance

::: pycanopy.wkb_points_to_xy
