"""Pinned Apache SpatialBench q12 for SedonaDB."""

SQL = """
-- Q12: Rank trip pickups by the average distance to their 5 nearest buildings (KNN join).
-- The top results are the most isolated pickups (farthest from surrounding buildings).
WITH trip_with_geom AS (
   SELECT t_tripkey, ST_GeomFromWKB(t_pickuploc) as pickup_geom
   FROM trip
),
    building_with_geom AS (
        SELECT ST_GeomFromWKB(b_boundary) as boundary_geom
        FROM building
    ),
    knn AS (
        SELECT
            t.t_tripkey,
            ST_Distance(t.pickup_geom, b.boundary_geom) AS distance_to_building
        FROM trip_with_geom t JOIN building_with_geom b
                                  ON ST_KNN(t.pickup_geom, b.boundary_geom, 5, FALSE)
    )
SELECT
   t_tripkey,
   AVG(distance_to_building) AS avg_distance_to_5_nearest
FROM knn
GROUP BY t_tripkey
ORDER BY avg_distance_to_5_nearest DESC, t_tripkey ASC
LIMIT 100 -- Return only the top 100 most-isolated pickups (bounded result set)
               """
