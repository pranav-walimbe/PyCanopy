"""Pinned Apache SpatialBench q12 for DuckDB."""

SQL = """
-- Q12 (DuckDB): No KNN join, using cross join lateral instead.
-- Ranks trip pickups by the average distance to their 5 nearest buildings.
SELECT
   t.t_tripkey,
   AVG(nb.distance_to_building) AS avg_distance_to_5_nearest
FROM trip t
        CROSS JOIN LATERAL (
   SELECT
       ST_Distance(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(b.b_boundary)) AS distance_to_building
   FROM building b
   ORDER BY distance_to_building
       LIMIT 5
) AS nb
GROUP BY t.t_tripkey
ORDER BY avg_distance_to_5_nearest DESC, t.t_tripkey ASC
LIMIT 100 -- Return only the top 100 most-isolated pickups (bounded result set)
               """
