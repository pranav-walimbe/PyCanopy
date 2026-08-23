"""Pinned Apache SpatialBench q8 for DuckDB."""

SQL = """
-- Q8: Count nearby pickups for each building within 500m radius
SELECT b.b_buildingkey, b.b_name, COUNT(*) AS nearby_pickup_count
FROM trip t JOIN building b ON ST_DWithin(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(b.b_boundary), 0.0045) -- ~500m
GROUP BY b.b_buildingkey, b.b_name
ORDER BY nearby_pickup_count DESC, b.b_buildingkey ASC
LIMIT 100 -- Return only the top 100 busiest buildings (bounded result set)
               """
