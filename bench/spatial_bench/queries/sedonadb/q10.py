"""Pinned Apache SpatialBench q10 for SedonaDB."""

SQL = """
-- Q10: Zone statistics for trips starting within each zone
SELECT
   z.z_zonekey, z.z_name AS pickup_zone, AVG(t.t_dropofftime - t.t_pickuptime) AS avg_duration,
   AVG(t.t_distance) AS avg_distance, COUNT(t.t_tripkey) AS num_trips
FROM zone z LEFT JOIN trip t ON ST_Within(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(z.z_boundary))
GROUP BY z.z_zonekey, z.z_name
ORDER BY avg_duration DESC NULLS LAST, z.z_zonekey ASC
LIMIT 100 -- Return only the top 100 zones by average trip duration (bounded result set)
               """
