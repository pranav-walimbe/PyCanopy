"""Pinned Apache SpatialBench q6 for DuckDB."""

SQL = """
-- Q6: Zone statistics for trips intersecting a bounding box
SELECT
   z.z_zonekey, z.z_name,
   COUNT(t.t_tripkey) AS total_pickups, AVG(t.t_distance) AS avg_distance,
   AVG(t.t_dropofftime - t.t_pickuptime) AS avg_duration
FROM trip t, zone z
WHERE ST_Intersects(ST_GeomFromText('POLYGON((-112.2110 34.4197, -111.3110 34.4197, -111.3110 35.3197, -112.2110 35.3197, -112.2110 34.4197))'), ST_GeomFromWKB(z.z_boundary))
 AND ST_Within(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(z.z_boundary))
GROUP BY z.z_zonekey, z.z_name
ORDER BY total_pickups DESC, z.z_zonekey ASC
               """
