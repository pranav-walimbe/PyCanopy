"""Pinned Apache SpatialBench q2 for SedonaDB."""

SQL = """
-- Q2: Count trips starting within Coconino County (Arizona) zone
SELECT COUNT(*) AS trip_count_in_coconino_county
FROM trip t
WHERE ST_Intersects(ST_GeomFromWKB(t.t_pickuploc), (SELECT ST_GeomFromWKB(z.z_boundary) FROM zone z WHERE z.z_name = 'Coconino County' LIMIT 1))
               """
