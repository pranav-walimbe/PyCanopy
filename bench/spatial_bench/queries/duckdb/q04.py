"""Pinned Apache SpatialBench q4 for DuckDB."""

SQL = """
-- Q4: Zone distribution of top 1000 trips by tip amount
SELECT z.z_zonekey, z.z_name, COUNT(*) AS trip_count
FROM
   zone z
       JOIN (
       SELECT t.t_pickuploc
       FROM trip t
       ORDER BY t.t_tip DESC, t.t_tripkey ASC
           LIMIT 1000 -- Replace 1000 with x (how many top tips you want)
   ) top_trips ON ST_Within(ST_GeomFromWKB(top_trips.t_pickuploc), ST_GeomFromWKB(z.z_boundary))
GROUP BY z.z_zonekey, z.z_name
ORDER BY trip_count DESC, z.z_zonekey ASC
               """
