"""Pinned Apache SpatialBench q11 for DuckDB."""

SQL = """
-- Q11: Count trips that cross between different zones
SELECT COUNT(*) AS cross_zone_trip_count
FROM
   trip t
       JOIN zone pickup_zone ON ST_Within(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(pickup_zone.z_boundary))
       JOIN zone dropoff_zone ON ST_Within(ST_GeomFromWKB(t.t_dropoffloc), ST_GeomFromWKB(dropoff_zone.z_boundary))
WHERE pickup_zone.z_zonekey != dropoff_zone.z_zonekey
               """
