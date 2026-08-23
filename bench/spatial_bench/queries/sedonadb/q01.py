"""Pinned Apache SpatialBench q1 for SedonaDB."""

SQL = """
-- Q1: Find trips starting within 50km of Sedona city center, ordered by distance
SELECT
   t.t_tripkey, ST_X(ST_GeomFromWKB(t.t_pickuploc)) AS pickup_lon, ST_Y(ST_GeomFromWKB(t.t_pickuploc)) AS pickup_lat, t.t_pickuptime,
   ST_Distance(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromText('POINT (-111.7610 34.8697)')) AS distance_to_center
FROM trip t
WHERE ST_DWithin(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromText('POINT (-111.7610 34.8697)'), 0.45) -- 50km radius around Sedona center
ORDER BY distance_to_center ASC, t.t_tripkey ASC
LIMIT 100 -- Return only the 100 closest trips (bounded result set)
               """
