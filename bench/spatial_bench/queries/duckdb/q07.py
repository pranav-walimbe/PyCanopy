"""Pinned Apache SpatialBench q7 for DuckDB."""

SQL = """
-- Q7: Detect potential route detours by comparing reported vs. geometric distances
WITH trip_lengths AS (
   SELECT
       t.t_tripkey,
       t.t_distance AS reported_distance_m,
       ST_Length(
               ST_MakeLine(
                       ST_GeomFromWKB(t.t_pickuploc),
                       ST_GeomFromWKB(t.t_dropoffloc)
               )
       ) / 0.000009 AS line_distance_m -- 1 meter = 0.000009 degree
   FROM trip t
)
SELECT
   t.t_tripkey,
   t.reported_distance_m,
   t.line_distance_m,
   t.reported_distance_m / NULLIF(t.line_distance_m, 0) AS detour_ratio
FROM trip_lengths t
ORDER BY detour_ratio DESC NULLS LAST, reported_distance_m DESC, t_tripkey ASC
LIMIT 100 -- Return only the top 100 highest-detour trips (bounded result set)
               """
