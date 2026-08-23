"""Pinned Apache SpatialBench q3 for SedonaDB."""

SQL = """
-- Q3: Monthly trip statistics for a buffered box around Sedona city center
SELECT
   DATE_TRUNC('month', t.t_pickuptime) AS pickup_month, COUNT(t.t_tripkey) AS total_trips,
   AVG(t.t_distance) AS avg_distance, AVG(t.t_dropofftime - t.t_pickuptime) AS avg_duration,
   AVG(t.t_fare) AS avg_fare
FROM trip t
WHERE ST_DWithin(
             ST_GeomFromWKB(t.t_pickuploc),
             ST_GeomFromText('POLYGON((-111.9060 34.7347, -111.6160 34.7347, -111.6160 35.0047, -111.9060 35.0047, -111.9060 34.7347))'), -- ~26.5 km E-W by ~30 km N-S box around Sedona; corners ~20 km out
             0.045 -- Additional 5km buffer
     )
GROUP BY pickup_month
ORDER BY pickup_month
"""
