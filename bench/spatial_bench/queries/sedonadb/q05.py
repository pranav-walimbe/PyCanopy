"""Pinned Apache SpatialBench q5 for SedonaDB."""

SQL = """
-- Q5 (SedonaDB): SedonaDB uses ST_Collect_Agg (with _Agg suffix) for aggregate functions.
SELECT
    c.c_custkey, c.c_name AS customer_name,
    DATE_TRUNC('month', t.t_pickuptime) AS pickup_month,
    ST_Area(ST_ConvexHull(ST_Collect_Agg(ST_GeomFromWKB(t.t_dropoffloc)))) AS monthly_travel_hull_area,
    COUNT(*) as dropoff_count
FROM trip t JOIN customer c ON t.t_custkey = c.c_custkey
GROUP BY c.c_custkey, c.c_name, pickup_month
HAVING dropoff_count > 5 -- Only include repeat customers for meaningful hulls
ORDER BY monthly_travel_hull_area DESC, c.c_custkey ASC, pickup_month ASC
LIMIT 100 -- Return only the top 100 repeat customer-months by travel-hull area (bounded result set)
               """
