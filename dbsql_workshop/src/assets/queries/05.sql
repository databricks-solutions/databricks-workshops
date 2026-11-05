SELECT
  day_of_week,
  business_hour,
  ROUND(AVG(transactions), 1) AS avg_transactions
FROM __CATALOG__.__GOLD__.store_hourly_traffic
WHERE business_date BETWEEN DATE'2026-08-24' AND DATE'2026-10-19'
GROUP BY day_of_week, business_hour
ORDER BY
  CASE day_of_week
    WHEN 'Monday'    THEN 1  WHEN 'Tuesday'   THEN 2
    WHEN 'Wednesday' THEN 3  WHEN 'Thursday'  THEN 4
    WHEN 'Friday'    THEN 5  WHEN 'Saturday'  THEN 6
    WHEN 'Sunday'    THEN 7
  END,
  business_hour
