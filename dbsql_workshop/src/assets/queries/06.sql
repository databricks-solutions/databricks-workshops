SELECT
  business_date,
  store_name,
  transactions,
  ROUND(AVG(transactions) OVER (
    ORDER BY business_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ), 1) AS moving_avg_7d
FROM __CATALOG__.__GOLD__.store_daily_performance
WHERE store_id = 'S007'
  AND business_date BETWEEN DATE'2026-09-15' AND DATE'2026-10-19'
ORDER BY business_date
