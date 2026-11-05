SELECT
  CASE
    WHEN `business_date` >= DATE'2026-06-01'
     AND `business_date` <  DATE'2026-09-01' THEN 'Summer (Jun-Aug)'
    ELSE 'Fall (Sep-Oct)'
  END                        AS season,
  `category`,
  `product_name`,
  MEASURE(`units_sold`)      AS units_sold,
  MEASURE(`net_sales`)       AS net_sales
FROM __CATALOG__.__GOLD__.product_sales_metrics
WHERE `business_date` >= DATE'2026-06-01'
  AND `business_date` <= DATE'2026-10-19'
GROUP BY ALL
ORDER BY season, net_sales DESC
