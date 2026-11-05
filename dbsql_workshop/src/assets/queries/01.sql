SELECT
  `store_name`,
  `region`,
  `store_format`,
  MEASURE(`net_sales`)        AS net_sales,
  MEASURE(`transactions`)     AS transactions,
  MEASURE(`avg_ticket`)       AS avg_ticket,
  MEASURE(`gross_margin_pct`) AS gross_margin_pct,
  MEASURE(`labor_cost_pct`)   AS labor_cost_pct,
  MEASURE(`digital_mix_pct`)  AS digital_mix_pct
FROM __CATALOG__.__GOLD__.store_sales_metrics
WHERE `business_date` BETWEEN DATE'2026-09-22' AND DATE'2026-10-19'
GROUP BY ALL
ORDER BY net_sales DESC
