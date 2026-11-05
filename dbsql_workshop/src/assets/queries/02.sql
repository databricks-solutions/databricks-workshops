SELECT
  `business_month`,
  `region`,
  MEASURE(`net_sales`) AS net_sales
FROM __CATALOG__.__GOLD__.store_sales_metrics
GROUP BY ALL
ORDER BY `business_month` ASC, `region`
