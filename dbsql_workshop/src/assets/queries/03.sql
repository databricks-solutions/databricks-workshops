WITH y2025 AS (
  SELECT
    MEASURE(`net_sales`)  AS net_sales,
    MEASURE(`avg_ticket`) AS avg_ticket
  FROM __CATALOG__.__GOLD__.store_sales_metrics
  WHERE `is_comp_store` = true
    AND `business_date` >= DATE'2025-01-01'
    AND `business_date` <  DATE'2025-10-01'
  GROUP BY ALL
),
y2026 AS (
  SELECT
    MEASURE(`net_sales`)  AS net_sales,
    MEASURE(`avg_ticket`) AS avg_ticket
  FROM __CATALOG__.__GOLD__.store_sales_metrics
  WHERE `is_comp_store` = true
    AND `business_date` >= DATE'2026-01-01'
    AND `business_date` <  DATE'2026-10-01'
  GROUP BY ALL
)
SELECT
  y2025.net_sales                                                 AS net_sales_2025,
  y2026.net_sales                                                 AS net_sales_2026,
  (y2026.net_sales - y2025.net_sales) / y2025.net_sales * 100     AS yoy_net_sales_pct,
  y2025.avg_ticket                                                AS avg_ticket_2025,
  y2026.avg_ticket                                                AS avg_ticket_2026,
  (y2026.avg_ticket - y2025.avg_ticket) / y2025.avg_ticket * 100  AS yoy_avg_ticket_pct
FROM y2025 CROSS JOIN y2026
