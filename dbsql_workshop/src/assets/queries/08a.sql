SELECT
  member_id,
  first_name,
  loyalty_tier,
  customer_segment,
  visits,
  lifetime_net_sales,
  avg_ticket,
  days_since_last_visit,
  favorite_product
FROM __CATALOG__.__GOLD__.customer_360
WHERE customer_segment = 'Champion'
ORDER BY lifetime_net_sales DESC
LIMIT 20
