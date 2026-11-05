SELECT
  c.member_id,
  c.first_name,
  c.loyalty_tier,
  c.lifetime_net_sales,
  s.store_name  AS home_store,
  s.region
FROM __CATALOG__.__GOLD__.customer_360 c
  JOIN __CATALOG__.__SILVER__.dim_store s
    ON c.home_store_id = s.store_id
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s.region
  ORDER BY c.lifetime_net_sales DESC
) = 1
ORDER BY c.lifetime_net_sales DESC
