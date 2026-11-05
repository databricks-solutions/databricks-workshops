SELECT
  member_id,
  first_name,
  favorite_product,
  ai_query(
    '__LLM_ENDPOINT__',
    'Classify this coffee-shop menu item into exactly one word: '
    || 'Hot, Cold, Food, or Retail. Item: ' || favorite_product
  ) AS ai_category
FROM __CATALOG__.__GOLD__.customer_360
WHERE customer_segment = 'Champion'
ORDER BY lifetime_net_sales DESC
LIMIT 5
