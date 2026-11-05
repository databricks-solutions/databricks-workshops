SELECT
  store_name,
  region,
  item_name,
  on_hand_qty,
  par_level,
  reorder_point,
  pct_of_par,
  stock_status
FROM __CATALOG__.__GOLD__.inventory_current_status
WHERE stock_status <> 'OK'
ORDER BY
  CASE stock_status WHEN 'REORDER NOW' THEN 1 WHEN 'LOW' THEN 2 END,
  pct_of_par
