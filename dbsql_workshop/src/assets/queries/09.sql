SELECT
  '__BRONZE__.raw_pos_transactions' AS layer,
  COUNT(*)                                                      AS total_rows,
  COUNT(*) - COUNT(DISTINCT transaction_id)                     AS duplicate_txn_ids,
  SUM(CASE WHEN store_id <> UPPER(store_id) THEN 1 ELSE 0 END) AS lowercase_store_ids
FROM __CATALOG__.__BRONZE__.raw_pos_transactions
UNION ALL
SELECT
  '__SILVER__.fact_transactions',
  COUNT(*),
  COUNT(*) - COUNT(DISTINCT transaction_id),
  SUM(CASE WHEN store_id <> UPPER(store_id) THEN 1 ELSE 0 END)
FROM __CATALOG__.__SILVER__.fact_transactions
