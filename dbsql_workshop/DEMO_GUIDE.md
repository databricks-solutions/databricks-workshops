# Bean & Brew Coffee Co.: Databricks SQL demo

Bean & Brew is a fictional chain of 12 coffee shops in four regions, with data from January 2025 to "today",
2026-10-19. Raw register data becomes governed tables, metric views, an AI/BI dashboard, a Genie agent and SQL alerts.

Six stories are built into the data:

| | Scenario | Where it shows |
|---|---|---|
| (a) | Traffic at the Phoenix Downtown Drive-Thru (S007) roughly halves from 2026-10-15 | Traffic alert, dashboard, query 06, Genie |
| (b) | Oat milk at two stores and espresso beans at a third fall below their reorder point | Inventory alert, dashboard, query 07, Genie |
| (c) | Menu prices rise 5% on 2026-01-05 | Query 03, Genie |
| (d) | RiNo, Denver, opens 2026-03-01 and isn't a comp store yet | Query 03, Genie |
| (e) | Cold-drink sales roughly quadruple from May to August | Query 04, dashboard |
| (f) | Pumpkin Spice Latte sells only from September to November | Query 04, Genie |

## 0. Connect

Finds the catalog the bundle deployed into and makes it the session default.

## 1. Bronze

Six raw tables, about 1.5M register tickets, with deliberate defects: re-sent duplicate tickets, lower-case store IDs,
line items as JSON text, timestamps as text. Raw data stays unchanged, so every number traces back to its source.

## 2. Silver

Fills the cleaned tables: types cast, IDs normalized, duplicates removed, JSON line items exploded, voided tickets
excluded. The keys were declared at deploy, and `INSERT OVERWRITE ... BY NAME` keeps them, so cells are safe to re-run.
In **Catalog**, `fact_transactions` → **Lineage** shows the path from bronze to gold.

## 3. Gold

Six tables at the grain business questions are asked: store-day, product-day, store-hour, current inventory, member,
and the promo list. Net sales reconcile to the cent across silver and three gold grains. The dashboard and Genie now
have data.

## 4. Semantic layer and governance

- **Metric views** define each KPI once (average ticket = `SUM(net_sales) / SUM(transactions)`), and `MEASURE()`
  recalculates it correctly at any grouping. Synonyms ("revenue", "AOV") help Genie.
- **Keys** tell Genie and the metric views how tables join.
- **Comments and tags** document the columns, classify layers and owners, and flag email as PII. If governed tag
  policies reject the tags, skip that cell.

## 5. Validate

Checks the cleanup, the reconciliation, every scenario and a data fingerprint, and stops the notebook if anything
failed. Re-running Steps 2 and 3 fixes a failure.

## 6. Modern SQL

| Query | Shows |
|---|---|
| 01 | `MEASURE()` on a metric view |
| 02 | Monthly trend by region (chart it) |
| 03 | Comp-store year over year: scenarios (c) and (d) |
| 04 | Seasonality: (e) and (f) |
| 05 | Hour × weekday traffic (as a heatmap) |
| 06 | 7-day moving average: (a) |
| 07 | The inventory alert's query: (b) |
| 08 | `QUALIFY`, and `ai_query()` classifying five menu items |
| 09 | Data quality, bronze vs silver |

Any of them runs in the SQL editor too: attach the same warehouse, and pick the deployed catalog if it isn't the
default. Query 08 selects its five rows before calling `databricks-gpt-oss-20b`, one call per row; if that model isn't
served, use one listed under **Serving**.

## 7. Dashboard, Genie, alerts

All three are named `[dbsql_workshop] ...`: search the workspace for that prefix.

- **Dashboard:** four pages on the metric views: Executive Overview, Products & Dayparts, Operations, Loyalty. Find the
  S007 dip on Operations, then ask Genie why.
- **Genie:** instructions carry the business rules (today is 2026-10-19, comp stores, the 30% labor target,
  REORDER NOW, the price increase), plus worked examples, value matching and four benchmarks. Every answer shows its
  SQL. Try:
  - What were net sales by region last month?
  - Which stores need to reorder inventory right now?
  - What are the busiest hours at Pike Place on Saturdays?
  - How did average ticket at comp stores change after the January 2026 price increase?
  - Why is traffic down at the Phoenix Downtown Drive-Thru this week? Then: break it down by order channel.
  - Which stores are over the 30% labor target in the last 28 days?
- **Alerts:** inventory below reorder point, store traffic drop, labor cost over 30%. All three fire on this data.
  They're weekly and created paused; resuming one emails the deployer.

## 8. Visual data prep (optional)

Builds the promo list without code:

1. **New → Visual data prep**, saved in your home folder.
2. Sources `dbsql_workshop_gold.customer_360` and `dbsql_workshop_silver.dim_store`, joined on
   `home_store_id = store_id`.
3. Filter to Champion and Loyal Regular members in the Gold and Reserve tiers.
4. Add `promo_offer`: `Free Nitro Upgrade` for "Cold Brew" favorites, `Free Oat Milk Swap` for "Latte",
   otherwise `$2 Off Pastry`.

The result matches `dbsql_workshop_gold.loyalty_promo_targets`. Delete the flow afterwards: it isn't part of the bundle.

## 9. Tear down

**Deployments → Destroy** (or `databricks bundle destroy`) removes the schemas, the setup job, the dashboard, the
Genie agent and the alerts. Delete by hand the Step 8 flow, any saved queries and the Git folder. The catalog and the
warehouse stay.

## Extras (admin)

- **Query history:** `system.query.history` shows how the demo data was used:

  ```sql
  SELECT statement_type, count(*) AS queries, round(percentile(total_duration_ms, 0.5) / 1000, 2) AS median_sec
  FROM system.query.history
  WHERE start_time >= current_timestamp() - INTERVAL 7 DAYS AND statement_text ILIKE '%dbsql_workshop_%'
  GROUP BY ALL
  ORDER BY queries DESC
  ```

- **Lakehouse RT (Beta):** enable it in the workspace previews, create a Real-Time SQL warehouse, and switch the
  dashboard or a query to it for sub-second reads. Keep Genie on the regular warehouse. Delete the Real-Time
  warehouse afterwards.

## Troubleshooting

| Symptom | Fix |
|---|---|
| The dashboard or Genie is empty | Run Steps 2 and 3 |
| A validation check fails | Re-run Steps 2 and 3 |
| The tags cell fails | Skip it |
| Query 08 can't find its model | Use a model listed under **Serving** |
| An alert doesn't fire | It starts paused: resume it |
| Something was changed or deleted | Destroy, deploy again, re-run the notebook |
| The deploy failed | See "Common errors" in [USER_GUIDE.md](../USER_GUIDE.md) |
