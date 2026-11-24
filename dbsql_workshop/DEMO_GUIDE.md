# Bean & Brew Coffee Co.: Databricks SQL demo

Bean & Brew is a fictional chain of 12 coffee shops in four regions, with data from January 2025 to "today",
2026-10-19. The deploy loads a governed silver layer; you build gold in SQL, including a model built with AI functions
on free-text customer comments, then use it through metric views, an AI/BI dashboard, a Genie agent and SQL alerts.

Seven stories are built into the data:

| | Scenario | Where it shows |
|---|---|---|
| (a) | Traffic at the Phoenix Downtown Drive-Thru (S007) roughly halves from 2026-10-15 | Traffic alert, dashboard, query 06, Genie |
| (b) | Oat milk at two stores and espresso beans at a third fall below their reorder point | Inventory alert, dashboard, query 07, Genie |
| (c) | Menu prices rise 5% on 2026-01-05 | Query 03, Genie |
| (d) | RiNo, Denver, opens 2026-03-01 and isn't a comp store yet | Query 03, Genie |
| (e) | Cold-drink sales roughly quadruple from May to August | Query 04, dashboard |
| (f) | Pumpkin Spice Latte sells only from September to November | Query 04, Genie |
| (g) | Customers at S007 report the drive-thru lane closed for construction; S004 and S008 customers report no oat milk | `feedback_insights`, query 09, Genie |

The notebook `Bean & Brew DBSQL Workshop` walks through every step, including the clicks for the steps that happen
outside it.

## 0. Connect

Attach the **Serverless Starter Warehouse** and run the Connect cell: it finds the catalog the bundle deployed into.

## 1. Compute

**SQL Warehouses → Serverless Starter Warehouse.** Serverless starts in seconds and bills only while it runs. Size sets
the speed of each query, scaling (min and max clusters) sets how many run at once, auto-stop ends idle time.
**Monitoring** shows load over time; **Connection details** has what BI tools connect with. **Create SQL warehouse**
shows every option; cancel it.

## 2. Catalog

**Catalog → `dbsql_workshop_silver`.** Typed, deduplicated tables with descriptions, primary and foreign keys, and tags
(email is tagged `contains_pii`). **Details** shows Predictive Optimization maintaining the tables. `customer_feedback`
holds free-text comments. Gold exists, empty.

## 3. Build gold

Five models at the grain business questions are asked: store-day, product-day, store-hour, current inventory and member.
A sixth, `feedback_insights`, reads 28 days of comments with `ai_analyze_sentiment`, `ai_classify` and `ai_extract`
(about 100 rows, about a minute). `INSERT OVERWRITE ... BY NAME` fills the deployed tables, so the dashboard, Genie and
the alerts have data as soon as it finishes. Any statement also runs in the **SQL Editor**, where Genie Code can explain
it.

## 4. Test gold

Net sales reconcile to the cent across silver and three gold grains, every scenario holds, and a fingerprint of the
deterministic tables matches the reference build. The AI test is informational. A failure stops the notebook;
re-running Step 3 fixes it.

## 5. Query gold

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
| 09 | `ai_summarize()` over this week's S007 comments: (g) |

In the **SQL Editor**: chart query 02, run it twice to hit the result cache, then open its query profile from
**Query History**. Query 08 calls `databricks-gpt-oss-20b`, one call per row on five rows; if that model isn't served, use one
listed under **Serving**.

## 6. Metric views

**Catalog → `dbsql_workshop_gold` → `store_sales_metrics`.** Each KPI is defined once (average ticket =
`SUM(net_sales) / SUM(transactions)`), and `MEASURE()` recalculates it correctly at any grouping. Synonyms ("revenue",
"AOV") help Genie. **Lineage** runs from silver through gold to the dashboard.

## 7. Dashboard

**Dashboards → `[dbsql_workshop] Bean & Brew Coffee Co - Store Operations`.** Four pages on the metric views: Executive
Overview, Products & Dayparts, Operations, Loyalty. Find the S007 dip on Operations, then ask Genie why.

## 8. Genie

**Genie → `[dbsql_workshop] Bean & Brew Coffee Ops Genie`.** Instructions carry the business rules (today is 2026-10-19,
comp stores, the 30% labor target, REORDER NOW, the price increase, where the customer feedback is), plus worked
examples, value matching and four benchmarks. Every answer shows its SQL. Try:

- Why is traffic down at the Phoenix Downtown Drive-Thru this week? (Agent mode.) Then: break it down by order channel.
- What are customers saying about the Phoenix Downtown Drive-Thru this week?
- Which stores need to reorder inventory right now?
- What are the busiest hours at Pike Place on Saturdays?
- How did average ticket at comp stores change after the January 2026 price increase?
- Which stores are over the 30% labor target in the last 28 days?

## 9. Alerts

**Alerts → `[dbsql_workshop] Bean & Brew - ...`.** Inventory below reorder point, store traffic drop, labor cost over
30%. All three fire on this data. They're weekly and created paused; resuming one emails the deployer.

## 10. Tear down

**Deployments → Destroy** (or `databricks bundle destroy`) removes the schemas, the setup job, the dashboard, the Genie
agent and the alerts. Delete by hand any saved SQL editor queries and the Git folder. The catalog and the warehouse
stay.

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
| The dashboard or Genie is empty | Run Step 3 |
| A test fails | Re-run Step 3 |
| The AI model cell fails | AI Functions aren't available in your region: skip the cell, the rest works |
| Query 08 can't find its model | Use a model listed under **Serving** |
| An alert doesn't fire | It starts paused: resume it |
| Something was changed or deleted | Destroy, deploy again, re-run the notebook |
| The deploy failed | See "Common errors" in [USER_GUIDE.md](../USER_GUIDE.md) |
