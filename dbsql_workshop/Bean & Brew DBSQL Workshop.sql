-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Bean & Brew Coffee Co.: Databricks SQL workshop
-- MAGIC
-- MAGIC A 12-store coffee chain, 2025-01-01 to 2026-10-19 (treated as "today"). The deploy loaded the **silver** layer (clean
-- MAGIC sales, inventory, labor, loyalty and customer-feedback data) and created the **gold** tables empty, with an AI/BI
-- MAGIC dashboard, a Genie agent and three alerts on top. You'll tour the SQL warehouse and the catalog, build gold in SQL
-- MAGIC (including AI functions on free text), test it, query it, then use the dashboard, Genie and the alerts.
-- MAGIC
-- MAGIC Cells run SQL here. Steps marked **In the workspace** happen elsewhere: follow the numbered clicks, then come back to
-- MAGIC this tab. Open things in a new browser tab (Ctrl/Cmd-click) to keep your place.
-- MAGIC
-- MAGIC | Step | Where | What you do |
-- MAGIC |---|---|---|
-- MAGIC | 0 | This notebook | Attach the SQL warehouse and connect to the catalog the bundle deployed into |
-- MAGIC | 1 | SQL warehouses | Tour your compute: serverless, size, scaling, auto-stop, connection details |
-- MAGIC | 2 | Catalog | Explore the governed silver layer: comments, keys, tags, lineage, free-text feedback |
-- MAGIC | 3 | This notebook | Build the gold models, including one built with AI functions |
-- MAGIC | 4 | This notebook | Test gold: every demo scenario, plus a data fingerprint |
-- MAGIC | 5 | This notebook, SQL editor | Query gold: metric views, window functions, `QUALIFY`, AI functions; query profile |
-- MAGIC | 6 | Catalog | Metric views: one definition of every KPI |
-- MAGIC | 7 | Dashboards | The store operations dashboard |
-- MAGIC | 8 | Genie | Ask why traffic dropped at the Phoenix drive-thru |
-- MAGIC | 9 | Alerts | Inventory, traffic and labor alerts |
-- MAGIC | 10 | Deployments | Tear down |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 0: Connect
-- MAGIC
-- MAGIC 1. At the top right of this notebook, open the compute selector (it may say **Connect**) and choose
-- MAGIC    **Serverless Starter Warehouse** under SQL warehouses. Serverless notebook compute works too.
-- MAGIC 2. Run the cell below. It finds the catalog the bundle deployed into and makes it this session's default, so every
-- MAGIC    query can name the schemas `dbsql_workshop_silver` and `dbsql_workshop_gold` without a catalog.

-- COMMAND ----------

-- Find the catalog the bundle deployed into, and make it this session's default.
-- Every query below names the schemas (dbsql_workshop_silver, dbsql_workshop_gold) without a catalog.
DECLARE OR REPLACE VARIABLE demo_catalog STRING;
SET VAR demo_catalog = (
  SELECT CASE WHEN count(*) = 1 THEN max(catalog_name)
              WHEN count(*) = 0 THEN raise_error('No dbsql_workshop_gold schema found. Deploy the dbsql_workshop bundle first (see README.md).')
              ELSE raise_error('dbsql_workshop_gold exists in ' || count(*) || ' catalogs (' || array_join(collect_list(catalog_name), ', ')
                               || '). Replace this cell with: USE CATALOG <the catalog you deployed to>')
         END
  FROM system.information_schema.schemata
  WHERE schema_name = 'dbsql_workshop_gold');
USE CATALOG IDENTIFIER(demo_catalog);
SELECT current_catalog() AS catalog, 'dbsql_workshop_silver, dbsql_workshop_gold' AS schemas

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 1: Compute, your SQL warehouse
-- MAGIC
-- MAGIC **In the workspace.** Everything in this workshop runs on one serverless SQL warehouse.
-- MAGIC
-- MAGIC 1. In the left sidebar, under **SQL**, click **SQL Warehouses**.
-- MAGIC 2. Click **Serverless Starter Warehouse**. On **Overview**, note:
-- MAGIC    - **Type: Serverless.** It starts in seconds and you pay only while it runs.
-- MAGIC    - **Cluster size.** How much compute each query gets (2X-Small to 4X-Large).
-- MAGIC    - **Auto stop.** Minutes of idle time before it shuts itself down.
-- MAGIC    - **Scaling.** The minimum and maximum number of clusters: more clusters serve more people at once.
-- MAGIC 3. Click the **Monitoring** tab: running clusters and queries over time, including what this notebook just ran.
-- MAGIC 4. Click the **Connection details** tab: the server hostname and HTTP path that Power BI, Tableau, dbt and JDBC/ODBC
-- MAGIC    clients connect with.
-- MAGIC 5. Go back to **SQL Warehouses** and click **Create SQL warehouse**. Look at the options, then click **Cancel**:
-- MAGIC    nothing is created.
-- MAGIC
-- MAGIC Why: size sets how fast each query runs, scaling sets how many run at once, and auto-stop keeps idle time free.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 2: Catalog, the governed silver layer
-- MAGIC
-- MAGIC **In the workspace.** The deploy loaded silver the way a data engineering team would hand it over: typed, deduplicated
-- MAGIC and documented.
-- MAGIC
-- MAGIC 1. In the left sidebar, click **Catalog**, then open your catalog (`workspace` unless you chose another) and the
-- MAGIC    schema `dbsql_workshop_silver`.
-- MAGIC 2. Open `fact_transactions`:
-- MAGIC    - **Columns:** the table and column descriptions, and the primary and foreign keys.
-- MAGIC    - **Sample Data:** one row per register ticket.
-- MAGIC    - **Details:** the owner and table properties, including whether **Predictive optimization** maintains the
-- MAGIC      table for you.
-- MAGIC    - **Lineage:** where the table's data goes. After Step 3 it shows the gold models and the dashboard.
-- MAGIC 3. Open `dim_loyalty_member`: the `email` column carries the tag `contains_pii`.
-- MAGIC 4. Open `customer_feedback`: star ratings with free-text comments from app reviews and receipt surveys.
-- MAGIC 5. Open the schema `dbsql_workshop_gold`: the tables are there, empty. You build them in Step 3.
-- MAGIC
-- MAGIC Why: descriptions, keys and tags are what people search for, and what Genie reads to answer correctly.
-- MAGIC
-- MAGIC The cell below shows this week's comments about the Phoenix Downtown Drive-Thru (S007). Keep them in mind for Step 8.

-- COMMAND ----------

-- Free text in silver: what customers wrote about the Phoenix Downtown Drive-Thru (S007) this week
SELECT submitted_date, channel, rating, comment
FROM dbsql_workshop_silver.customer_feedback
WHERE store_id = 'S007' AND submitted_date >= DATE'2026-10-13'
ORDER BY submitted_date DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 3: Build gold
-- MAGIC
-- MAGIC Each statement builds one model at the grain a business question needs, from silver. `INSERT OVERWRITE ... BY NAME`
-- MAGIC fills the tables the deploy created, so the dashboard, the Genie agent and the alerts read them as soon as this step
-- MAGIC finishes. Re-running a cell replaces the rows and keeps everything built on the table.
-- MAGIC
-- MAGIC **Try it in the SQL editor:** copy a statement, then in the left sidebar click **SQL Editor**, open a new tab with
-- MAGIC **+**, and paste it. Highlight part of it and ask the assistant (**Genie Code**) to explain it.

-- COMMAND ----------

-- Gold models: one table per business grain

-- store_daily_performance: one row per store and day (traffic, sales, margin, loyalty mix, labor)
INSERT OVERWRITE dbsql_workshop_gold.store_daily_performance BY NAME
WITH t AS (
  SELECT store_id, business_date,
    count(*) transactions, sum(item_qty) items_sold, sum(gross_sales) gross_sales, sum(discount_amount) discount_amount,
    sum(net_sales) net_sales, sum(cost_of_goods) cost_of_goods,
    count_if(is_loyalty_transaction) loyalty_transactions,
    count_if(order_channel IN ('Mobile App','Delivery')) digital_transactions,
    count_if(order_channel = 'Drive-Thru') drive_thru_transactions
  FROM dbsql_workshop_silver.fact_transactions WHERE NOT is_voided GROUP BY ALL)
SELECT t.business_date, t.store_id, s.store_name, s.city, s.state, s.region, s.store_format,
  datediff(t.business_date, s.open_date) >= 365 AS is_comp_store,
  t.transactions, t.items_sold, t.gross_sales, t.discount_amount, t.net_sales, t.cost_of_goods,
  t.net_sales - t.cost_of_goods AS gross_profit,
  cast(round(t.net_sales / t.transactions, 2) AS DECIMAL(10,2)) AS avg_ticket,
  t.loyalty_transactions, t.digital_transactions, t.drive_thru_transactions,
  l.worked_hours AS labor_hours, l.labor_cost,
  cast(round(l.labor_cost / t.net_sales, 4) AS DECIMAL(8,4)) AS labor_cost_pct
FROM t JOIN dbsql_workshop_silver.dim_store s USING (store_id)
LEFT JOIN dbsql_workshop_silver.fact_labor_daily l ON l.store_id = t.store_id AND l.business_date = t.business_date;

-- product_sales_daily: one row per day, store and product
INSERT OVERWRITE dbsql_workshop_gold.product_sales_daily BY NAME
SELECT f.business_date, f.store_id, s.store_name, s.region, f.product_id, p.product_name, p.category, p.serving_type, p.is_beverage,
  sum(f.quantity) units_sold, sum(f.gross_sales) gross_sales, sum(f.discount_amount) discount_amount, sum(f.net_sales) net_sales,
  sum(f.cost_of_goods) cost_of_goods, sum(f.net_sales) - sum(f.cost_of_goods) gross_profit
FROM dbsql_workshop_silver.fact_sales_line_items f JOIN dbsql_workshop_silver.dim_product p USING (product_id) JOIN dbsql_workshop_silver.dim_store s USING (store_id)
GROUP BY ALL;

-- store_hourly_traffic: one row per store, day and hour
INSERT OVERWRITE dbsql_workshop_gold.store_hourly_traffic BY NAME
SELECT store_id, business_date, business_hour, date_format(business_date,'EEEE') AS day_of_week,
  count(*) transactions, sum(net_sales) net_sales, count_if(order_channel = 'Mobile App') mobile_transactions
FROM dbsql_workshop_silver.fact_transactions WHERE NOT is_voided GROUP BY ALL;

-- inventory_current_status: the latest count per store and stock item, with a stock status
INSERT OVERWRITE dbsql_workshop_gold.inventory_current_status BY NAME
WITH latest AS (SELECT * FROM dbsql_workshop_silver.fact_inventory_daily QUALIFY row_number() OVER (PARTITION BY store_id, item_code ORDER BY count_date DESC) = 1)
SELECT l.store_id, s.store_name, s.region, l.count_date AS as_of_date, l.item_code, l.item_name, l.unit_of_measure,
  l.on_hand_qty, l.par_level, l.reorder_point, cast(round(l.on_hand_qty / l.par_level, 3) AS DECIMAL(6,3)) AS pct_of_par,
  l.inventory_value,
  CASE WHEN l.on_hand_qty < l.reorder_point THEN 'REORDER NOW' WHEN l.on_hand_qty < l.reorder_point * 1.5 THEN 'LOW' ELSE 'OK' END AS stock_status
FROM latest l JOIN dbsql_workshop_silver.dim_store s USING (store_id);

-- customer_360: one row per loyalty member, with lifetime value and a value segment.
-- mode(x, true) breaks ties on the lowest value, so favorite store/product are stable across runs
INSERT OVERWRITE dbsql_workshop_gold.customer_360 BY NAME
WITH tx AS (
  SELECT loyalty_member_id member_id, count(*) visits, sum(net_sales) lifetime_net_sales, min(business_date) first_visit_date, max(business_date) last_visit_date,
    mode(store_id, true) favorite_store_id, count_if(order_channel = 'Mobile App') mobile_visits
  FROM dbsql_workshop_silver.fact_transactions WHERE NOT is_voided AND loyalty_member_id IS NOT NULL GROUP BY 1),
fav AS (
  SELECT loyalty_member_id member_id, mode(p.product_name, true) favorite_product
  FROM dbsql_workshop_silver.fact_sales_line_items f JOIN dbsql_workshop_silver.dim_product p USING (product_id) WHERE loyalty_member_id IS NOT NULL GROUP BY 1),
base AS (
  SELECT m.member_id, m.first_name, m.last_name, m.loyalty_tier, m.signup_date, m.home_store_id,
    tx.visits, tx.lifetime_net_sales, cast(round(tx.lifetime_net_sales / tx.visits, 2) AS DECIMAL(10,2)) avg_ticket,
    tx.first_visit_date, tx.last_visit_date, datediff(DATE'2026-10-19', tx.last_visit_date) AS days_since_last_visit,
    tx.favorite_store_id, fav.favorite_product, round(tx.mobile_visits / tx.visits, 3) AS mobile_share,
    percent_rank() OVER (ORDER BY tx.lifetime_net_sales DESC, m.member_id) pr
  FROM dbsql_workshop_silver.dim_loyalty_member m LEFT JOIN tx USING (member_id) LEFT JOIN fav USING (member_id))
SELECT * EXCEPT (pr),
  CASE WHEN pr < 0.10 THEN 'Champion' WHEN pr < 0.40 THEN 'Loyal Regular' WHEN pr < 0.80 THEN 'Occasional' ELSE 'Light' END AS customer_segment
FROM base;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **An AI model, in SQL.** `feedback_insights` turns the free-text comments into columns you can group and filter:
-- MAGIC
-- MAGIC - `ai_analyze_sentiment()`: positive, negative, neutral or mixed
-- MAGIC - `ai_classify()`: one topic from a fixed list
-- MAGIC - `ai_extract()`: the product mentioned and the issue, from a small JSON schema
-- MAGIC
-- MAGIC The query picks the rows first (the last 28 days, about 100 comments), so each function runs about 100 times. It takes
-- MAGIC about a minute. If AI Functions aren't available in your workspace's region, skip this cell; everything else works
-- MAGIC without it.

-- COMMAND ----------

-- feedback_insights: one row per customer comment from the last 28 days, read with AI functions in plain SQL.
-- Pick the rows first: every AI function runs once per row, so the CTE keeps it to about 100 recent comments.
INSERT OVERWRITE dbsql_workshop_gold.feedback_insights BY NAME
WITH recent AS (
  SELECT f.feedback_id, f.store_id, s.store_name, f.submitted_date, f.channel, f.rating, f.comment
  FROM dbsql_workshop_silver.customer_feedback f JOIN dbsql_workshop_silver.dim_store s USING (store_id)
  WHERE f.submitted_date BETWEEN DATE'2026-09-22' AND DATE'2026-10-19'),
enriched AS (
  SELECT *,
    ai_analyze_sentiment(comment) AS sentiment,
    ai_classify(comment,
      '["Speed of service", "Store access", "Product availability", "Price", "Product quality", "Staff", "Store experience"]',
      map('version', '2.0')) AS topic_result,
    ai_extract(comment,
      '{"product": {"type": "string", "description": "Menu item the customer mentions, if any"},
        "issue": {"type": "string", "description": "The problem the customer describes, in a few words, or null if there is none"}}',
      map('version', '2.0')) AS extract_result
  FROM recent)
SELECT feedback_id, store_id, store_name, submitted_date, channel, rating, comment, sentiment,
  variant_get(topic_result, '$.response[0]', 'STRING') AS topic,
  variant_get(extract_result, '$.response.product', 'STRING') AS product_mentioned,
  variant_get(extract_result, '$.response.issue', 'STRING') AS issue
FROM enriched;

-- What the AI functions found, store by store
SELECT store_name,
  count(*)                          AS comments,
  count_if(sentiment = 'negative')  AS negative,
  mode(topic)                       AS top_topic,
  max_by(issue, submitted_date) FILTER (WHERE sentiment = 'negative' AND issue IS NOT NULL) AS latest_complaint
FROM dbsql_workshop_gold.feedback_insights
GROUP BY store_name
ORDER BY negative DESC, store_name

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 4: Test gold
-- MAGIC
-- MAGIC Data tests for the gold you just built: net sales reconcile to the cent across three grains, every demo scenario holds,
-- MAGIC and a fingerprint of the deterministic tables matches the reference build. Test 11 checks the AI model and is
-- MAGIC informational, because model output varies from run to run. The second cell stops the notebook if a required test
-- MAGIC failed; re-running Step 3 fixes it.

-- COMMAND ----------

-- Data tests: every demo scenario the dashboard, Genie and alerts rely on, checked against the gold you just built.
-- FAIL means a model above didn't run or was edited. The AI test is informational: model output varies run to run.
CREATE OR REPLACE TEMPORARY VIEW workshop_checks AS
WITH
rec AS (SELECT (SELECT sum(net_sales) FROM dbsql_workshop_silver.fact_transactions WHERE NOT is_voided) AS silver,
               (SELECT sum(net_sales) FROM dbsql_workshop_gold.store_daily_performance) AS store_day,
               (SELECT sum(net_sales) FROM dbsql_workshop_gold.product_sales_daily) AS product_day,
               (SELECT sum(net_sales) FROM dbsql_workshop_gold.store_hourly_traffic) AS store_hour),
traffic AS (SELECT avg(CASE WHEN business_date >= DATE'2026-10-15' THEN transactions END)
                 / avg(CASE WHEN business_date BETWEEN DATE'2026-09-15' AND DATE'2026-10-14' THEN transactions END) - 1 AS chg
            FROM dbsql_workshop_gold.store_daily_performance WHERE store_id = 'S007'),
reorder AS (SELECT array_join(array_sort(collect_list(concat(store_id, ' ', item_name))), ', ') AS items
            FROM dbsql_workshop_gold.inventory_current_status WHERE stock_status = 'REORDER NOW'),
ticket AS (SELECT max(CASE WHEN period = 'before' THEN t END) AS before, max(CASE WHEN period = 'after' THEN t END) AS after
           FROM (SELECT CASE WHEN business_date >= DATE'2026-01-05' THEN 'after' ELSE 'before' END AS period, MEASURE(avg_ticket) AS t
                 FROM dbsql_workshop_gold.store_sales_metrics
                 WHERE is_comp_store AND business_date BETWEEN DATE'2025-10-05' AND DATE'2026-04-05' GROUP BY ALL)),
rino AS (SELECT min(business_date) AS first_day, count_if(is_comp_store) AS comp_days
         FROM dbsql_workshop_gold.store_daily_performance WHERE store_id = 'S012'),
cold AS (SELECT avg(CASE WHEN month(m) BETWEEN 5 AND 8 THEN u END) / avg(CASE WHEN month(m) NOT BETWEEN 5 AND 8 THEN u END) AS ratio
         FROM (SELECT date_trunc('MONTH', business_date) AS m, sum(units_sold) AS u FROM dbsql_workshop_gold.product_sales_daily
               WHERE category = 'Cold Drinks' AND year(business_date) = 2025 GROUP BY 1)),
psl AS (SELECT array_sort(collect_set(month(business_date))) AS months
        FROM dbsql_workshop_gold.product_sales_daily WHERE product_name = 'Pumpkin Spice Latte'),
labor AS (SELECT array_join(array_sort(collect_list(store_id)), ', ') AS stores
          FROM (SELECT store_id FROM dbsql_workshop_gold.store_daily_performance
                WHERE business_date BETWEEN DATE'2026-09-22' AND DATE'2026-10-19'
                GROUP BY ALL HAVING sum(labor_cost) / sum(net_sales) > 0.30)),
seg AS (SELECT count(*) AS members, count_if(customer_segment = 'Champion') AS champions FROM dbsql_workshop_gold.customer_360),
fb AS (SELECT count(*) AS n, count_if(sentiment IS NOT NULL AND topic IS NOT NULL) AS enriched,
              count_if(store_id = 'S007' AND submitted_date >= DATE'2026-10-15') AS s007,
              count_if(store_id = 'S007' AND submitted_date >= DATE'2026-10-15' AND sentiment = 'negative') AS s007_negative
       FROM dbsql_workshop_gold.feedback_insights),
fp AS (SELECT sha2(concat_ws('|',
         (SELECT array_join(sort_array(collect_list(concat_ws(',', store_id, business_date, transactions, net_sales, labor_cost))), ';') FROM dbsql_workshop_gold.store_daily_performance),
         (SELECT array_join(sort_array(collect_list(concat_ws(',', member_id, visits, lifetime_net_sales, favorite_product, customer_segment))), ';') FROM dbsql_workshop_gold.customer_360),
         (SELECT array_join(sort_array(collect_list(concat_ws(',', store_id, item_code, on_hand_qty, stock_status))), ';') FROM dbsql_workshop_gold.inventory_current_status)), 256) AS fingerprint)
SELECT 1 AS n, 'Net sales reconcile across silver and three gold grains' AS check_name,
       rec.silver = rec.store_day AND rec.silver = rec.product_day AND rec.silver = rec.store_hour AS ok, true AS required,
       format_number(rec.silver, 2) AS detail FROM rec
UNION ALL SELECT 2, '(a) S007 traffic drops from 2026-10-15', traffic.chg < -0.4, true,
       format_string('%+.1f%% vs prior 30 days', traffic.chg * 100) FROM traffic
UNION ALL SELECT 3, '(b) Exactly 3 items to reorder', reorder.items = 'S004 Oat Milk, S008 Oat Milk, S010 Espresso Beans', true, reorder.items FROM reorder
UNION ALL SELECT 4, '(c) Average ticket rises after the 2026-01-05 price increase', ticket.after > ticket.before, true,
       format_string('%.2f -> %.2f', cast(ticket.before AS DOUBLE), cast(ticket.after AS DOUBLE)) FROM ticket
UNION ALL SELECT 5, '(d) RiNo opened 2026-03-01, no comp days yet', rino.first_day = DATE'2026-03-01' AND rino.comp_days = 0, true,
       format_string('first day %s, %d comp days', cast(rino.first_day AS STRING), rino.comp_days) FROM rino
UNION ALL SELECT 6, '(e) Cold drinks peak May-Aug', cold.ratio > 3, true, format_string('%.1fx the rest of 2025', cold.ratio) FROM cold
UNION ALL SELECT 7, '(f) Pumpkin Spice Latte only Sep-Nov', size(psl.months) > 0 AND size(array_except(psl.months, array(9, 10, 11))) = 0, true,
       'months ' || array_join(psl.months, ', ') FROM psl
UNION ALL SELECT 8, 'Labor alert has a store over 30%', labor.stores <> '', true, labor.stores FROM labor
UNION ALL SELECT 9, 'Customer 360 covers every member', seg.members = 6000 AND seg.champions > 0, true,
       format_string('%,d members, %,d Champions', seg.members, seg.champions) FROM seg
UNION ALL SELECT 10, 'Data fingerprint matches the reference build',
       fp.fingerprint = 'a39d45e4a4e89d5f1806e891ac587466f005a4490c03da241d298e21790a4f68', true, fp.fingerprint FROM fp
UNION ALL SELECT 11, '(g) AI: every recent comment has a sentiment and a topic', fb.n > 0 AND fb.enriched = fb.n, false,
       format_string('%d comments enriched; S007 since 2026-10-15: %d of %d negative', fb.enriched, fb.s007_negative, fb.s007) FROM fb;

SELECT n, check_name, CASE WHEN ok THEN 'PASS' WHEN required THEN 'FAIL' ELSE 'CHECK' END AS result, detail
FROM workshop_checks ORDER BY n

-- COMMAND ----------

-- Stops the notebook here if any check failed
SELECT CASE WHEN count_if(required AND NOT ok) > 0
            THEN raise_error('Validation failed: ' || array_join(collect_list(CASE WHEN required AND NOT ok THEN check_name END), '; '))
            ELSE 'All ' || count_if(required) || ' required checks passed' END AS validation
FROM workshop_checks

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 5: Query gold
-- MAGIC
-- MAGIC One feature per query. Every one also runs in the **SQL Editor** on the same warehouse (pick the deployed catalog in the
-- MAGIC editor's catalog selector if it isn't your default).
-- MAGIC
-- MAGIC | Query | Shows |
-- MAGIC |---|---|
-- MAGIC | 01 | `MEASURE()` on a metric view: a store scorecard |
-- MAGIC | 02 | A monthly trend by region: chart it |
-- MAGIC | 03 | Comp-store year over year: scenarios (c) and (d) |
-- MAGIC | 04 | Seasonality: scenarios (e) and (f) |
-- MAGIC | 05 | Traffic by hour and weekday: chart it as a heatmap |
-- MAGIC | 06 | A 7-day moving average over the S007 drop: scenario (a) |
-- MAGIC | 07 | The inventory alert's query: scenario (b) |
-- MAGIC | 08 | `QUALIFY`, and `ai_query()` classifying five menu items with an open model |
-- MAGIC | 09 | `ai_summarize()`: what customers said about S007 this week, in one model call |
-- MAGIC
-- MAGIC **In the SQL editor:**
-- MAGIC
-- MAGIC 1. Paste query 02 into a new tab and run it. Add a visualization to the results (the **+** next to the results
-- MAGIC    tab) and make a line chart: `business_month` on X, `net_sales` on Y, grouped by `region`.
-- MAGIC 2. Run it again. The second run returns from the result cache.
-- MAGIC 3. In the left sidebar, under **SQL**, click **Query History**. Click the query to open its details, then open its
-- MAGIC    **query profile**: time spent in each step, rows read, and how much was pruned.

-- COMMAND ----------

-- 01 - Store Scorecard (Trailing 28 Days)
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
FROM dbsql_workshop_gold.store_sales_metrics
WHERE `business_date` BETWEEN DATE'2026-09-22' AND DATE'2026-10-19'
GROUP BY ALL
ORDER BY net_sales DESC

-- COMMAND ----------

-- 02 - Monthly Net Sales Trend by Region
SELECT
  `business_month`,
  `region`,
  MEASURE(`net_sales`) AS net_sales
FROM dbsql_workshop_gold.store_sales_metrics
GROUP BY ALL
ORDER BY `business_month` ASC, `region`

-- COMMAND ----------

-- 03 - YoY Comp Store Sales Comparison
WITH y2025 AS (
  SELECT
    MEASURE(`net_sales`)  AS net_sales,
    MEASURE(`avg_ticket`) AS avg_ticket
  FROM dbsql_workshop_gold.store_sales_metrics
  WHERE `is_comp_store` = true
    AND `business_date` >= DATE'2025-01-01'
    AND `business_date` <  DATE'2025-10-01'
  GROUP BY ALL
),
y2026 AS (
  SELECT
    MEASURE(`net_sales`)  AS net_sales,
    MEASURE(`avg_ticket`) AS avg_ticket
  FROM dbsql_workshop_gold.store_sales_metrics
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

-- COMMAND ----------

-- 04 - Product Mix Summer vs Fall
SELECT
  CASE
    WHEN `business_date` >= DATE'2026-06-01'
     AND `business_date` <  DATE'2026-09-01' THEN 'Summer (Jun-Aug)'
    ELSE 'Fall (Sep-Oct)'
  END                        AS season,
  `category`,
  `product_name`,
  MEASURE(`units_sold`)      AS units_sold,
  MEASURE(`net_sales`)       AS net_sales
FROM dbsql_workshop_gold.product_sales_metrics
WHERE `business_date` >= DATE'2026-06-01'
  AND `business_date` <= DATE'2026-10-19'
GROUP BY ALL
ORDER BY season, net_sales DESC

-- COMMAND ----------

-- 05 - Peak Hour Heatmap
SELECT
  day_of_week,
  business_hour,
  ROUND(AVG(transactions), 1) AS avg_transactions
FROM dbsql_workshop_gold.store_hourly_traffic
WHERE business_date BETWEEN DATE'2026-08-24' AND DATE'2026-10-19'
GROUP BY day_of_week, business_hour
ORDER BY
  CASE day_of_week
    WHEN 'Monday'    THEN 1  WHEN 'Tuesday'   THEN 2
    WHEN 'Wednesday' THEN 3  WHEN 'Thursday'  THEN 4
    WHEN 'Friday'    THEN 5  WHEN 'Saturday'  THEN 6
    WHEN 'Sunday'    THEN 7
  END,
  business_hour

-- COMMAND ----------

-- 06 - S007 Traffic Drop Investigation
SELECT
  business_date,
  store_name,
  transactions,
  ROUND(AVG(transactions) OVER (
    ORDER BY business_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ), 1) AS moving_avg_7d
FROM dbsql_workshop_gold.store_daily_performance
WHERE store_id = 'S007'
  AND business_date BETWEEN DATE'2026-09-15' AND DATE'2026-10-19'
ORDER BY business_date

-- COMMAND ----------

-- 07 - Low Inventory Alert
SELECT
  store_name,
  region,
  item_name,
  on_hand_qty,
  par_level,
  reorder_point,
  pct_of_par,
  stock_status
FROM dbsql_workshop_gold.inventory_current_status
WHERE stock_status <> 'OK'
ORDER BY
  CASE stock_status WHEN 'REORDER NOW' THEN 1 WHEN 'LOW' THEN 2 END,
  pct_of_par

-- COMMAND ----------

-- 08 - Loyalty Champions and QUALIFY and AI
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
FROM dbsql_workshop_gold.customer_360
WHERE customer_segment = 'Champion'
ORDER BY lifetime_net_sales DESC
LIMIT 20;

SELECT
  c.member_id,
  c.first_name,
  c.loyalty_tier,
  c.lifetime_net_sales,
  s.store_name  AS home_store,
  s.region
FROM dbsql_workshop_gold.customer_360 c
  JOIN dbsql_workshop_silver.dim_store s
    ON c.home_store_id = s.store_id
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s.region
  ORDER BY c.lifetime_net_sales DESC
) = 1
ORDER BY c.lifetime_net_sales DESC;

-- Pick the 5 rows first, so ai_query() calls the model 5 times instead of once per Champion (600)
WITH top_champions AS (
  SELECT member_id, first_name, favorite_product, lifetime_net_sales
  FROM dbsql_workshop_gold.customer_360
  WHERE customer_segment = 'Champion'
  ORDER BY lifetime_net_sales DESC
  LIMIT 5
)
SELECT
  member_id,
  first_name,
  favorite_product,
  ai_query(
    'databricks-gpt-oss-20b',
    'Classify this coffee-shop menu item into exactly one word: '
    || 'Hot, Cold, Food, or Retail. Item: ' || favorite_product
  ) AS ai_category
FROM top_champions
ORDER BY lifetime_net_sales DESC

-- COMMAND ----------

-- 09 - What customers say this week: ai_summarize over one store's comments (a single model call)
SELECT s.store_name,
  count(*)                AS comments,
  round(avg(f.rating), 1) AS avg_rating,
  ai_summarize(array_join(collect_list(f.comment), ' || '), 40) AS what_customers_say
FROM dbsql_workshop_silver.customer_feedback f
  JOIN dbsql_workshop_silver.dim_store s USING (store_id)
WHERE f.store_id = 'S007' AND f.submitted_date >= DATE'2026-10-13'
GROUP BY s.store_name

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 6: Metric views
-- MAGIC
-- MAGIC **In the workspace.** A metric view defines each KPI once. SQL, the dashboard and Genie all use it, so "average ticket"
-- MAGIC means `SUM(net_sales) / SUM(transactions)` everywhere, recalculated correctly at any grouping.
-- MAGIC
-- MAGIC 1. In **Catalog**, open `dbsql_workshop_gold` → `store_sales_metrics`.
-- MAGIC 2. Its page lists the dimensions and measures, each with a description and synonyms ("revenue", "AOV") that help
-- MAGIC    Genie.
-- MAGIC 3. **Lineage** now runs from silver through the gold models you built to this metric view and the dashboard.
-- MAGIC
-- MAGIC The cell below asks for four measures by region. Group by anything else and the ratios stay correct.

-- COMMAND ----------

-- Store KPIs by region for the trailing 28 days
SELECT region, MEASURE(net_sales) AS net_sales, MEASURE(transactions) AS transactions,
       MEASURE(avg_ticket) AS avg_ticket, MEASURE(labor_cost_pct) AS labor_cost_pct
FROM dbsql_workshop_gold.store_sales_metrics
WHERE business_date BETWEEN DATE'2026-09-22' AND DATE'2026-10-19'
GROUP BY ALL
ORDER BY net_sales DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 7: The dashboard
-- MAGIC
-- MAGIC **In the workspace.**
-- MAGIC
-- MAGIC 1. In the left sidebar, click **Dashboards**, and search for `[dbsql_workshop]`.
-- MAGIC 2. Open **[dbsql_workshop] Bean & Brew Coffee Co - Store Operations**. It's published, and runs on the metric views.
-- MAGIC 3. Pages: **Executive Overview**, **Products & Dayparts**, **Operations**, **Loyalty**.
-- MAGIC 4. On **Operations**, find the traffic dip at the Phoenix Downtown Drive-Thru from 2026-10-15. Set the **Region** filter
-- MAGIC    to **Southwest** to isolate it.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 8: Genie
-- MAGIC
-- MAGIC **In the workspace.** The Genie agent answers questions in plain English on the same governed tables and metric views.
-- MAGIC Its instructions carry the business rules: today is 2026-10-19, comp stores, the 30% labor target, REORDER NOW, the
-- MAGIC price increase, and where customer feedback lives.
-- MAGIC
-- MAGIC 1. In the left sidebar, click **Genie**, and open **[dbsql_workshop] Bean & Brew Coffee Ops Genie**.
-- MAGIC 2. Ask: *Why is traffic down at the Phoenix Downtown Drive-Thru this week?* Turn on **Agent mode** for a multi-step
-- MAGIC    answer that checks the sales data and the customer comments you enriched in Step 3.
-- MAGIC 3. Follow up: *Break it down by order channel.*
-- MAGIC 4. Expand the SQL under any answer to see the query Genie wrote and ran.
-- MAGIC 5. More to try:
-- MAGIC    - Which stores need to reorder inventory right now?
-- MAGIC    - What are the busiest hours at Pike Place on Saturdays?
-- MAGIC    - How did average ticket at comp stores change after the January 2026 price increase?
-- MAGIC    - What are customers saying about the Phoenix Downtown Drive-Thru this week?
-- MAGIC 6. Open the agent's **Instructions** to see the rules and worked examples, and **Benchmarks** to score Genie
-- MAGIC    against four questions with known answers.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 9: Alerts
-- MAGIC
-- MAGIC **In the workspace.** Three SQL alerts watch the gold tables: inventory below its reorder point, a store's traffic
-- MAGIC dropping, and labor cost over 30%. All three fire on this data.
-- MAGIC
-- MAGIC 1. In the left sidebar, under **SQL**, click **Alerts**.
-- MAGIC 2. Open **[dbsql_workshop] Bean & Brew - Inventory Items Below Reorder Point**: the query, the condition
-- MAGIC    (`items_to_reorder > 0`) and the weekly schedule.
-- MAGIC 3. Run the alert's query (or **Preview**) to see the value it checks: 3 items.
-- MAGIC 4. The schedules start **paused**, so nothing emails you. Resume one to have it notify the deployer every Monday.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 10: Tear down
-- MAGIC
-- MAGIC - **Everything the bundle deployed** (the two schemas with every table and metric view, the setup job, the dashboard,
-- MAGIC   the Genie agent and the alerts): open `databricks.yml` in this folder, click the **Deployments** icon (the rocket),
-- MAGIC   then **Destroy**. From a terminal: `databricks bundle destroy`.
-- MAGIC - **By hand:** any query you saved in the SQL editor in Steps 3 or 5, and this Git folder if you no longer need it.
-- MAGIC - **Stays:** the catalog and the SQL warehouse. The bundle never created or changed them.
