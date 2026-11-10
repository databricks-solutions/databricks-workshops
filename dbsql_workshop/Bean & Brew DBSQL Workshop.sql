-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Bean & Brew Coffee Co.: Databricks SQL workshop
-- MAGIC
-- MAGIC A 12-store coffee chain, 2025-01-01 to 2026-10-19 (treated as "today"). You'll clean raw register data, model it,
-- MAGIC govern it, then explore it with modern SQL, an AI/BI dashboard and a Genie agent.
-- MAGIC
-- MAGIC **Before you start.** Deploy the `dbsql_workshop` bundle (Open in Databricks deploys it for you). The deploy loads the
-- MAGIC raw data, creates the silver and gold tables **empty**, and creates the dashboard, the Genie agent and three alerts on
-- MAGIC top of them. This notebook fills the tables, so those assets come to life as you go.
-- MAGIC
-- MAGIC **Compute.** Attach **serverless** compute (or the SQL warehouse the bundle uses: the **Serverless Starter Warehouse**
-- MAGIC unless you chose another) and run the cells in order, or
-- MAGIC click **Run all**. Every cell is SQL and safe to re-run.
-- MAGIC
-- MAGIC **Step-by-step overview:** `DEMO_GUIDE.md`, next to this notebook.
-- MAGIC
-- MAGIC | Step | What you do |
-- MAGIC |---|---|
-- MAGIC | 0 | Connect to the catalog the bundle deployed into |
-- MAGIC | 1 | Look at the raw (bronze) data and its defects |
-- MAGIC | 2 | Silver: clean, conform and deduplicate |
-- MAGIC | 3 | Gold: build business-ready tables |
-- MAGIC | 4 | Semantic layer and governance: metric views, keys, comments, tags |
-- MAGIC | 5 | Validate every demo scenario |
-- MAGIC | 6 | Modern SQL: metric views, window functions, QUALIFY, `ai_query()` |
-- MAGIC | 7 | The dashboard, Genie agent and alerts |
-- MAGIC | 8 | Optional: build a visual data prep flow |
-- MAGIC | 9 | Tear down |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 0: Connect

-- COMMAND ----------

-- Find the catalog the bundle deployed into, and make it this session's default.
-- Every query below names the schemas (dbsql_workshop_bronze, _silver, _gold) without a catalog.
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
SELECT current_catalog() AS catalog, 'dbsql_workshop_bronze, dbsql_workshop_silver, dbsql_workshop_gold' AS schemas

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 1: Bronze, loaded by the deploy
-- MAGIC
-- MAGIC Six raw tables, about 1.5M register tickets. The defects are deliberate: mixed-case store IDs, re-sent duplicate
-- MAGIC tickets, line items as JSON text, upper-case emails. Silver fixes them.

-- COMMAND ----------

-- The six raw tables the deploy loaded
SELECT 'raw_stores' AS table_name, count(*) AS row_count FROM dbsql_workshop_bronze.raw_stores
UNION ALL SELECT 'raw_products', count(*) FROM dbsql_workshop_bronze.raw_products
UNION ALL SELECT 'raw_loyalty_members', count(*) FROM dbsql_workshop_bronze.raw_loyalty_members
UNION ALL SELECT 'raw_pos_transactions', count(*) FROM dbsql_workshop_bronze.raw_pos_transactions
UNION ALL SELECT 'raw_inventory_counts', count(*) FROM dbsql_workshop_bronze.raw_inventory_counts
UNION ALL SELECT 'raw_labor_shifts', count(*) FROM dbsql_workshop_bronze.raw_labor_shifts

-- COMMAND ----------

-- Raw register tickets with a lower-case store ID: note the line items arrive as a JSON string
SELECT * FROM dbsql_workshop_bronze.raw_pos_transactions WHERE store_id <> upper(store_id) LIMIT 5

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 2: Silver, cleaned and conformed
-- MAGIC
-- MAGIC Types cast, store IDs upper-cased, duplicate re-sends removed, JSON line items exploded. The deploy created these tables
-- MAGIC empty, with their primary and foreign keys, so each statement fills one with `INSERT OVERWRITE ... BY NAME`. Re-running
-- MAGIC replaces the rows and keeps the table, its keys and everything built on it.

-- COMMAND ----------

-- Silver dimensions: typed, trimmed, IDs and emails normalized
INSERT OVERWRITE dbsql_workshop_silver.dim_store BY NAME
SELECT upper(trim(store_id)) AS store_id, store_name, city, state, region, store_format,
  to_date(open_date) AS open_date, cast(square_feet AS INT) AS square_feet, cast(seating_capacity AS INT) AS seating_capacity,
  store_manager, cast(latitude AS DOUBLE) latitude, cast(longitude AS DOUBLE) longitude
FROM dbsql_workshop_bronze.raw_stores;

INSERT OVERWRITE dbsql_workshop_silver.dim_product BY NAME
SELECT product_id, product_name, category, temperature AS serving_type,
  cast(list_price AS DECIMAL(10,2)) AS list_price_2025,
  cast(round(list_price * 1.05, 2) AS DECIMAL(10,2)) AS list_price_2026,
  cast(unit_cost AS DECIMAL(10,2)) AS unit_cost,
  cast(round((list_price - unit_cost) / list_price, 4) AS DECIMAL(6,4)) AS standard_margin_pct,
  cast(is_seasonal AS BOOLEAN) AS is_seasonal,
  category IN ('Espresso Drinks','Brewed Coffee','Cold Drinks','Tea','Seasonal') AS is_beverage
FROM dbsql_workshop_bronze.raw_products;

INSERT OVERWRITE dbsql_workshop_silver.dim_loyalty_member BY NAME
SELECT member_id, first_name, last_name, lower(trim(email)) AS email, to_date(signup_date) AS signup_date,
  upper(home_store_id) AS home_store_id, tier AS loyalty_tier, birth_month
FROM dbsql_workshop_bronze.raw_loyalty_members;

-- COMMAND ----------

-- Silver facts: duplicates removed, voids excluded from line items, JSON exploded, price book and promos applied
INSERT OVERWRITE dbsql_workshop_silver.fact_sales_line_items BY NAME
WITH dedup AS (
  SELECT * EXCEPT (rn) FROM (
    SELECT *, row_number() OVER (PARTITION BY transaction_id ORDER BY _ingested_at DESC) rn FROM dbsql_workshop_bronze.raw_pos_transactions) WHERE rn = 1),
exploded AS (
  SELECT d.transaction_id, upper(d.store_id) AS store_id, to_timestamp(d.transaction_ts) AS transaction_ts,
    d.order_channel, d.loyalty_member_id, d.promo_code, d.price_book, li.line, li.sku, li.qty
  FROM dedup d
  LATERAL VIEW explode(from_json(d.line_items_json, 'array<struct<line:int,sku:string,qty:int>>')) e AS li
  WHERE d.ticket_status = 'COMPLETED'),
priced AS (
  SELECT
    concat(e.transaction_id, '-', e.line) AS line_item_id,
    e.transaction_id, e.line AS line_number, e.store_id, e.transaction_ts,
    to_date(e.transaction_ts) AS business_date, hour(e.transaction_ts) AS business_hour,
    e.order_channel, e.loyalty_member_id, e.promo_code,
    e.sku AS product_id, e.qty AS quantity,
    CASE WHEN e.price_book = 'menu_v2026' THEN p.list_price_2026 ELSE p.list_price_2025 END AS unit_price,
    cast(e.qty * CASE WHEN e.price_book = 'menu_v2026' THEN p.list_price_2026 ELSE p.list_price_2025 END AS DECIMAL(12,2)) AS gross_sales,
    cast(round(e.qty * CASE WHEN e.price_book = 'menu_v2026' THEN p.list_price_2026 ELSE p.list_price_2025 END *
      CASE WHEN e.promo_code = 'LOYALTY10' THEN 0.10 WHEN e.promo_code = 'HAPPYHOUR' AND p.is_beverage THEN 0.20 ELSE 0 END, 2) AS DECIMAL(12,2)) AS discount_amount,
    cast(e.qty * p.unit_cost AS DECIMAL(12,2)) AS cost_of_goods
  FROM exploded e JOIN dbsql_workshop_silver.dim_product p ON e.sku = p.product_id)
SELECT *, cast(gross_sales - discount_amount AS DECIMAL(12,2)) AS net_sales FROM priced;

INSERT OVERWRITE dbsql_workshop_silver.fact_transactions BY NAME
WITH dedup AS (
  SELECT * EXCEPT (rn) FROM (
    SELECT *, row_number() OVER (PARTITION BY transaction_id ORDER BY _ingested_at DESC) rn FROM dbsql_workshop_bronze.raw_pos_transactions) WHERE rn = 1),
items AS (
  SELECT transaction_id, count(*) line_count, sum(quantity) item_qty, sum(gross_sales) gross_sales, sum(discount_amount) discount_amount,
    sum(net_sales) net_sales, sum(cost_of_goods) cost_of_goods
  FROM dbsql_workshop_silver.fact_sales_line_items GROUP BY transaction_id)
SELECT d.transaction_id, upper(d.store_id) AS store_id, d.register_id, to_timestamp(d.transaction_ts) AS transaction_ts,
  to_date(to_timestamp(d.transaction_ts)) AS business_date, hour(to_timestamp(d.transaction_ts)) AS business_hour,
  d.order_channel, d.loyalty_member_id, d.loyalty_member_id IS NOT NULL AS is_loyalty_transaction,
  d.payment_method, d.promo_code, d.ticket_status = 'VOIDED' AS is_voided,
  coalesce(i.line_count,0) line_count, coalesce(i.item_qty,0) item_qty,
  coalesce(i.gross_sales,0) gross_sales, coalesce(i.discount_amount,0) discount_amount, coalesce(i.net_sales,0) net_sales,
  cast(round(coalesce(i.net_sales,0) * CASE s.state WHEN 'WA' THEN 0.1025 WHEN 'OR' THEN 0 WHEN 'CA' THEN 0.0875 WHEN 'AZ' THEN 0.086 WHEN 'TX' THEN 0.0825 ELSE 0.0881 END, 2) AS DECIMAL(12,2)) AS sales_tax,
  coalesce(i.cost_of_goods,0) cost_of_goods
FROM dedup d LEFT JOIN items i USING (transaction_id) JOIN dbsql_workshop_silver.dim_store s ON upper(d.store_id) = s.store_id;

INSERT OVERWRITE dbsql_workshop_silver.fact_inventory_daily BY NAME
SELECT upper(store_id) store_id, count_date, item_code, item_name, unit_of_measure,
  cast(on_hand_qty AS DECIMAL(12,1)) on_hand_qty, cast(par_level AS DECIMAL(12,1)) par_level, cast(reorder_point AS DECIMAL(12,1)) reorder_point,
  cast(unit_cost AS DECIMAL(10,2)) unit_cost, cast(on_hand_qty * unit_cost AS DECIMAL(12,2)) AS inventory_value,
  on_hand_qty < reorder_point AS is_below_reorder_point
FROM dbsql_workshop_bronze.raw_inventory_counts;

INSERT OVERWRITE dbsql_workshop_silver.fact_labor_daily BY NAME
SELECT upper(store_id) store_id, business_date, cast(scheduled_hours AS DECIMAL(8,1)) scheduled_hours, cast(worked_hours AS DECIMAL(8,1)) worked_hours,
  headcount, cast(avg_hourly_wage AS DECIMAL(8,2)) avg_hourly_wage, cast(worked_hours * avg_hourly_wage AS DECIMAL(12,2)) AS labor_cost
FROM dbsql_workshop_bronze.raw_labor_shifts;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 3: Gold, business-ready
-- MAGIC
-- MAGIC One table per business question: store-day performance, product mix, hourly traffic, current inventory, a customer
-- MAGIC 360 and the loyalty promo list. The dashboard, the Genie agent and the alerts read these tables, so they show data as
-- MAGIC soon as this step finishes.

-- COMMAND ----------

-- Gold: one table per business grain
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

INSERT OVERWRITE dbsql_workshop_gold.product_sales_daily BY NAME
SELECT f.business_date, f.store_id, s.store_name, s.region, f.product_id, p.product_name, p.category, p.serving_type, p.is_beverage,
  sum(f.quantity) units_sold, sum(f.gross_sales) gross_sales, sum(f.discount_amount) discount_amount, sum(f.net_sales) net_sales,
  sum(f.cost_of_goods) cost_of_goods, sum(f.net_sales) - sum(f.cost_of_goods) gross_profit
FROM dbsql_workshop_silver.fact_sales_line_items f JOIN dbsql_workshop_silver.dim_product p USING (product_id) JOIN dbsql_workshop_silver.dim_store s USING (store_id)
GROUP BY ALL;

INSERT OVERWRITE dbsql_workshop_gold.store_hourly_traffic BY NAME
SELECT store_id, business_date, business_hour, date_format(business_date,'EEEE') AS day_of_week,
  count(*) transactions, sum(net_sales) net_sales, count_if(order_channel = 'Mobile App') mobile_transactions
FROM dbsql_workshop_silver.fact_transactions WHERE NOT is_voided GROUP BY ALL;

INSERT OVERWRITE dbsql_workshop_gold.inventory_current_status BY NAME
WITH latest AS (SELECT * FROM dbsql_workshop_silver.fact_inventory_daily QUALIFY row_number() OVER (PARTITION BY store_id, item_code ORDER BY count_date DESC) = 1)
SELECT l.store_id, s.store_name, s.region, l.count_date AS as_of_date, l.item_code, l.item_name, l.unit_of_measure,
  l.on_hand_qty, l.par_level, l.reorder_point, cast(round(l.on_hand_qty / l.par_level, 3) AS DECIMAL(6,3)) AS pct_of_par,
  l.inventory_value,
  CASE WHEN l.on_hand_qty < l.reorder_point THEN 'REORDER NOW' WHEN l.on_hand_qty < l.reorder_point * 1.5 THEN 'LOW' ELSE 'OK' END AS stock_status
FROM latest l JOIN dbsql_workshop_silver.dim_store s USING (store_id);

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

-- The loyalty promo list. Step 8 builds the same list without code, in a visual data prep flow.
-- CHAR(36) is '$': a literal dollar-digit is read as a parameter in notebook SQL.
INSERT OVERWRITE dbsql_workshop_gold.loyalty_promo_targets BY NAME
SELECT c.member_id, c.first_name, c.loyalty_tier, c.customer_segment, c.lifetime_net_sales, c.visits, c.favorite_product,
       s.store_name, s.region,
       CASE WHEN c.favorite_product LIKE '%Cold Brew%' THEN 'Free Nitro Upgrade'
            WHEN c.favorite_product LIKE '%Latte%'     THEN 'Free Oat Milk Swap'
            ELSE CONCAT(CHAR(36), '2 Off Pastry') END AS promo_offer
FROM dbsql_workshop_gold.customer_360 c
JOIN dbsql_workshop_silver.dim_store s ON c.home_store_id = s.store_id
WHERE c.customer_segment IN ('Champion', 'Loyal Regular') AND c.loyalty_tier IN ('Gold', 'Reserve');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 4: Semantic layer and governance
-- MAGIC
-- MAGIC **Metric views** (created by the deploy) give every KPI one governed definition. SQL, the dashboard and Genie all use
-- MAGIC them, so "average ticket" means the same thing everywhere. Query a measure with `MEASURE()`:

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
-- MAGIC **Keys** were declared when the deploy created the tables. Genie uses them to join correctly:

-- COMMAND ----------

SELECT tc.table_schema, tc.table_name, tc.constraint_name, tc.constraint_type
FROM information_schema.table_constraints tc
WHERE tc.table_schema IN ('dbsql_workshop_silver', 'dbsql_workshop_gold')
ORDER BY tc.constraint_type DESC, tc.table_schema, tc.table_name

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Column comments** document the data for people and for Genie:

-- COMMAND ----------

ALTER TABLE dbsql_workshop_gold.customer_360 ALTER COLUMN
  loyalty_tier COMMENT 'Rewards tier: Green (entry), Gold, Reserve (top tier).',
  home_store_id COMMENT 'Store the member enrolled at. FK to dim_store.',
  lifetime_net_sales COMMENT 'Total net sales (pre-tax, after discounts) across all non-voided visits since 2025-01-01, USD.',
  days_since_last_visit COMMENT 'Days between last visit and 2026-10-19 (data as-of date).',
  customer_segment COMMENT 'Value segment by lifetime spend percentile: Champion (top 10%), Loyal Regular (next 30%), Occasional (next 40%), Light (bottom 20%).';

ALTER TABLE dbsql_workshop_gold.inventory_current_status ALTER COLUMN
  store_name COMMENT 'Customer-facing store name.',
  region COMMENT 'Operating region: Pacific Northwest, California, Southwest, Mountain.',
  stock_status COMMENT 'REORDER NOW = below reorder point; LOW = below 1.5x reorder point; OK otherwise.';

ALTER TABLE dbsql_workshop_gold.loyalty_promo_targets ALTER COLUMN
  loyalty_tier COMMENT 'Rewards tier: Green (entry), Gold, Reserve (top tier).',
  customer_segment COMMENT 'Value segment by lifetime spend percentile: Champion (top 10%), Loyal Regular (next 30%), Occasional (next 40%), Light (bottom 20%).',
  lifetime_net_sales COMMENT 'Total net sales (pre-tax, after discounts) across all non-voided visits since 2025-01-01, USD.',
  store_name COMMENT 'Customer-facing store name.',
  region COMMENT 'Operating region: Pacific Northwest, California, Southwest, Mountain.';

ALTER TABLE dbsql_workshop_gold.product_sales_daily ALTER COLUMN
  business_date COMMENT 'Store-local business date of the sale.',
  store_name COMMENT 'Customer-facing store name.',
  region COMMENT 'Operating region: Pacific Northwest, California, Southwest, Mountain.',
  category COMMENT 'Menu category: Espresso Drinks, Brewed Coffee, Cold Drinks, Tea, Bakery, Breakfast, Seasonal, Retail Beans.',
  is_beverage COMMENT 'True for drink categories; false for food and retail beans.';

ALTER TABLE dbsql_workshop_gold.store_daily_performance ALTER COLUMN
  store_name COMMENT 'Customer-facing store name.',
  region COMMENT 'Operating region: Pacific Northwest, California, Southwest, Mountain.',
  store_format COMMENT 'Store format: Flagship Cafe, Cafe, Drive-Thru, or Express Kiosk.',
  is_comp_store COMMENT 'True when the store has been open at least 365 days as of business_date (comparable-store sales).',
  transactions COMMENT 'Count of completed (non-voided) tickets. Proxy for customer traffic.',
  net_sales COMMENT 'Net sales = gross sales - discounts, pre-tax, USD. Primary revenue measure.',
  gross_profit COMMENT 'net_sales - cost_of_goods, USD.',
  avg_ticket COMMENT 'Average ticket = net_sales / transactions, USD.',
  digital_transactions COMMENT 'Tickets ordered via Mobile App or Delivery.',
  labor_cost COMMENT 'worked_hours * avg_hourly_wage, USD (excludes benefits).',
  labor_cost_pct COMMENT 'labor_cost / net_sales. Target is below 0.30.';

ALTER TABLE dbsql_workshop_silver.dim_loyalty_member ALTER COLUMN
  email COMMENT 'Member email address (PII). Lower-cased.',
  home_store_id COMMENT 'Store the member enrolled at. FK to dim_store.',
  loyalty_tier COMMENT 'Rewards tier: Green (entry), Gold, Reserve (top tier).';

ALTER TABLE dbsql_workshop_silver.dim_product ALTER COLUMN
  category COMMENT 'Menu category: Espresso Drinks, Brewed Coffee, Cold Drinks, Tea, Bakery, Breakfast, Seasonal, Retail Beans.',
  list_price_2025 COMMENT 'Menu list price in USD under the 2025 price book.',
  list_price_2026 COMMENT 'Menu list price in USD under the 2026 price book (+5%, effective 2026-01-05).',
  unit_cost COMMENT 'Standard cost of goods per unit in USD (ingredients + packaging).',
  standard_margin_pct COMMENT 'Standard gross margin at 2025 list price: (list_price - unit_cost) / list_price.',
  is_beverage COMMENT 'True for drink categories; false for food and retail beans.';

ALTER TABLE dbsql_workshop_silver.dim_store ALTER COLUMN
  store_id COMMENT 'Unique store identifier (e.g. S001). Primary key.',
  store_name COMMENT 'Customer-facing store name.',
  region COMMENT 'Operating region: Pacific Northwest, California, Southwest, Mountain.',
  store_format COMMENT 'Store format: Flagship Cafe, Cafe, Drive-Thru, or Express Kiosk.',
  open_date COMMENT 'Date the store opened to the public. Stores open < 12 months are considered new (non-comp) stores.',
  seating_capacity COMMENT 'Number of customer seats.';

ALTER TABLE dbsql_workshop_silver.fact_inventory_daily ALTER COLUMN
  is_below_reorder_point COMMENT 'True when on-hand quantity is below the reorder point and a replenishment order is required.';

ALTER TABLE dbsql_workshop_silver.fact_labor_daily ALTER COLUMN
  labor_cost COMMENT 'worked_hours * avg_hourly_wage, USD (excludes benefits).';

ALTER TABLE dbsql_workshop_silver.fact_sales_line_items ALTER COLUMN
  business_date COMMENT 'Store-local business date of the sale.',
  business_hour COMMENT 'Hour of day (0-23) the ticket was rung up, store-local.',
  order_channel COMMENT 'How the order was placed: In-Store, Mobile App, Drive-Thru, Delivery.',
  gross_sales COMMENT 'quantity * unit_price before discounts, USD.',
  discount_amount COMMENT 'Promotional discount: LOYALTY10 = 10% off; HAPPYHOUR = 20% off beverages.',
  cost_of_goods COMMENT 'quantity * standard unit cost, USD.',
  net_sales COMMENT 'gross_sales - discount_amount, pre-tax.';

ALTER TABLE dbsql_workshop_silver.fact_transactions ALTER COLUMN
  is_loyalty_transaction COMMENT 'True if a Bean & Brew Rewards member was attached to the ticket.',
  is_voided COMMENT 'True if the ticket was voided at the register; excluded from sales metrics.',
  net_sales COMMENT 'Ticket net sales (gross - discounts), pre-tax, USD. Zero for voided tickets.',
  sales_tax COMMENT 'Sales tax collected based on store state rate.';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Tags** classify schemas, tables and PII columns (`contains_pii` on the email columns). If your account has
-- MAGIC governed tag policies that reject these values, this cell fails: skip it and carry on.

-- COMMAND ----------

ALTER SCHEMA dbsql_workshop_bronze SET TAGS ('layer' = 'bronze', 'domain' = 'sales');
ALTER SCHEMA dbsql_workshop_silver SET TAGS ('layer' = 'silver', 'domain' = 'sales');
ALTER SCHEMA dbsql_workshop_gold SET TAGS ('layer' = 'gold', 'domain' = 'sales');
ALTER TABLE dbsql_workshop_silver.fact_inventory_daily SET TAGS ('domain' = 'supply_chain');
ALTER TABLE dbsql_workshop_gold.inventory_current_status SET TAGS ('domain' = 'supply_chain');
ALTER TABLE dbsql_workshop_silver.fact_labor_daily SET TAGS ('domain' = 'hr');
ALTER TABLE dbsql_workshop_silver.dim_loyalty_member ALTER COLUMN email SET TAGS ('contains_pii' = 'true');
ALTER TABLE dbsql_workshop_bronze.raw_loyalty_members ALTER COLUMN email SET TAGS ('contains_pii' = 'true');
ALTER TABLE dbsql_workshop_gold.store_daily_performance SET TAGS ('owner_team' = 'analytics');
ALTER TABLE dbsql_workshop_gold.customer_360 SET TAGS ('owner_team' = 'customer_data');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 5: Validate
-- MAGIC
-- MAGIC Checks every demo scenario the dashboard, Genie and alerts rely on, plus a data fingerprint that is identical on every
-- MAGIC build. The second cell stops the notebook if a check failed.

-- COMMAND ----------

-- Every demo scenario, checked against the data you just built. FAIL means a step above didn't run or was edited.
CREATE OR REPLACE TEMPORARY VIEW workshop_checks AS
WITH
b AS (SELECT count(*) - count(DISTINCT transaction_id) AS dups, count_if(store_id <> upper(store_id)) AS lower_ids
      FROM dbsql_workshop_bronze.raw_pos_transactions),
s AS (SELECT count(*) AS n, count(*) - count(DISTINCT transaction_id) AS dups, count_if(store_id <> upper(store_id)) AS lower_ids
      FROM dbsql_workshop_silver.fact_transactions),
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
promo AS (SELECT count(*) AS n FROM dbsql_workshop_gold.loyalty_promo_targets),
fp AS (SELECT sha2(concat_ws('|',
         (SELECT array_join(sort_array(collect_list(concat_ws(',', store_id, business_date, transactions, net_sales, labor_cost))), ';') FROM dbsql_workshop_gold.store_daily_performance),
         (SELECT array_join(sort_array(collect_list(concat_ws(',', member_id, visits, lifetime_net_sales, favorite_product, customer_segment))), ';') FROM dbsql_workshop_gold.customer_360),
         (SELECT array_join(sort_array(collect_list(concat_ws(',', store_id, item_code, on_hand_qty, stock_status))), ';') FROM dbsql_workshop_gold.inventory_current_status),
         (SELECT cast(count(*) AS STRING) FROM dbsql_workshop_bronze.raw_pos_transactions)), 256) AS fingerprint)
SELECT 1 AS n, 'Bronze has duplicate and lower-case defects' AS check_name, b.dups > 0 AND b.lower_ids > 0 AS ok,
       format_string('%,d duplicate IDs, %,d lower-case IDs', b.dups, b.lower_ids) AS detail FROM b
UNION ALL SELECT 2, 'Silver defects cleaned', s.dups = 0 AND s.lower_ids = 0,
       format_string('%,d tickets, %d duplicates, %d lower-case', s.n, s.dups, s.lower_ids) FROM s
UNION ALL SELECT 3, 'Net sales reconcile across layers',
       rec.silver = rec.store_day AND rec.silver = rec.product_day AND rec.silver = rec.store_hour, format_number(rec.silver, 2) FROM rec
UNION ALL SELECT 4, '(a) S007 traffic drops from 2026-10-15', traffic.chg < -0.4,
       format_string('%+.1f%% vs prior 30 days', traffic.chg * 100) FROM traffic
UNION ALL SELECT 5, '(b) Exactly 3 items to reorder', reorder.items = 'S004 Oat Milk, S008 Oat Milk, S010 Espresso Beans', reorder.items FROM reorder
UNION ALL SELECT 6, '(c) Average ticket rises after the 2026-01-05 price increase', ticket.after > ticket.before,
       format_string('%.2f -> %.2f', cast(ticket.before AS DOUBLE), cast(ticket.after AS DOUBLE)) FROM ticket
UNION ALL SELECT 7, '(d) RiNo opened 2026-03-01, no comp days yet', rino.first_day = DATE'2026-03-01' AND rino.comp_days = 0,
       format_string('first day %s, %d comp days', cast(rino.first_day AS STRING), rino.comp_days) FROM rino
UNION ALL SELECT 8, '(e) Cold drinks peak May-Aug', cold.ratio > 3, format_string('%.1fx the rest of 2025', cold.ratio) FROM cold
UNION ALL SELECT 9, '(f) Pumpkin Spice Latte only Sep-Nov', size(psl.months) > 0 AND size(array_except(psl.months, array(9, 10, 11))) = 0,
       'months ' || array_join(psl.months, ', ') FROM psl
UNION ALL SELECT 10, 'Labor alert has a store over 30%', labor.stores <> '', labor.stores FROM labor
UNION ALL SELECT 11, 'Promo target list built', promo.n > 0, format_string('%,d members', promo.n) FROM promo
UNION ALL SELECT 12, 'Data fingerprint matches the reference build',
       fp.fingerprint = 'ab02b95e172468b55ac67d170273121b11f8e795348cd90d4f8ac547a17ba1bc', fp.fingerprint FROM fp;

SELECT n, check_name, CASE WHEN ok THEN 'PASS' ELSE 'FAIL' END AS result, detail FROM workshop_checks ORDER BY n

-- COMMAND ----------

-- Stops the notebook here if any check failed
SELECT CASE WHEN count_if(NOT ok) > 0
            THEN raise_error('Validation failed: ' || array_join(collect_list(CASE WHEN NOT ok THEN check_name END), '; '))
            ELSE 'All ' || count(*) || ' checks passed' END AS validation
FROM workshop_checks

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 6: Modern SQL
-- MAGIC
-- MAGIC One feature per query, on the data you just built. Paste any of them into the **SQL editor** to show it there (attach
-- MAGIC the same SQL warehouse, and pick the deployed catalog in the editor's catalog picker if it isn't your
-- MAGIC workspace default). Query 08 calls an LLM from SQL with `ai_query()` on the pay-per-token `databricks-gpt-oss-20b` endpoint.

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

-- 09 - Data Quality Bronze vs Silver
SELECT
  'dbsql_workshop_bronze.raw_pos_transactions' AS layer,
  COUNT(*)                                                      AS total_rows,
  COUNT(*) - COUNT(DISTINCT transaction_id)                     AS duplicate_txn_ids,
  SUM(CASE WHEN store_id <> UPPER(store_id) THEN 1 ELSE 0 END) AS lowercase_store_ids
FROM dbsql_workshop_bronze.raw_pos_transactions
UNION ALL
SELECT
  'dbsql_workshop_silver.fact_transactions',
  COUNT(*),
  COUNT(*) - COUNT(DISTINCT transaction_id),
  SUM(CASE WHEN store_id <> UPPER(store_id) THEN 1 ELSE 0 END)
FROM dbsql_workshop_silver.fact_transactions

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 7: The dashboard, Genie agent and alerts
-- MAGIC
-- MAGIC The deploy created these with the bundle, on the tables you just filled. Find them in the workspace search by typing
-- MAGIC **`[dbsql_workshop]`**, or open `databricks.yml` and click the **Deployments** icon to see links to each one.
-- MAGIC
-- MAGIC | Asset | Name | Try this |
-- MAGIC |---|---|---|
-- MAGIC | AI/BI dashboard | `[dbsql_workshop] Bean & Brew Coffee Co - Store Operations` | Operations page: spot the Phoenix (S007) traffic dip, then ask Genie why |
-- MAGIC | Genie agent | `[dbsql_workshop] Bean & Brew Coffee Ops Genie` | "Which stores need to reorder inventory right now?" · "What are the busiest hours at Pike Place on Saturdays?" · Run the 4 benchmarks from the Benchmarks tab |
-- MAGIC | 3 SQL alerts | `[dbsql_workshop] Bean & Brew - ...` | Open one and click **Preview** to see the value it checks. Their weekly schedules start paused |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 8 (optional): Build a visual data prep flow
-- MAGIC
-- MAGIC Build the promo list from Step 3 without code. This creates one workspace object you delete by hand in Step 9.
-- MAGIC
-- MAGIC 1. **New → Visual data prep**, and save the flow in your home folder as `Bean & Brew promo list`.
-- MAGIC 2. Add a **source**: `dbsql_workshop_gold.customer_360`. Add a second source: `dbsql_workshop_silver.dim_store`.
-- MAGIC 3. **Join** them on `home_store_id = store_id`.
-- MAGIC 4. **Filter**: `customer_segment` is `Champion` or `Loyal Regular`, and `loyalty_tier` is `Gold` or `Reserve`.
-- MAGIC 5. **Add a column** `promo_offer`: `Free Nitro Upgrade` if `favorite_product` contains "Cold Brew", `Free Oat Milk Swap`
-- MAGIC    if it contains "Latte", otherwise `$2 Off Pastry`.
-- MAGIC 6. Preview each step: the result should have 974 members, the same as `dbsql_workshop_gold.loyalty_promo_targets`.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 9: Tear down
-- MAGIC
-- MAGIC - **Everything the bundle deployed** (the three schemas with every table and metric view, the setup job, the
-- MAGIC   dashboard, the Genie agent and the alerts): open `databricks.yml`, click the **Deployments** icon, then **Destroy**.
-- MAGIC   From a terminal: `databricks bundle destroy`.
-- MAGIC - **By hand:** the flow from Step 8 if you built it, any query you saved from Step 6, and this Git folder if you no
-- MAGIC   longer need it.
