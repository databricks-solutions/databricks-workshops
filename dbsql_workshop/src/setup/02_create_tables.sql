-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Bean & Brew setup 2 of 2: create the silver and gold tables and the metric views
-- MAGIC
-- MAGIC Runs during `databricks bundle deploy`, after the raw data load. It creates every silver and gold table **empty**, with
-- MAGIC its comment and its primary and foreign keys, then the three metric views on top of them. The AI/BI dashboard, the
-- MAGIC Genie agent and the alerts are created next, and they need these tables to exist. The workshop notebook fills them.
-- MAGIC
-- MAGIC Tables that already exist are left as they are, so redeploying never loses data. The metric views are replaced, so a
-- MAGIC changed definition takes effect on the next deploy.

-- COMMAND ----------

USE CATALOG IDENTIFIER(:catalog);

-- COMMAND ----------

-- Silver: dimensions first, because the facts' foreign keys reference them

CREATE TABLE IF NOT EXISTS dbsql_workshop_silver.dim_store (
  store_id STRING NOT NULL,
  store_name STRING,
  city STRING,
  state STRING,
  region STRING,
  store_format STRING,
  open_date DATE,
  square_feet INT,
  seating_capacity INT,
  store_manager STRING,
  latitude DOUBLE,
  longitude DOUBLE,
  CONSTRAINT pk_dim_store PRIMARY KEY (store_id)
)
COMMENT 'Conformed store dimension for Bean & Brew Coffee Co. One row per store location with typed attributes. Source: bronze.raw_stores.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_silver.dim_product (
  product_id STRING NOT NULL,
  product_name STRING,
  category STRING,
  serving_type STRING,
  list_price_2025 DECIMAL(10,2),
  list_price_2026 DECIMAL(10,2),
  unit_cost DECIMAL(10,2),
  standard_margin_pct DECIMAL(6,4),
  is_seasonal BOOLEAN,
  is_beverage BOOLEAN,
  CONSTRAINT pk_dim_product PRIMARY KEY (product_id)
)
COMMENT 'Conformed product/menu dimension. One row per sellable SKU with 2025 list price, standard unit cost, and standard margin. Source: bronze.raw_products.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_silver.dim_loyalty_member (
  member_id STRING NOT NULL,
  first_name STRING,
  last_name STRING,
  email STRING,
  signup_date DATE,
  home_store_id STRING,
  loyalty_tier STRING,
  birth_month INT,
  CONSTRAINT pk_dim_loyalty_member PRIMARY KEY (member_id),
  CONSTRAINT fk_member_home_store FOREIGN KEY (home_store_id) REFERENCES dbsql_workshop_silver.dim_store(store_id)
)
COMMENT 'Cleaned Bean & Brew Rewards loyalty member dimension. Emails lower-cased, home store IDs normalized, dates typed. Source: bronze.raw_loyalty_members.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_silver.fact_transactions (
  transaction_id STRING NOT NULL,
  store_id STRING,
  register_id STRING,
  transaction_ts TIMESTAMP,
  business_date DATE,
  business_hour INT,
  order_channel STRING,
  loyalty_member_id STRING,
  is_loyalty_transaction BOOLEAN,
  payment_method STRING,
  promo_code STRING,
  is_voided BOOLEAN,
  line_count BIGINT,
  item_qty BIGINT,
  gross_sales DECIMAL(22,2),
  discount_amount DECIMAL(22,2),
  net_sales DECIMAL(22,2),
  sales_tax DECIMAL(12,2),
  cost_of_goods DECIMAL(22,2),
  CONSTRAINT pk_fact_transactions PRIMARY KEY (transaction_id),
  CONSTRAINT fk_txn_store FOREIGN KEY (store_id) REFERENCES dbsql_workshop_silver.dim_store(store_id),
  CONSTRAINT fk_txn_member FOREIGN KEY (loyalty_member_id) REFERENCES dbsql_workshop_silver.dim_loyalty_member(member_id)
)
COMMENT 'Cleaned POS ticket (transaction header) fact. One row per unique ticket, including voided tickets (flagged). Totals rolled up from line items; sales tax applied by store state. Sources: bronze.raw_pos_transactions, silver.fact_sales_line_items, silver.dim_store.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_silver.fact_sales_line_items (
  line_item_id STRING NOT NULL,
  transaction_id STRING,
  line_number INT,
  store_id STRING,
  transaction_ts TIMESTAMP,
  business_date DATE,
  business_hour INT,
  order_channel STRING,
  loyalty_member_id STRING,
  promo_code STRING,
  product_id STRING,
  quantity INT,
  unit_price DECIMAL(10,2),
  gross_sales DECIMAL(12,2),
  discount_amount DECIMAL(12,2),
  cost_of_goods DECIMAL(12,2),
  net_sales DECIMAL(12,2),
  CONSTRAINT pk_fact_sales_line_items PRIMARY KEY (line_item_id),
  CONSTRAINT fk_line_txn FOREIGN KEY (transaction_id) REFERENCES dbsql_workshop_silver.fact_transactions(transaction_id),
  CONSTRAINT fk_line_product FOREIGN KEY (product_id) REFERENCES dbsql_workshop_silver.dim_product(product_id),
  CONSTRAINT fk_line_store FOREIGN KEY (store_id) REFERENCES dbsql_workshop_silver.dim_store(store_id)
)
COMMENT 'Cleaned POS line-item fact. One row per completed ticket line. Duplicate POS re-sends removed, voided tickets excluded, JSON payload exploded, prices resolved from the active price book, promo discounts applied. Sources: bronze.raw_pos_transactions, silver.dim_product.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_silver.fact_inventory_daily (
  store_id STRING,
  count_date DATE,
  item_code STRING,
  item_name STRING,
  unit_of_measure STRING,
  on_hand_qty DECIMAL(12,1),
  par_level DECIMAL(12,1),
  reorder_point DECIMAL(12,1),
  unit_cost DECIMAL(10,2),
  inventory_value DECIMAL(12,2),
  is_below_reorder_point BOOLEAN,
  CONSTRAINT fk_inv_store FOREIGN KEY (store_id) REFERENCES dbsql_workshop_silver.dim_store(store_id)
)
COMMENT 'Cleaned end-of-day inventory position per store and stock item, with valuation and reorder flag. Source: bronze.raw_inventory_counts.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_silver.fact_labor_daily (
  store_id STRING,
  business_date DATE,
  scheduled_hours DECIMAL(8,1),
  worked_hours DECIMAL(8,1),
  headcount INT,
  avg_hourly_wage DECIMAL(8,2),
  labor_cost DECIMAL(12,2),
  CONSTRAINT fk_labor_store FOREIGN KEY (store_id) REFERENCES dbsql_workshop_silver.dim_store(store_id)
)
COMMENT 'Cleaned daily labor fact per store from the WFM system, with computed labor cost. Source: bronze.raw_labor_shifts.';

-- COMMAND ----------

-- Gold: one table per business grain

CREATE TABLE IF NOT EXISTS dbsql_workshop_gold.store_daily_performance (
  business_date DATE,
  store_id STRING,
  store_name STRING,
  city STRING,
  state STRING,
  region STRING,
  store_format STRING,
  is_comp_store BOOLEAN,
  transactions BIGINT,
  items_sold BIGINT,
  gross_sales DECIMAL(32,2),
  discount_amount DECIMAL(32,2),
  net_sales DECIMAL(32,2),
  cost_of_goods DECIMAL(32,2),
  gross_profit DECIMAL(33,2),
  avg_ticket DECIMAL(10,2),
  loyalty_transactions BIGINT,
  digital_transactions BIGINT,
  drive_thru_transactions BIGINT,
  labor_hours DECIMAL(8,1),
  labor_cost DECIMAL(12,2),
  labor_cost_pct DECIMAL(8,4)
)
COMMENT 'Gold: daily store P&L-lite. One row per store per business date with traffic, sales, margin, loyalty mix, and labor. Excludes voided tickets. Primary table for store performance dashboards, Genie, and alerts. Sources: silver.fact_transactions, silver.fact_labor_daily, silver.dim_store.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_gold.product_sales_daily (
  business_date DATE,
  store_id STRING,
  store_name STRING,
  region STRING,
  product_id STRING,
  product_name STRING,
  category STRING,
  serving_type STRING,
  is_beverage BOOLEAN,
  units_sold BIGINT,
  gross_sales DECIMAL(22,2),
  discount_amount DECIMAL(22,2),
  net_sales DECIMAL(22,2),
  cost_of_goods DECIMAL(22,2),
  gross_profit DECIMAL(23,2)
)
COMMENT 'Gold: daily product sales by store. One row per business date, store, and product with units, sales, and margin. Sources: silver.fact_sales_line_items, silver.dim_product, silver.dim_store.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_gold.store_hourly_traffic (
  store_id STRING,
  business_date DATE,
  business_hour INT,
  day_of_week STRING,
  transactions BIGINT,
  net_sales DECIMAL(32,2),
  mobile_transactions BIGINT
)
COMMENT 'Gold: hourly ticket counts and sales per store and day, for staffing and peak-hour analysis. Excludes voided tickets. Source: silver.fact_transactions.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_gold.inventory_current_status (
  store_id STRING,
  store_name STRING,
  region STRING,
  as_of_date DATE,
  item_code STRING,
  item_name STRING,
  unit_of_measure STRING,
  on_hand_qty DECIMAL(12,1),
  par_level DECIMAL(12,1),
  reorder_point DECIMAL(12,1),
  pct_of_par DECIMAL(6,3),
  inventory_value DECIMAL(12,2),
  stock_status STRING
)
COMMENT 'Gold: latest inventory position per store and stock item with stock health status (OK / LOW / REORDER NOW). Source: silver.fact_inventory_daily, silver.dim_store.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_gold.customer_360 (
  member_id STRING NOT NULL,
  first_name STRING,
  last_name STRING,
  loyalty_tier STRING,
  signup_date DATE,
  home_store_id STRING,
  visits BIGINT,
  lifetime_net_sales DECIMAL(32,2),
  avg_ticket DECIMAL(10,2),
  first_visit_date DATE,
  last_visit_date DATE,
  days_since_last_visit INT,
  favorite_store_id STRING,
  favorite_product STRING,
  mobile_share DOUBLE,
  customer_segment STRING,
  CONSTRAINT pk_customer_360 PRIMARY KEY (member_id)
)
COMMENT 'Gold: one row per Bean & Brew Rewards member with lifetime value, visit frequency, recency, favorite store/product, and value segment (Champion = top 10% by lifetime spend, Loyal Regular = next 30%, Occasional = next 40%, Light = bottom 20%). Sources: silver.dim_loyalty_member, silver.fact_transactions, silver.fact_sales_line_items, silver.dim_product.';

CREATE TABLE IF NOT EXISTS dbsql_workshop_gold.loyalty_promo_targets (
  member_id STRING,
  first_name STRING,
  loyalty_tier STRING,
  customer_segment STRING,
  lifetime_net_sales DECIMAL(32,2),
  visits BIGINT,
  favorite_product STRING,
  store_name STRING,
  region STRING,
  promo_offer STRING
)
COMMENT 'Gold: loyalty promo target list. Champion and Loyal Regular members in the Gold or Reserve tier, with an offer based on their favorite product.';

-- COMMAND ----------

-- Metric views: one governed definition of every KPI, used by SQL, the dashboard and Genie

CREATE OR REPLACE VIEW dbsql_workshop_gold.store_sales_metrics
WITH METRICS LANGUAGE YAML
COMMENT 'Certified Bean & Brew store performance KPIs: net sales, traffic, average ticket, margin, labor and loyalty mix. Use for any store-level or regional sales question.'
AS $$
version: 1.1
source: dbsql_workshop_gold.store_daily_performance
comment: Certified Bean & Brew store performance KPIs. One source row = one store-day. Excludes voided tickets.
dimensions:
  - name: business_date
    expr: business_date
    display_name: Business Date
    comment: Store-local business date.
    format:
      type: date
      date_format: year_month_day
    synonyms: [date, day, sales date]
  - name: business_month
    expr: DATE_TRUNC('MONTH', business_date)
    display_name: Month
    comment: Calendar month of the business date.
    format:
      type: date
      date_format: locale_short_month
    synonyms: [month, sales month]
  - name: business_week
    expr: DATE_TRUNC('WEEK', business_date)
    display_name: Week
    comment: Week (Monday start) of the business date.
    synonyms: [week]
  - name: day_of_week
    expr: date_format(business_date, 'EEEE')
    display_name: Day of Week
    synonyms: [weekday]
  - name: store_name
    expr: store_name
    display_name: Store
    comment: Customer-facing store name.
    synonyms: [store, location, cafe, shop]
  - name: store_id
    expr: store_id
    display_name: Store ID
  - name: city
    expr: city
    display_name: City
  - name: region
    expr: region
    display_name: Region
    comment: Operating region (Pacific Northwest, California, Southwest, Mountain).
    synonyms: [market, territory]
  - name: store_format
    expr: store_format
    display_name: Store Format
    comment: Flagship Cafe, Cafe, Drive-Thru, or Express Kiosk.
    synonyms: [format, store type]
  - name: is_comp_store
    expr: is_comp_store
    display_name: Comp Store
    comment: True when the store has been open 365+ days (comparable-store sales).
    synonyms: [comparable store, same store]
measures:
  - name: net_sales
    expr: SUM(net_sales)
    display_name: Net Sales
    comment: Gross sales minus discounts, pre-tax. Primary revenue KPI.
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: exact, places: 0}
      abbreviation: compact
    synonyms: [revenue, sales, total sales, turnover]
  - name: transactions
    expr: SUM(transactions)
    display_name: Transactions
    comment: Completed tickets, used as a proxy for customer traffic.
    format:
      type: number
      decimal_places: {type: exact, places: 0}
    synonyms: [tickets, orders, traffic, visits, customer count]
  - name: avg_ticket
    expr: SUM(net_sales) / SUM(transactions)
    display_name: Average Ticket
    comment: Net sales divided by transactions.
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: exact, places: 2}
    synonyms: [average order value, AOV, average check, basket size]
  - name: gross_profit
    expr: SUM(gross_profit)
    display_name: Gross Profit
    comment: Net sales minus cost of goods sold.
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: exact, places: 0}
      abbreviation: compact
    synonyms: [margin dollars, profit]
  - name: gross_margin_pct
    expr: SUM(gross_profit) / SUM(net_sales)
    display_name: Gross Margin %
    comment: Gross profit as a share of net sales.
    format:
      type: percentage
      decimal_places: {type: exact, places: 1}
    synonyms: [margin, GM%]
  - name: labor_cost_pct
    expr: SUM(labor_cost) / SUM(net_sales)
    display_name: Labor Cost %
    comment: Labor cost as a share of net sales. Target is below 30%.
    format:
      type: percentage
      decimal_places: {type: exact, places: 1}
    synonyms: [labor percent, labor ratio, wage ratio]
  - name: loyalty_mix_pct
    expr: SUM(loyalty_transactions) / SUM(transactions)
    display_name: Loyalty Mix %
    comment: Share of tickets with a Bean & Brew Rewards member attached.
    format:
      type: percentage
      decimal_places: {type: exact, places: 1}
    synonyms: [rewards penetration, loyalty share]
  - name: digital_mix_pct
    expr: SUM(digital_transactions) / SUM(transactions)
    display_name: Digital Mix %
    comment: Share of tickets ordered via Mobile App or Delivery.
    format:
      type: percentage
      decimal_places: {type: exact, places: 1}
    synonyms: [mobile share, digital share, app orders share]
  - name: discount_rate
    expr: SUM(discount_amount) / SUM(gross_sales)
    display_name: Discount Rate
    comment: Promotional discounts as a share of gross sales.
    format:
      type: percentage
      decimal_places: {type: exact, places: 2}
    synonyms: [promo rate, markdown rate]
  - name: store_days
    expr: COUNT(1)
    display_name: Store Operating Days
    comment: Number of store-days with sales.
$$;

CREATE OR REPLACE VIEW dbsql_workshop_gold.product_sales_metrics
WITH METRICS LANGUAGE YAML
COMMENT 'Certified Bean & Brew menu/product KPIs at line-item grain with star-schema joins to product and store dimensions. Use for product mix, category, daypart, and channel questions.'
AS $$
version: 1.1
source: dbsql_workshop_silver.fact_sales_line_items
comment: Product mix KPIs. Grain = one POS line item. Joins product and store dimensions (many-to-one, PK-enforced).
joins:
  - name: product
    source: dbsql_workshop_silver.dim_product
    on: source.product_id = product.product_id
    cardinality: many_to_one
    rely:
      at_most_one_match: true
  - name: store
    source: dbsql_workshop_silver.dim_store
    on: source.store_id = store.store_id
    cardinality: many_to_one
    rely:
      at_most_one_match: true
dimensions:
  - name: business_date
    expr: source.business_date
    display_name: Business Date
    synonyms: [date, day]
  - name: business_month
    expr: DATE_TRUNC('MONTH', source.business_date)
    display_name: Month
    synonyms: [month]
  - name: business_hour
    expr: source.business_hour
    display_name: Hour of Day
    comment: Hour (0-23) the ticket was rung up, store-local.
    synonyms: [hour, time of day]
  - name: daypart
    expr: |-
      CASE WHEN source.business_hour < 11 THEN 'Morning Rush (6-11)'
           WHEN source.business_hour < 14 THEN 'Midday (11-14)'
           WHEN source.business_hour < 17 THEN 'Afternoon (14-17)'
           ELSE 'Evening (17+)' END
    display_name: Daypart
    comment: Operational daypart bucket.
    synonyms: [shift, time of day bucket]
  - name: order_channel
    expr: source.order_channel
    display_name: Order Channel
    comment: In-Store, Mobile App, Drive-Thru, or Delivery.
    synonyms: [channel, ordering method]
  - name: product_name
    expr: product.product_name
    display_name: Product
    synonyms: [item, drink, menu item, sku]
  - name: category
    expr: product.category
    display_name: Category
    comment: Menu category (Espresso Drinks, Brewed Coffee, Cold Drinks, Tea, Bakery, Breakfast, Seasonal, Retail Beans).
    synonyms: [product category, menu category]
  - name: serving_type
    expr: product.serving_type
    display_name: Serving Type
    comment: Hot, Iced, Food, or Retail.
    synonyms: [hot or iced, temperature]
  - name: is_beverage
    expr: product.is_beverage
    display_name: Is Beverage
  - name: store_name
    expr: store.store_name
    display_name: Store
    synonyms: [store, location, cafe]
  - name: region
    expr: store.region
    display_name: Region
    synonyms: [market]
  - name: is_loyalty_member
    expr: source.loyalty_member_id IS NOT NULL
    display_name: Loyalty Member Purchase
    synonyms: [rewards member, member purchase]
measures:
  - name: units_sold
    expr: SUM(source.quantity)
    display_name: Units Sold
    comment: Total item quantity sold.
    format:
      type: number
      decimal_places: {type: exact, places: 0}
    synonyms: [units, quantity, items sold, cups sold]
  - name: net_sales
    expr: SUM(source.net_sales)
    display_name: Net Sales
    comment: Gross sales minus discounts, pre-tax.
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: exact, places: 0}
      abbreviation: compact
    synonyms: [revenue, sales]
  - name: gross_profit
    expr: SUM(source.net_sales) - SUM(source.cost_of_goods)
    display_name: Gross Profit
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: exact, places: 0}
      abbreviation: compact
    synonyms: [profit, margin dollars]
  - name: gross_margin_pct
    expr: (SUM(source.net_sales) - SUM(source.cost_of_goods)) / SUM(source.net_sales)
    display_name: Gross Margin %
    format:
      type: percentage
      decimal_places: {type: exact, places: 1}
    synonyms: [margin, GM%]
  - name: discount_amount
    expr: SUM(source.discount_amount)
    display_name: Discounts
    comment: Promotional discounts given (LOYALTY10, HAPPYHOUR).
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: exact, places: 0}
    synonyms: [promo cost, markdowns]
  - name: tickets
    expr: COUNT(DISTINCT source.transaction_id)
    display_name: Tickets Containing Item
    comment: Distinct tickets that included the item(s) in the current grouping.
    synonyms: [orders, attach count]
$$;

CREATE OR REPLACE VIEW dbsql_workshop_gold.customer_metrics
WITH METRICS LANGUAGE YAML
COMMENT 'Certified Bean & Brew Rewards loyalty KPIs: member counts, lifetime value, visit frequency, and mobile adoption by tier and segment.'
AS $$
version: 1.1
source: dbsql_workshop_gold.customer_360
comment: Loyalty member KPIs. Grain = one Bean & Brew Rewards member.
dimensions:
  - name: loyalty_tier
    expr: loyalty_tier
    display_name: Loyalty Tier
    comment: Green (entry), Gold, Reserve (top tier).
    synonyms: [tier, rewards tier, membership level]
  - name: customer_segment
    expr: customer_segment
    display_name: Customer Segment
    comment: Champion (top 10% spend), Loyal Regular, Occasional, Light (bottom 20%).
    synonyms: [segment, value segment, RFM segment]
  - name: favorite_product
    expr: favorite_product
    display_name: Favorite Product
    synonyms: [usual order, go-to drink]
  - name: favorite_store_id
    expr: favorite_store_id
    display_name: Favorite Store ID
  - name: signup_year
    expr: year(signup_date)
    display_name: Signup Year
    synonyms: [cohort, enrollment year]
measures:
  - name: member_count
    expr: COUNT(1)
    display_name: Members
    synonyms: [members, customers, number of members]
  - name: total_lifetime_value
    expr: SUM(lifetime_net_sales)
    display_name: Total Lifetime Value
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: exact, places: 0}
      abbreviation: compact
    synonyms: [member revenue, total LTV]
  - name: avg_lifetime_value
    expr: AVG(lifetime_net_sales)
    display_name: Avg Lifetime Value
    comment: Average net sales per member since 2025-01-01.
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: exact, places: 2}
    synonyms: [LTV, CLV, average customer value]
  - name: avg_visits
    expr: AVG(visits)
    display_name: Avg Visits per Member
    format:
      type: number
      decimal_places: {type: exact, places: 1}
    synonyms: [visit frequency, frequency]
  - name: avg_mobile_share
    expr: AVG(mobile_share)
    display_name: Avg Mobile Order Share
    format:
      type: percentage
      decimal_places: {type: exact, places: 1}
    synonyms: [app adoption, mobile adoption]
$$;
