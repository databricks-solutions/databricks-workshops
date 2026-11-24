-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Bean & Brew setup 3 of 3: column comments and tags
-- MAGIC
-- MAGIC Runs during `databricks bundle deploy`, after silver is loaded, on the SQL warehouse. Column comments document the data
-- MAGIC for people and for Genie; tags classify schemas and tables by layer, domain and owner, and flag email as PII. Both show
-- MAGIC in **Catalog**. Re-running sets the same values again.

-- COMMAND ----------

USE CATALOG IDENTIFIER(:catalog);

-- COMMAND ----------

-- Column comments, grouped by table
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

ALTER TABLE dbsql_workshop_silver.customer_feedback ALTER COLUMN
  channel COMMENT 'Where the comment came from: App review or Receipt survey.',
  rating COMMENT 'Star rating, 1 (worst) to 5 (best).',
  comment COMMENT 'Free-text customer comment, as written. Unstructured: read it with AI functions such as ai_extract and ai_classify.';

ALTER TABLE dbsql_workshop_gold.feedback_insights ALTER COLUMN
  comment COMMENT 'Free-text customer comment, as written.',
  sentiment COMMENT 'positive, negative, neutral or mixed, from ai_analyze_sentiment(comment).',
  topic COMMENT 'One of: Speed of service, Store access, Product availability, Price, Product quality, Staff, Store experience. From ai_classify(comment).',
  product_mentioned COMMENT 'Menu item the comment mentions, from ai_extract(comment). NULL if none.',
  issue COMMENT 'The problem the customer describes, in a few words, from ai_extract(comment). NULL if none.';

-- COMMAND ----------

-- Tags. A governed tag policy in your account may reject some of these values: the handler skips any statement that
-- fails, so tags never stop the deploy.
BEGIN
  DECLARE skipped INT DEFAULT 0;
  DECLARE CONTINUE HANDLER FOR SQLEXCEPTION SET skipped = skipped + 1;
  ALTER SCHEMA dbsql_workshop_silver SET TAGS ('layer' = 'silver', 'domain' = 'sales');
  ALTER SCHEMA dbsql_workshop_gold SET TAGS ('layer' = 'gold', 'domain' = 'sales');
  ALTER TABLE dbsql_workshop_silver.fact_inventory_daily SET TAGS ('domain' = 'supply_chain');
  ALTER TABLE dbsql_workshop_gold.inventory_current_status SET TAGS ('domain' = 'supply_chain');
  ALTER TABLE dbsql_workshop_silver.fact_labor_daily SET TAGS ('domain' = 'hr');
  ALTER TABLE dbsql_workshop_silver.customer_feedback SET TAGS ('domain' = 'operations');
  ALTER TABLE dbsql_workshop_gold.feedback_insights SET TAGS ('domain' = 'operations');
  ALTER TABLE dbsql_workshop_gold.feedback_insights SET TAGS ('owner_team' = 'customer_data');
  ALTER TABLE dbsql_workshop_silver.dim_loyalty_member ALTER COLUMN email SET TAGS ('contains_pii' = 'true');
  ALTER TABLE dbsql_workshop_gold.store_daily_performance SET TAGS ('owner_team' = 'analytics');
  ALTER TABLE dbsql_workshop_gold.customer_360 SET TAGS ('owner_team' = 'customer_data');
  SELECT 11 - skipped AS tag_statements_applied, skipped AS tag_statements_skipped;
END
