-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Bean & Brew setup 2 of 3: load the silver tables
-- MAGIC
-- MAGIC Runs during `databricks bundle deploy`, after the tables are created, on the SQL warehouse. It generates the raw source
-- MAGIC extracts (12 stores, 22 menu items, 6,000 loyalty members, about 1.5M register tickets, inventory counts, labor hours
-- MAGIC and customer comments) as temporary views, with the demo scenarios built in, then loads the cleaned silver tables from
-- MAGIC them: types cast, IDs normalized, duplicate re-sends removed, JSON line items exploded. Only silver is kept.
-- MAGIC
-- MAGIC Every random value is a hash of the row key, so every run produces identical tables, and re-running replaces the rows
-- MAGIC (`INSERT OVERWRITE ... BY NAME`) without touching the tables, their keys or anything built on them. The job passes the
-- MAGIC bundle's catalog as the `catalog` parameter. Run it from the job, not interactively.

-- COMMAND ----------

-- The bundle created this catalog's dbsql_workshop_* schemas, and the previous task the tables in them
USE CATALOG IDENTIFIER(:catalog);

-- COMMAND ----------

-- Raw extract: store master (12 stores, 4 regions)
-- Raw store master extract from the Bean & Brew real-estate/operations system. One row per store location. Loaded as-is from CSV.
CREATE OR REPLACE TEMPORARY VIEW raw_stores AS SELECT * FROM VALUES
 ('S001','Bean & Brew - Pike Place','Seattle','WA','Pacific Northwest','Flagship Cafe','2019-04-12',4200,85,'Maya Chen',47.6097,-122.3422),
 ('S002','Bean & Brew - Capitol Hill','Seattle','WA','Pacific Northwest','Cafe','2020-06-01',2600,48,'Jordan Ellis',47.6253,-122.3222),
 ('S003','Bean & Brew - Pearl District','Portland','OR','Pacific Northwest','Cafe','2020-09-15',2400,42,'Sam Okafor',45.5276,-122.6814),
 ('S004','Bean & Brew - Mission','San Francisco','CA','California','Cafe','2021-02-20',2200,40,'Priya Raman',37.7599,-122.4148),
 ('S005','Bean & Brew - Financial District','San Francisco','CA','California','Express Kiosk','2021-08-05',650,6,'Luis Ortega',37.7946,-122.3999),
 ('S006','Bean & Brew - Santa Monica','Los Angeles','CA','California','Cafe','2022-01-10',2800,52,'Ava Thompson',34.0195,-118.4912),
 ('S007','Bean & Brew - Downtown Drive-Thru','Phoenix','AZ','Southwest','Drive-Thru','2022-05-18',1400,12,'Marcus Reed',33.4484,-112.0740),
 ('S008','Bean & Brew - South Congress','Austin','TX','Southwest','Cafe','2022-10-03',2500,46,'Elena Garcia',30.2500,-97.7500),
 ('S009','Bean & Brew - Domain Drive-Thru','Austin','TX','Southwest','Drive-Thru','2023-03-14',1300,10,'Tyler Brooks',30.4021,-97.7253),
 ('S010','Bean & Brew - LoDo','Denver','CO','Mountain','Cafe','2023-07-22',2300,44,'Hannah Kim',39.7527,-105.0000),
 ('S011','Bean & Brew - Union Station','Denver','CO','Mountain','Express Kiosk','2024-02-01',600,4,'Omar Haddad',39.7530,-105.0002),
 ('S012','Bean & Brew - RiNo','Denver','CO','Mountain','Cafe','2026-03-01',2700,50,'Grace Liu',39.7681,-104.9813)
AS t(store_id, store_name, city, state, region, store_format, open_date, square_feet, seating_capacity, store_manager, latitude, longitude);

-- COMMAND ----------

-- Raw extract: menu (22 items, 2025 list prices)
-- Raw menu/product catalog extract from the POS menu management system. One row per sellable item with list price and standard unit cost.
CREATE OR REPLACE TEMPORARY VIEW raw_products AS SELECT * FROM VALUES
 ('P001','Espresso','Espresso Drinks','Hot',3.25,0.60,false),
 ('P002','Americano','Espresso Drinks','Hot',3.75,0.70,false),
 ('P003','Cappuccino','Espresso Drinks','Hot',4.75,1.10,false),
 ('P004','Caffe Latte','Espresso Drinks','Hot',4.95,1.20,false),
 ('P005','Oat Milk Latte','Espresso Drinks','Hot',5.45,1.50,false),
 ('P006','Caramel Macchiato','Espresso Drinks','Hot',5.65,1.45,false),
 ('P007','Mocha','Espresso Drinks','Hot',5.45,1.40,false),
 ('P008','House Drip Coffee','Brewed Coffee','Hot',2.95,0.45,false),
 ('P009','Single Origin Pour Over','Brewed Coffee','Hot',4.50,0.90,false),
 ('P010','Cold Brew','Cold Drinks','Iced',4.75,0.80,false),
 ('P011','Nitro Cold Brew','Cold Drinks','Iced',5.25,0.95,false),
 ('P012','Iced Latte','Cold Drinks','Iced',5.25,1.25,false),
 ('P013','Iced Matcha Latte','Tea','Iced',5.75,1.60,false),
 ('P014','Chai Latte','Tea','Hot',4.95,1.15,false),
 ('P015','Butter Croissant','Bakery','Food',3.75,1.10,false),
 ('P016','Blueberry Muffin','Bakery','Food',3.50,0.95,false),
 ('P017','Chocolate Chip Cookie','Bakery','Food',2.75,0.60,false),
 ('P018','Egg & Cheese Breakfast Sandwich','Breakfast','Food',6.95,2.40,false),
 ('P019','Avocado Toast','Breakfast','Food',8.50,3.10,false),
 ('P020','Pumpkin Spice Latte','Seasonal','Hot',5.95,1.55,true),
 ('P021','House Blend Whole Bean 12oz','Retail Beans','Retail',15.95,6.50,false),
 ('P022','Ethiopia Yirgacheffe Whole Bean 12oz','Retail Beans','Retail',18.95,8.20,false)
AS t(product_id, product_name, category, temperature, list_price, unit_cost, is_seasonal);

-- COMMAND ----------

-- Raw extract: 6,000 loyalty members. Defects: upper-case emails (every 9th), lower-case home store IDs (every 13th)
-- Raw extract of the Bean & Brew Rewards loyalty program CRM. One row per enrolled member; contains PII (email) and messy casing.
CREATE OR REPLACE TEMPORARY VIEW raw_loyalty_members AS
WITH ids AS (
  SELECT id,
    pmod(xxhash64(id,'a'),1000000)/1e6 r1, pmod(xxhash64(id,'b'),1000000)/1e6 r2,
    pmod(xxhash64(id,'c'),1000000)/1e6 r3, pmod(xxhash64(id,'d'),1000000)/1e6 r4,
    pmod(xxhash64(id,'e'),1000000)/1e6 r5, pmod(xxhash64(id,'f'),1000000)/1e6 r6
  FROM range(1, 6001) AS r(id))
SELECT
  concat('C', lpad(cast(id AS STRING), 5, '0')) AS member_id,
  element_at(array('Olivia','Liam','Emma','Noah','Ava','Elijah','Sophia','Mateo','Isabella','Lucas','Mia','Levi','Amelia','Ezra','Harper','Kai','Luna','Leo','Aria','Hudson'), cast(floor(r1*20) AS INT)+1) AS first_name,
  element_at(array('Smith','Johnson','Nguyen','Garcia','Patel','Kim','Brown','Martinez','Lee','Davis','Lopez','Wilson','Chen','Anderson','Singh','Clark'), cast(floor(r2*16) AS INT)+1) AS last_name,
  CASE WHEN id % 9 = 0 THEN upper(concat('member', id, '@example.com')) ELSE concat('member', id, '@example.com') END AS email,
  cast(date_add(DATE'2019-06-01', cast(floor(r3*2650) AS INT)) AS STRING) AS signup_date,
  CASE WHEN id % 13 = 0 THEN lower(concat('S', lpad(cast(cast(floor(r4*11) AS INT)+1 AS STRING),3,'0')))
       ELSE concat('S', lpad(cast(cast(floor(r4*11) AS INT)+1 AS STRING),3,'0')) END AS home_store_id,
  CASE WHEN r5 < 0.60 THEN 'Green' WHEN r5 < 0.88 THEN 'Gold' ELSE 'Reserve' END AS tier,
  cast(floor(r6*12)+1 AS INT) AS birth_month
FROM ids;

-- COMMAND ----------

-- Raw extract: ~1.5M register tickets, 2025-01-01 to 2026-10-19, with the demo scenarios built in:
--   S007 traffic drop from 2026-10-15, S012 opening ramp, summer cold drinks, fall-only Pumpkin Spice Latte,
--   2026 price book from 2026-01-05. Defects: ~2% lower-case store IDs, ~0.3% re-sent duplicates, JSON line items.
-- Every random draw is a hash of the row key, so the table is identical on every run.
-- Raw point-of-sale transaction feed landed from store POS terminals. One row per ticket (with occasional duplicate re-sends). Line items arrive as a JSON string payload; timestamps are strings in store-local time.
CREATE OR REPLACE TEMPORARY VIEW raw_pos_transactions AS
WITH stores AS (
  SELECT store_id, store_format, to_date(open_date) open_dt,
    CASE store_id WHEN 'S001' THEN 260 WHEN 'S002' THEN 170 WHEN 'S003' THEN 150 WHEN 'S004' THEN 160 WHEN 'S005' THEN 140
      WHEN 'S006' THEN 190 WHEN 'S007' THEN 210 WHEN 'S008' THEN 175 WHEN 'S009' THEN 200 WHEN 'S010' THEN 150
      WHEN 'S011' THEN 120 ELSE 150 END AS base_traffic
  FROM raw_stores),
days AS (SELECT explode(sequence(DATE'2025-01-01', DATE'2026-10-19')) AS d),
arrs AS (SELECT
  flatten(array(array_repeat('P001',2),array_repeat('P002',3),array_repeat('P003',4),array_repeat('P004',6),array_repeat('P005',4),array_repeat('P006',4),array_repeat('P007',3),array_repeat('P008',6),array_repeat('P009',2),array_repeat('P010',2),array_repeat('P011',1),array_repeat('P012',2),array_repeat('P013',1),array_repeat('P014',3),array_repeat('P015',4),array_repeat('P016',3),array_repeat('P017',2),array_repeat('P018',4),array_repeat('P019',2),array_repeat('P021',1),array_repeat('P022',1))) AS cool_menu,
  flatten(array(array_repeat('P001',2),array_repeat('P002',2),array_repeat('P003',2),array_repeat('P004',3),array_repeat('P005',2),array_repeat('P006',2),array_repeat('P007',1),array_repeat('P008',3),array_repeat('P009',1),array_repeat('P010',6),array_repeat('P011',4),array_repeat('P012',6),array_repeat('P013',4),array_repeat('P014',1),array_repeat('P015',3),array_repeat('P016',3),array_repeat('P017',3),array_repeat('P018',3),array_repeat('P019',2),array_repeat('P021',1),array_repeat('P022',1))) AS warm_menu,
  array_repeat('P020',6) AS psl
),
store_days AS (
  SELECT s.store_id, s.store_format, d.d,
    cast(round(s.base_traffic
      * CASE dayofweek(d.d) WHEN 1 THEN 1.15 WHEN 7 THEN 1.25 WHEN 2 THEN 0.92 ELSE 1.0 END
      * (1 + 0.14 * datediff(d.d, DATE'2025-01-01') / 365.0)
      * CASE month(d.d) WHEN 12 THEN 1.15 WHEN 1 THEN 0.94 WHEN 2 THEN 0.95 WHEN 7 THEN 1.05 ELSE 1.0 END
      * CASE WHEN s.store_id = 'S012' THEN least(1.0, 0.45 + datediff(d.d, s.open_dt) / 200.0) ELSE 1.0 END
      * CASE WHEN s.store_id = 'S007' AND d.d >= DATE'2026-10-15' THEN 0.45 ELSE 1.0 END
      * CASE WHEN s.store_format = 'Express Kiosk' AND dayofweek(d.d) IN (1,7) THEN 0.6 ELSE 1.0 END
      -- daily noise of +/-12%: a hash of (store, day) instead of rand(), so the build is deterministic
      * (0.88 + 0.24 * pmod(xxhash64(s.store_id, d.d, 'traffic'), 1000000) / 1e6)) AS INT) AS n_orders
  FROM stores s JOIN days d ON d.d >= s.open_dt),
orders AS (
  SELECT sd.store_id, sd.store_format, sd.d, seq,
    concat('T', replace(sd.store_id,'S',''), date_format(sd.d,'yyMMdd'), lpad(cast(seq AS STRING),4,'0')) AS transaction_id
  FROM store_days sd LATERAL VIEW explode(sequence(1, sd.n_orders)) e AS seq),
scored AS (
  SELECT o.*,
    pmod(xxhash64(transaction_id,'h'),1000000)/1e6 rh, pmod(xxhash64(transaction_id,'m'),1000000)/1e6 rm,
    pmod(xxhash64(transaction_id,'c'),1000000)/1e6 rc, pmod(xxhash64(transaction_id,'l'),1000000)/1e6 rl,
    pmod(xxhash64(transaction_id,'p'),1000000)/1e6 rp, pmod(xxhash64(transaction_id,'n'),1000000)/1e6 rn,
    pmod(xxhash64(transaction_id,'v'),1000000)/1e6 rv, pmod(xxhash64(transaction_id,'q'),1000000)/1e6 rq,
    pmod(xxhash64(transaction_id,'k'),1000000)/1e6 rk
  FROM orders o),
base AS (
SELECT
  transaction_id,
  CASE WHEN rk < 0.02 THEN lower(store_id) ELSE store_id END AS store_id,
  concat('REG-', cast(cast(floor(rq*3) AS INT)+1 AS STRING)) AS register_id,
  date_format(
    timestamp_seconds(unix_timestamp(cast(d AS TIMESTAMP))
      + element_at(array(6,6,7,7,7,7,8,8,8,8,8,9,9,9,10,10,11,11,12,12,13,14,14,15,16,17,18,19), cast(floor(rh*28) AS INT)+1) * 3600
      + cast(floor(rm*3600) AS INT)),
    'yyyy-MM-dd HH:mm:ss') AS transaction_ts,
  CASE
    WHEN store_format = 'Drive-Thru' THEN CASE WHEN rc < 0.65 THEN 'Drive-Thru' WHEN rc < 0.87 THEN 'Mobile App' ELSE 'In-Store' END
    WHEN store_format = 'Express Kiosk' THEN CASE WHEN rc < 0.62 THEN 'In-Store' ELSE 'Mobile App' END
    ELSE CASE WHEN rc < 0.66 THEN 'In-Store' WHEN rc < 0.94 THEN 'Mobile App' ELSE 'Delivery' END
  END AS order_channel,
  CASE WHEN rl < 0.55 THEN concat('C', lpad(cast(cast(floor(rl/0.55*6000) AS INT)+1 AS STRING),5,'0')) ELSE NULL END AS loyalty_member_id,
  CASE WHEN rp < 0.58 THEN 'Credit Card' WHEN rp < 0.83 THEN 'Mobile Wallet' WHEN rp < 0.93 THEN 'Cash' ELSE 'Gift Card' END AS payment_method,
  CASE WHEN rv < 0.04 THEN 'LOYALTY10' WHEN rv < 0.06 THEN 'HAPPYHOUR' ELSE NULL END AS promo_code,
  CASE WHEN rv > 0.988 THEN 'VOIDED' ELSE 'COMPLETED' END AS ticket_status,
  to_json(transform(
    sequence(1, CASE WHEN rn < 0.55 THEN 1 WHEN rn < 0.85 THEN 2 WHEN rn < 0.97 THEN 3 ELSE 4 END),
    i -> named_struct(
      'line', i,
      'sku', element_at(
          CASE WHEN month(d) BETWEEN 5 AND 9 AND month(d) <> 9 THEN warm_menu
               WHEN month(d) IN (9,10,11) THEN concat(cool_menu, psl)
               ELSE cool_menu END,
          cast(pmod(xxhash64(transaction_id, i), CASE WHEN month(d) BETWEEN 5 AND 8 THEN size(warm_menu) WHEN month(d) IN (9,10,11) THEN size(cool_menu)+size(psl) ELSE size(cool_menu) END) AS INT) + 1),
      'qty', CASE WHEN pmod(xxhash64(transaction_id, i, 'qty'), 100) < 88 THEN 1 ELSE 2 END
    ))) AS line_items_json,
  CASE WHEN d >= DATE'2026-01-05' THEN 'menu_v2026' ELSE 'menu_v2025' END AS price_book,
  concat('s3://beanbrew-landing/pos/', date_format(d,'yyyy/MM/dd'), '/', store_id, '.json') AS _source_file,
  timestamp_seconds(unix_timestamp(cast(date_add(d,1) AS TIMESTAMP)) + 7200) AS _ingested_at
FROM scored CROSS JOIN arrs)
-- Simulate POS re-send duplicates: about 0.3% of tickets arrive twice
SELECT * EXCEPT (copy) FROM base
LATERAL VIEW explode(sequence(1, 1 + CAST(pmod(xxhash64(transaction_id, 'dup'), 1000) < 3 AS INT))) e AS copy;

-- COMMAND ----------

-- Raw extract: end-of-day inventory counts for the last ~6 months.
-- Scenario: oat milk at S004 and S008 and espresso beans at S010 are below reorder point on 2026-10-19.
-- Raw end-of-day inventory counts submitted by store shift leads via the back-of-house app. One row per store, stock item, and count date. Units vary by item.
CREATE OR REPLACE TEMPORARY VIEW raw_inventory_counts AS
WITH items AS (
  SELECT * FROM VALUES
   ('INV01','Espresso Beans','kg',40.0,12.0,18.50),
   ('INV02','Whole Milk','liter',120.0,35.0,1.10),
   ('INV03','Oat Milk','liter',60.0,18.0,2.60),
   ('INV04','Cold Brew Concentrate','liter',45.0,12.0,4.20),
   ('INV05','Hot Cups 12oz','unit',2500.0,700.0,0.09),
   ('INV06','Pastry Case Items','unit',160.0,40.0,1.05)
  AS t(item_code, item_name, unit_of_measure, par_level, reorder_point, unit_cost)),
st AS (SELECT store_id, store_format, to_date(open_date) open_dt FROM raw_stores),
days AS (SELECT explode(sequence(DATE'2026-04-23', DATE'2026-10-19')) AS d)
SELECT
  s.store_id, d.d AS count_date, i.item_code, i.item_name, i.unit_of_measure,
  round(CASE
    WHEN d.d = DATE'2026-10-19' AND s.store_id IN ('S004','S008') AND i.item_code = 'INV03' THEN i.reorder_point * 0.35
    WHEN d.d = DATE'2026-10-19' AND s.store_id = 'S010' AND i.item_code = 'INV01' THEN i.reorder_point * 0.6
    ELSE i.par_level * CASE s.store_format WHEN 'Express Kiosk' THEN 0.5 WHEN 'Flagship Cafe' THEN 1.5 ELSE 1.0 END
      * (1 - (pmod(datediff(d.d, DATE'2026-01-01') + pmod(xxhash64(s.store_id, i.item_code), 3), 4) / 4.0) * 0.72)
      * (0.92 + 0.16 * pmod(xxhash64(s.store_id, i.item_code, d.d), 1000) / 1000.0)
  END, 1) AS on_hand_qty,
  i.par_level * CASE s.store_format WHEN 'Express Kiosk' THEN 0.5 WHEN 'Flagship Cafe' THEN 1.5 ELSE 1.0 END AS par_level,
  i.reorder_point * CASE s.store_format WHEN 'Express Kiosk' THEN 0.5 WHEN 'Flagship Cafe' THEN 1.5 ELSE 1.0 END AS reorder_point,
  i.unit_cost,
  concat('shiftlead_', lower(s.store_id)) AS counted_by,
  -- fixed load time (next morning) instead of current_timestamp(), so reruns are identical
  timestamp_seconds(unix_timestamp(cast(date_add(d.d,1) AS TIMESTAMP)) + 7200) AS _ingested_at
FROM st s CROSS JOIN items i JOIN days d ON d.d >= s.open_dt;

-- COMMAND ----------

-- Raw extract: daily labor per store. Scenario: Pike Place (Flagship) is the only store over the 30% labor target.
-- Flagship base hours are 46 (the original used 44 and landed over 30% only by random chance).
-- Raw daily labor summary exported from the workforce management (WFM) system. One row per store per business day with scheduled vs. worked hours and blended wage.
CREATE OR REPLACE TEMPORARY VIEW raw_labor_shifts AS
WITH st AS (
  SELECT store_id, store_format, to_date(open_date) open_dt,
    CASE store_format WHEN 'Flagship Cafe' THEN 46 WHEN 'Cafe' THEN 26 WHEN 'Drive-Thru' THEN 25 ELSE 12 END base_hours,
    CASE state WHEN 'CA' THEN 21.50 WHEN 'WA' THEN 21.00 WHEN 'CO' THEN 18.75 WHEN 'OR' THEN 18.25 ELSE 16.25 END wage
  FROM raw_stores),
days AS (SELECT explode(sequence(DATE'2025-01-01', DATE'2026-10-19')) AS d)
SELECT s.store_id, d.d AS business_date,
  round(s.base_hours * CASE WHEN dayofweek(d.d) IN (1,7) THEN 1.15 ELSE 1.0 END, 1) AS scheduled_hours,
  round(s.base_hours * CASE WHEN dayofweek(d.d) IN (1,7) THEN 1.15 ELSE 1.0 END * (0.93 + 0.14 * pmod(xxhash64(s.store_id, d.d),1000)/1000.0), 1) AS worked_hours,
  cast(greatest(2, round(s.base_hours / 6.5)) AS INT) AS headcount,
  round(s.wage * CASE WHEN d.d >= DATE'2026-01-01' THEN 1.04 ELSE 1.0 END, 2) AS avg_hourly_wage,
  timestamp_seconds(unix_timestamp(cast(date_add(d.d,1) AS TIMESTAMP)) + 7200) AS _ingested_at
FROM st s JOIN days d ON d.d >= s.open_dt;

-- COMMAND ----------

-- Silver dimensions: typed, trimmed, IDs and emails normalized
INSERT OVERWRITE dbsql_workshop_silver.dim_store BY NAME
SELECT upper(trim(store_id)) AS store_id, store_name, city, state, region, store_format,
  to_date(open_date) AS open_date, cast(square_feet AS INT) AS square_feet, cast(seating_capacity AS INT) AS seating_capacity,
  store_manager, cast(latitude AS DOUBLE) latitude, cast(longitude AS DOUBLE) longitude
FROM raw_stores;

INSERT OVERWRITE dbsql_workshop_silver.dim_product BY NAME
SELECT product_id, product_name, category, temperature AS serving_type,
  cast(list_price AS DECIMAL(10,2)) AS list_price_2025,
  cast(round(list_price * 1.05, 2) AS DECIMAL(10,2)) AS list_price_2026,
  cast(unit_cost AS DECIMAL(10,2)) AS unit_cost,
  cast(round((list_price - unit_cost) / list_price, 4) AS DECIMAL(6,4)) AS standard_margin_pct,
  cast(is_seasonal AS BOOLEAN) AS is_seasonal,
  category IN ('Espresso Drinks','Brewed Coffee','Cold Drinks','Tea','Seasonal') AS is_beverage
FROM raw_products;

INSERT OVERWRITE dbsql_workshop_silver.dim_loyalty_member BY NAME
SELECT member_id, first_name, last_name, lower(trim(email)) AS email, to_date(signup_date) AS signup_date,
  upper(home_store_id) AS home_store_id, tier AS loyalty_tier, birth_month
FROM raw_loyalty_members;

-- COMMAND ----------

-- Silver facts: duplicates removed, voids excluded from line items, JSON exploded, price book and promos applied
INSERT OVERWRITE dbsql_workshop_silver.fact_sales_line_items BY NAME
WITH dedup AS (
  SELECT * EXCEPT (rn) FROM (
    SELECT *, row_number() OVER (PARTITION BY transaction_id ORDER BY _ingested_at DESC) rn FROM raw_pos_transactions) WHERE rn = 1),
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
    SELECT *, row_number() OVER (PARTITION BY transaction_id ORDER BY _ingested_at DESC) rn FROM raw_pos_transactions) WHERE rn = 1),
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
FROM raw_inventory_counts;

INSERT OVERWRITE dbsql_workshop_silver.fact_labor_daily BY NAME
SELECT upper(store_id) store_id, business_date, cast(scheduled_hours AS DECIMAL(8,1)) scheduled_hours, cast(worked_hours AS DECIMAL(8,1)) worked_hours,
  headcount, cast(avg_hourly_wage AS DECIMAL(8,2)) avg_hourly_wage, cast(worked_hours * avg_hourly_wage AS DECIMAL(12,2)) AS labor_cost
FROM raw_labor_shifts;

-- COMMAND ----------

-- Customer feedback: about one comment per store every four days, plus the scenario comments:
--   S007 drive-thru lane closed from 2026-10-15, oat milk out of stock at S004 and S008 from 2026-10-10,
--   price comments after the 2026-01-05 increase, Pumpkin Spice Latte in the fall.
-- Free text on purpose: the workshop's gold model reads it with AI functions.
INSERT OVERWRITE dbsql_workshop_silver.customer_feedback BY NAME
WITH store_days AS (
  SELECT s.store_id, d.d,
    CASE WHEN s.store_id = 'S007' AND d.d >= DATE'2026-10-15' THEN 2
         WHEN s.store_id IN ('S004','S008') AND d.d >= DATE'2026-10-10' THEN 1
         WHEN pmod(xxhash64(s.store_id, d.d, 'fb'), 1000) < 250 THEN 1
         ELSE 0 END AS n_comments
  FROM dbsql_workshop_silver.dim_store s
  JOIN (SELECT explode(sequence(DATE'2025-01-01', DATE'2026-10-19')) AS d) d ON d.d >= s.open_date),
comments AS (
  SELECT store_id, d, seq,
    pmod(xxhash64(store_id, d, seq, 'kind'), 1000) / 1000.0 AS r_kind,
    pmod(xxhash64(store_id, d, seq, 'tone'), 1000) / 1000.0 AS r_tone,
    cast(pmod(xxhash64(store_id, d, seq, 'text'), 1000) AS INT) AS r_text,
    cast(pmod(xxhash64(store_id, d, seq, 'item'), 1000) AS INT) AS r_item,
    pmod(xxhash64(store_id, d, seq, 'star'), 1000) / 1000.0 AS r_star,
    pmod(xxhash64(store_id, d, seq, 'chan'), 1000) / 1000.0 AS r_chan
  FROM store_days LATERAL VIEW explode(sequence(1, n_comments)) e AS seq
  WHERE n_comments > 0),
kinds AS (
  SELECT *,
    CASE WHEN store_id = 'S007' AND d >= DATE'2026-10-15' THEN 'drive_thru'
         WHEN store_id IN ('S004','S008') AND d >= DATE'2026-10-10' AND r_kind < 0.7 THEN 'oat_milk'
         WHEN d BETWEEN DATE'2026-01-05' AND DATE'2026-03-15' AND r_kind < 0.3 THEN 'price'
         WHEN month(d) IN (9, 10, 11) AND r_kind < 0.25 THEN 'psl'
         WHEN r_tone < 0.55 THEN 'positive'
         WHEN r_tone < 0.80 THEN 'neutral'
         ELSE 'negative' END AS kind
  FROM comments),
templates AS (
  SELECT * FROM VALUES
    ('drive_thru', array(
      'The drive-thru lane was closed for construction, so I had to park and wait inside for my {item}.',
      'Drive-thru blocked off with cones again. Gave up and went to another coffee shop.',
      'No drive-thru this week because of road work. The line inside took 20 minutes.',
      'Construction crews have the drive-thru entrance closed. Please put up a sign about where to order.')),
    ('oat_milk', array(
      'Out of oat milk again. Had to get my {item} with whole milk.',
      'No oat milk for the second day in a row. Please restock!',
      'Asked for an Oat Milk Latte and they were out of oat milk. Disappointing.')),
    ('price', array(
      'Prices went up again. A {item} costs too much now.',
      'Noticed the new menu prices. Still good, but my {item} is pricier than last year.',
      'Love the coffee, but the price increase on the {item} is hard to justify.')),
    ('psl', array(
      'Pumpkin Spice Latte is back and it is perfect.',
      'Came in just for the Pumpkin Spice Latte. Worth it.',
      'Fall menu is great. The Pumpkin Spice Latte tastes just like last year.')),
    ('positive', array(
      'Great {item} today, and the barista remembered my name.',
      'Fast service even with a line out the door. My {item} was perfect.',
      'Quiet spot to work, good wifi and a really good {item}.',
      'Mobile order was ready when I walked in. {item} was spot on.',
      'Friendly staff and a great {item}. My favorite Bean & Brew.')),
    ('neutral', array(
      'The {item} was fine, nothing special. Store was busy.',
      'Okay visit. Had to wait a few minutes for my {item}.',
      'Decent {item}, but the music was too loud to talk.')),
    ('negative', array(
      'My {item} was lukewarm and the order was wrong.',
      'Waited over 15 minutes for a {item} at the counter.',
      'Tables were dirty and the trash was overflowing.',
      'The mobile app charged me twice for one {item}.'))
  AS t(kind, texts)),
items AS (
  SELECT array('Caffe Latte','Cold Brew','Cappuccino','Caramel Macchiato','Mocha','House Drip Coffee','Iced Latte',
               'Chai Latte','Oat Milk Latte','Nitro Cold Brew','Americano','Blueberry Muffin','Butter Croissant') AS names)
SELECT
  concat('F', replace(k.store_id, 'S', ''), date_format(k.d, 'yyMMdd'), cast(k.seq AS STRING)) AS feedback_id,
  k.store_id,
  k.d AS submitted_date,
  CASE WHEN k.r_chan < 0.6 THEN 'App review' ELSE 'Receipt survey' END AS channel,
  CASE WHEN k.kind IN ('positive', 'psl') THEN CASE WHEN k.r_star < 0.7 THEN 5 ELSE 4 END
       WHEN k.kind = 'neutral' THEN 3
       WHEN k.kind = 'price' THEN CASE WHEN k.r_star < 0.5 THEN 3 ELSE 2 END
       ELSE CASE WHEN k.r_star < 0.6 THEN 1 ELSE 2 END END AS rating,
  replace(element_at(tp.texts, pmod(k.r_text, size(tp.texts)) + 1), '{item}', element_at(i.names, pmod(k.r_item, size(i.names)) + 1)) AS comment
FROM kinds k JOIN templates tp USING (kind) CROSS JOIN items i;

-- COMMAND ----------

-- Stops the deploy if silver didn't load as expected. Every run produces the same rows, so the counts are exact.
SELECT CASE
  WHEN (SELECT count(*) FROM dbsql_workshop_silver.dim_store) <> 12
    OR (SELECT count(*) FROM dbsql_workshop_silver.dim_product) <> 22
    OR (SELECT count(*) FROM dbsql_workshop_silver.dim_loyalty_member) <> 6000
    OR (SELECT count(*) FROM dbsql_workshop_silver.fact_transactions) <> 1504178
    OR (SELECT count(*) - count(DISTINCT transaction_id) FROM dbsql_workshop_silver.fact_transactions) <> 0
    OR (SELECT count_if(store_id <> upper(store_id)) FROM dbsql_workshop_silver.fact_transactions) <> 0
    OR (SELECT count(*) FROM dbsql_workshop_silver.customer_feedback WHERE submitted_date >= DATE'2026-09-22') = 0
  THEN raise_error('Silver did not load as expected. Re-run the setup job: [dbsql_workshop] Bean & Brew - Set up demo data')
  ELSE 'Silver loaded' END AS silver_check
