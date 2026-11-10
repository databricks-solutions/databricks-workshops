-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Bean & Brew setup 1 of 2: load the raw (bronze) data
-- MAGIC
-- MAGIC Runs during `databricks bundle deploy`, as the first task of the bundle's setup job, on the SQL warehouse. It loads six
-- MAGIC raw tables (about 1.5M register tickets) with the defects and demo scenarios built in. Every random value is a hash of
-- MAGIC the row key, so every run produces identical tables. Tables that already exist are left as they are.
-- MAGIC
-- MAGIC The job passes the bundle's catalog as the `catalog` parameter. Run it from the job, not interactively.

-- COMMAND ----------

-- The bundle created this catalog's dbsql_workshop_* schemas just before this job runs
USE CATALOG IDENTIFIER(:catalog);

-- COMMAND ----------

-- Bronze: store master (12 stores, 4 regions)
CREATE TABLE IF NOT EXISTS dbsql_workshop_bronze.raw_stores
COMMENT 'Raw store master extract from the Bean & Brew real-estate/operations system. One row per store location. Loaded as-is from CSV.'
AS SELECT * FROM VALUES
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

-- Bronze: menu (22 items, 2025 list prices)
CREATE TABLE IF NOT EXISTS dbsql_workshop_bronze.raw_products
COMMENT 'Raw menu/product catalog extract from the POS menu management system. One row per sellable item with list price and standard unit cost.'
AS SELECT * FROM VALUES
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

-- Bronze: 6,000 loyalty members. Defects: upper-case emails (every 9th), lower-case home store IDs (every 13th)
CREATE TABLE IF NOT EXISTS dbsql_workshop_bronze.raw_loyalty_members
COMMENT 'Raw extract of the Bean & Brew Rewards loyalty program CRM. One row per enrolled member; contains PII (email) and messy casing.'
AS
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

-- Bronze: ~1.5M register tickets, 2025-01-01 to 2026-10-19, with the demo scenarios built in:
--   S007 traffic drop from 2026-10-15, S012 opening ramp, summer cold drinks, fall-only Pumpkin Spice Latte,
--   2026 price book from 2026-01-05. Defects: ~2% lower-case store IDs, ~0.3% re-sent duplicates, JSON line items.
-- Every random draw is a hash of the row key, so the table is identical on every run.
CREATE TABLE IF NOT EXISTS dbsql_workshop_bronze.raw_pos_transactions
COMMENT 'Raw point-of-sale transaction feed landed from store POS terminals. One row per ticket (with occasional duplicate re-sends). Line items arrive as a JSON string payload; timestamps are strings in store-local time.'
AS
WITH stores AS (
  SELECT store_id, store_format, to_date(open_date) open_dt,
    CASE store_id WHEN 'S001' THEN 260 WHEN 'S002' THEN 170 WHEN 'S003' THEN 150 WHEN 'S004' THEN 160 WHEN 'S005' THEN 140
      WHEN 'S006' THEN 190 WHEN 'S007' THEN 210 WHEN 'S008' THEN 175 WHEN 'S009' THEN 200 WHEN 'S010' THEN 150
      WHEN 'S011' THEN 120 ELSE 150 END AS base_traffic
  FROM dbsql_workshop_bronze.raw_stores),
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

-- Bronze: end-of-day inventory counts for the last ~6 months.
-- Scenario: oat milk at S004 and S008 and espresso beans at S010 are below reorder point on 2026-10-19.
CREATE TABLE IF NOT EXISTS dbsql_workshop_bronze.raw_inventory_counts
COMMENT 'Raw end-of-day inventory counts submitted by store shift leads via the back-of-house app. One row per store, stock item, and count date. Units vary by item.'
AS
WITH items AS (
  SELECT * FROM VALUES
   ('INV01','Espresso Beans','kg',40.0,12.0,18.50),
   ('INV02','Whole Milk','liter',120.0,35.0,1.10),
   ('INV03','Oat Milk','liter',60.0,18.0,2.60),
   ('INV04','Cold Brew Concentrate','liter',45.0,12.0,4.20),
   ('INV05','Hot Cups 12oz','unit',2500.0,700.0,0.09),
   ('INV06','Pastry Case Items','unit',160.0,40.0,1.05)
  AS t(item_code, item_name, unit_of_measure, par_level, reorder_point, unit_cost)),
st AS (SELECT store_id, store_format, to_date(open_date) open_dt FROM dbsql_workshop_bronze.raw_stores),
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

-- Bronze: daily labor per store. Scenario: Pike Place (Flagship) is the only store over the 30% labor target.
-- Flagship base hours are 46 (the original used 44 and landed over 30% only by random chance).
CREATE TABLE IF NOT EXISTS dbsql_workshop_bronze.raw_labor_shifts
COMMENT 'Raw daily labor summary exported from the workforce management (WFM) system. One row per store per business day with scheduled vs. worked hours and blended wage.'
AS
WITH st AS (
  SELECT store_id, store_format, to_date(open_date) open_dt,
    CASE store_format WHEN 'Flagship Cafe' THEN 46 WHEN 'Cafe' THEN 26 WHEN 'Drive-Thru' THEN 25 ELSE 12 END base_hours,
    CASE state WHEN 'CA' THEN 21.50 WHEN 'WA' THEN 21.00 WHEN 'CO' THEN 18.75 WHEN 'OR' THEN 18.25 ELSE 16.25 END wage
  FROM dbsql_workshop_bronze.raw_stores),
days AS (SELECT explode(sequence(DATE'2025-01-01', DATE'2026-10-19')) AS d)
SELECT s.store_id, d.d AS business_date,
  round(s.base_hours * CASE WHEN dayofweek(d.d) IN (1,7) THEN 1.15 ELSE 1.0 END, 1) AS scheduled_hours,
  round(s.base_hours * CASE WHEN dayofweek(d.d) IN (1,7) THEN 1.15 ELSE 1.0 END * (0.93 + 0.14 * pmod(xxhash64(s.store_id, d.d),1000)/1000.0), 1) AS worked_hours,
  cast(greatest(2, round(s.base_hours / 6.5)) AS INT) AS headcount,
  round(s.wage * CASE WHEN d.d >= DATE'2026-01-01' THEN 1.04 ELSE 1.0 END, 2) AS avg_hourly_wage,
  timestamp_seconds(unix_timestamp(cast(date_add(d.d,1) AS TIMESTAMP)) + 7200) AS _ingested_at
FROM st s JOIN days d ON d.d >= s.open_dt;
