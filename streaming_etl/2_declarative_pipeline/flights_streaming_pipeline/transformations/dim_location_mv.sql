CREATE OR REFRESH STREAMING TABLE location_src_st AS 
WITH locations as (
  SELECT 
    Origin AS location_code,
    event_timestamp
  FROM STREAM flights_silver
  UNION ALL
  SELECT 
    Dest AS location_code,
    event_timestamp
  FROM STREAM flights_silver
)
SELECT
  location_code,
  location_code as location_name,
  max(event_timestamp) as last_updated_timestamp
FROM locations
GROUP BY ALL;


CREATE OR REFRESH MATERIALIZED VIEW dim_location
(
  location_code STRING,
  location_name STRING,
  last_updated_timestamp TIMESTAMP
)
TBLPROPERTIES ('delta.enableDeletionVectors' = true, 'delta.enableRowTracking' = false, 'delta.enableChangeDataFeed' = true)
AS
SELECT 
  location_code,
  location_name,
  max(last_updated_timestamp) as last_updated_timestamp
FROM location_src_st
GROUP BY ALL
