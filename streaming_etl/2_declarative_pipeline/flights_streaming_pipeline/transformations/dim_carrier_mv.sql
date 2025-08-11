CREATE MATERIALIZED VIEW dim_carrier AS
WITH carrier as (
  SELECT 
    *,
    ROW_NUMBER() OVER(PARTITION BY coalesce(UNIQUE_CARRIER, CARRIER) ORDER BY coalesce(thru_date_source, '2099-12-31') DESC ) as rownum
  FROM LIVE.dlt_py_carrier_silver
)
SELECT
    coalesce(UNIQUE_CARRIER, CARRIER) unique_carrier_code,
    CARRIER as carrier_code,
    airline_id,
    carrier_entity,
    carrier_name,
    unique_carrier_entity,
    unique_carrier_name,
    wac,
    carrier_group,
    carrier_group_new,
    region,
    start_date_source,
    thru_date_source as end_date_source,
    case when rownum = 1 then True else False end as is_current
FROM carrier
WHERE coalesce(UNIQUE_CARRIER, CARRIER) is not null