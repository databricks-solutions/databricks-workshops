-- -- This file defines transformations for silver zone.
-- -- Keep this file small (<5 tables)
-- -- Use "+ Add" in the file browser to add more.

CREATE TEMPORARY VIEW flights_dlt_sql_fact_flight_vw AS
SELECT
    FlightKey as flight_key,
    UniqueCarrier as carrier_code,
    Origin as origin_location_code,
    Dest as destination_location_code,
    dd.Date_id as dim_date_id,
    event_date,
    event_timestamp,
    FlightNum as flight_num,
    (CASE 
        WHEN ArrDelay < 0 THEN 1
        WHEN ArrDelay = 0 THEN 2
        WHEN ArrDelay > 0 AND ArrDelay <= 30 THEN 4
        WHEN ArrDelay > 30 AND ArrDelay <= 60 THEN 8
        WHEN ArrDelay > 60 THEN 16
        ELSE 0
    END)
    | (CASE 
        WHEN DepDelay < 0 THEN 1
        WHEN DepDelay = 0 THEN 2
        WHEN DepDelay > 0 AND DepDelay <= 30 THEN 4
        WHEN DepDelay > 30 AND DepDelay <= 60 THEN 8
        WHEN DepDelay > 60 THEN 16
        ELSE 0
    END << 5)
    | (CASE 
        WHEN CarrierDelay < 0 THEN 1
        WHEN CarrierDelay = 0 THEN 2
        WHEN CarrierDelay > 0 AND CarrierDelay <= 30 THEN 4
        WHEN CarrierDelay > 30 AND CarrierDelay <= 60 THEN 8
        WHEN CarrierDelay > 60 THEN 16
        ELSE 0
    END << 10)
    | (CASE 
        WHEN WeatherDelay < 0 THEN 1
        WHEN WeatherDelay = 0 THEN 2
        WHEN WeatherDelay > 0 AND WeatherDelay <= 30 THEN 4
        WHEN WeatherDelay > 30 AND WeatherDelay <= 60 THEN 8
        WHEN WeatherDelay > 60 THEN 16
        ELSE 0
    END << 15)
    | (CASE 
        WHEN NASDelay < 0 THEN 1
        WHEN NASDelay = 0 THEN 2
        WHEN NASDelay > 0 AND NASDelay <= 30 THEN 4
        WHEN NASDelay > 30 AND NASDelay <= 60 THEN 8
        WHEN NASDelay > 60 THEN 16
        ELSE 0
    END << 20)
    | (CASE 
        WHEN SecurityDelay < 0 THEN 1
        WHEN SecurityDelay = 0 THEN 2
        WHEN SecurityDelay > 0 AND SecurityDelay <= 30 THEN 4
        WHEN SecurityDelay > 30 AND SecurityDelay <= 60 THEN 8
        WHEN SecurityDelay > 60 THEN 16
        ELSE 0
    END << 25)
    | (CASE 
        WHEN LateAircraftDelay < 0 THEN 1
        WHEN LateAircraftDelay = 0 THEN 2
        WHEN LateAircraftDelay > 0 AND LateAircraftDelay <= 30 THEN 4
        WHEN LateAircraftDelay > 30 AND LateAircraftDelay <= 60 THEN 8
        WHEN LateAircraftDelay > 60 THEN 16
        ELSE 0
    END << 30)
       as dim_delay_id,
    STRUCT (
        ArrDelay,
        DepDelay,
        CarrierDelay,
        WeatherDelay, 
        NASDelay,
        SecurityDelay,
        LateAircraftDelay, 
        IsArrDelayed,
        IsDepDelayed
    ) as delay,
    TaxiIn as taxi_in, 
    TaxiOut as taxi_out, 
    Diverted as diverted,
    Cancelled as cancelled, 
    CancellationCode as cancelled_code,
    Distance as distance,
    current_timestamp() as created_timestamp
FROM STREAM flights_silver
INNER JOIN dim_date dd
        ON event_date = dd.Date
;


CREATE OR REFRESH STREAMING TABLE fact_flight
CLUSTER BY (flight_key) 
TBLPROPERTIES ('delta.enableDeletionVectors' = true, 'delta.enableRowTracking' = false, 'delta.enableChangeDataFeed' = true);


CREATE FLOW fact_flight_flow AS AUTO CDC 
INTO fact_flight
FROM STREAM flights_dlt_sql_fact_flight_vw
KEYS (flight_key)
SEQUENCE BY event_timestamp
STORED AS SCD TYPE 1;
