-- This file defines transformations for silver zone.
-- Keep this file small (<5 tables)
-- Use "+ Add" in the file browser to add more.

CREATE STREAMING TABLE flights_silver  (
  FlightKey STRING,
  UniqueCarrier STRING,
  FlightNum INT,
  TailNum STRING,
  event_date DATE,
  event_timestamp TIMESTAMP,
  Year INT,
  Month INT,
  DayofMonth INT,
  DayOfWeek INT,
  CRSDepTime STRING,
  DepTime STRING,
  CRSArrTime STRING,
  ArrTime STRING,
  CRSElapsedTime INT,
  ActualElapsedTime STRING,
  AirTime STRING,
  ArrDelay INT,
  DepDelay INT,
  Origin STRING,
  Dest STRING,
  Distance STRING,
  TaxiIn STRING,
  TaxiOut STRING,
  Cancelled INT,
  CancellationCode STRING,
  Diverted INT,
  CarrierDelay INT,
  WeatherDelay INT,
  NASDelay INT,
  SecurityDelay INT,
  LateAircraftDelay INT,
  IsArrDelayed STRING,
  IsDepDelayed STRING,
  created_timestamp TIMESTAMP)
CLUSTER BY (FlightKey)
TBLPROPERTIES ('delta.enableDeletionVectors' = true, 'delta.enableRowTracking' = false)
AS 
WITH flights_cleaned as (
 SELECT
  * EXCEPT (DepTime, CRSDepTime, ArrTime, CRSArrTime),
  lpad(replace(DepTime, 'NA', ''), 4, '0') DepTime,
  lpad(replace(CRSDepTime, 'NA', ''), 4, '0') CRSDepTime,
  lpad(replace(ArrTime, 'NA', ''), 4, '0') ArrTime,
  lpad(replace(CRSArrTime, 'NA', ''), 4, '0') CRSArrTime
FROM STREAM flights_bronze
 )
  SELECT
      FlightKey, 
      UniqueCarrier, 
      FlightNum, 
      TailNum, 
      event_date, 
      try_to_timestamp(concat(event_date, ' ', left(CRSDepTime,2), ':', right(CRSDepTime, 2))) as event_timestamp,
      Year, 
      Month, 
      DayofMonth, 
      DayOfWeek,
      DepTime,
      CRSDepTime,
      ArrTime,
      CRSArrTime, 
      CRSElapsedTime,
      ActualElapsedTime,
      AirTime, 
      ArrDelay,
      DepDelay,
      Origin,
      Dest, 
      Distance, 
      TaxiIn, 
      TaxiOut, 
      Cancelled, 
      CancellationCode,
      Diverted, 
      CarrierDelay, 
      WeatherDelay, 
      NASDelay, 
      SecurityDelay,
      LateAircraftDelay, 
      IsArrDelayed, 
      IsDepDelayed,
      current_timestamp() as created_timestamp
FROM flights_cleaned
;

