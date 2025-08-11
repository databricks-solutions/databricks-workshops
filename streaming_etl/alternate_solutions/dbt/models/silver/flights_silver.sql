{{
    config(
        liquid_clustered_by="FlightKey"
    )
}}

WITH flights_cleaned as (
 SELECT
  * EXCEPT (DepTime, CRSDepTime, ArrTime, CRSArrTime),
  lpad(replace(DepTime, 'NA', ''), 4, '0') DepTime,
  lpad(replace(CRSDepTime, 'NA', ''), 4, '0') CRSDepTime,
  lpad(replace(ArrTime, 'NA', ''), 4, '0') ArrTime,
  lpad(replace(CRSArrTime, 'NA', ''), 4, '0') CRSArrTime
FROM {{ ref('flights_bronze') }}
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
