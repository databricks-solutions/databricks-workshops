{{
    config(
        materialized='table',
        tblproperties={
            'delta.autoOptimize.optimizeWrite': 'true',
            'delta.autoOptimize.autoCompact': 'true'
        }
    )
}}

SELECT 
    carrier_code,
    YEAR(event_date) as flight_year,
    COUNT(*) as flight_count,
    COUNT(DISTINCT origin_location_code) as unique_origins,
    COUNT(DISTINCT destination_location_code) as unique_destinations,
    AVG(distance) as avg_distance,
    SUM(CASE WHEN cancelled = true THEN 1 ELSE 0 END) as cancelled_flights,
    SUM(CASE WHEN diverted = true THEN 1 ELSE 0 END) as diverted_flights,
    ROUND(
        (SUM(CASE WHEN cancelled = true THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2
    ) as cancellation_rate_pct
FROM {{ ref('fact_flight') }}
WHERE event_date IS NOT NULL
GROUP BY carrier_code, YEAR(event_date)
ORDER BY flight_year DESC, flight_count DESC
