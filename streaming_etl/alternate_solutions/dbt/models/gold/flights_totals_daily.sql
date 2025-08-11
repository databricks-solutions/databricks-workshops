SELECT dim_date_id, COUNT(DISTINCT carrier_code) carriers, COUNT(DISTINCT flight_num) flights
FROM {{ ref('fact_flight')}}
GROUP BY ALL