SELECT carrier_code, flight_num, 1 as n
FROM {{ ref('fact_flight') }}
GROUP BY ALL
