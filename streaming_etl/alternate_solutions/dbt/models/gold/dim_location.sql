{{
    config(
        materialized='incremental',
        unique_key='location_code'
    )
}}
with locations as (
    select Origin as location_code
    from {{ ref('flights_silver') }}
    union
    select Dest as location_code
    from {{ ref('flights_silver') }}
),
filtered_locations as (
    select distinct
        location_code
    from locations
    where
        location_code is not null
        and location_code not in ('Dest', 'Origin')
)
select
    location_code,
    location_code as location_name
from filtered_locations