with
-- Define possible values for each delay type
arr_dep_delay_values as (
    select 'Early' as delay_value union all
    select 'No Delay' union all
    select '1-30m Delay' union all
    select '31-60m Delay' union all
    select '>60m Delay'
),
other_delay_values as (
    select 'No Delay' as delay_value union all
    select '1-30m Delay' union all
    select '31-60m Delay' union all
    select '>60m Delay'
),

-- Generate all combinations using cross join
combinations as (
    select
        arr1.delay_value as ArrDelay,
        arr2.delay_value as DepDelay,
        oth1.delay_value as CarrierDelay,
        oth2.delay_value as WeatherDelay,
        oth3.delay_value as NASDelay,
        oth4.delay_value as SecurityDelay,
        oth5.delay_value as LateAircraftDelay
    from arr_dep_delay_values arr1
    cross join arr_dep_delay_values arr2
    cross join other_delay_values oth1
    cross join other_delay_values oth2
    cross join other_delay_values oth3
    cross join other_delay_values oth4
    cross join other_delay_values oth5
),

-- Map delay values to integers
delay_map as (
    select 'Early' as delay_value, 1 as delay_int union all
    select 'No Delay', 2 union all
    select '1-30m Delay', 4 union all
    select '31-60m Delay', 8 union all
    select '>60m Delay', 16
),

-- Join to get integer values for each delay type
with_ints as (
    select
        c.*,
        dm1.delay_int as ArrDelay_Int,
        dm2.delay_int as DepDelay_Int,
        dm3.delay_int as CarrierDelay_Int,
        dm4.delay_int as WeatherDelay_Int,
        dm5.delay_int as NASDelay_Int,
        dm6.delay_int as SecurityDelay_Int,
        dm7.delay_int as LateAircraftDelay_Int
    from combinations c
    left join delay_map dm1 on c.ArrDelay = dm1.delay_value
    left join delay_map dm2 on c.DepDelay = dm2.delay_value
    left join delay_map dm3 on c.CarrierDelay = dm3.delay_value
    left join delay_map dm4 on c.WeatherDelay = dm4.delay_value
    left join delay_map dm5 on c.NASDelay = dm5.delay_value
    left join delay_map dm6 on c.SecurityDelay = dm6.delay_value
    left join delay_map dm7 on c.LateAircraftDelay = dm7.delay_value
),

-- Calculate delay_id using bitwise shifts and ORs
final as (
    select
        ArrDelay,
        DepDelay,
        CarrierDelay,
        WeatherDelay,
        NASDelay,
        SecurityDelay,
        LateAircraftDelay,
        (ArrDelay_Int << 0)
        | (DepDelay_Int << 5)
        | (CarrierDelay_Int << 10)
        | (WeatherDelay_Int << 15)
        | (NASDelay_Int << 20)
        | (SecurityDelay_Int << 25)
        | (LateAircraftDelay_Int << 30)
        as delay_id
    from with_ints
)

select *
from final
order by delay_id