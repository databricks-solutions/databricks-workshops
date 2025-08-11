with date_series as (
    -- Generate a series of dates from 1980-01-01 to 2030-12-31
    select
        sequence(to_date('1980-01-01'), to_date('2030-12-31'), interval 1 day) as date_array
),
dates as (
    select explode(date_array) as date
    from date_series
),
date_dim as (
    select
        cast(date_format(date, 'yyyyMMdd') as int) as date_id,
        year(date) as year,
        month(date) as month,
        day(date) as dayofmonth,
        dayofweek(date) as dayofweek,
        date_format(date, 'MMMM') as monthname,
        date_format(date, 'EEEE') as dayname,
        concat('Q', quarter(date)) as quarter,
        date as date
    from dates
),
placeholder as (
    select
        99990101 as date_id,
        9999 as year,
        1 as month,
        1 as dayofmonth,
        1 as dayofweek,
        'Unknown' as monthname,
        'Unknown' as dayname,
        'Q1' as quarter,
        cast(null as date) as date
),
final as (
    select * from date_dim
    union all
    select * from placeholder
)
select *
from final
