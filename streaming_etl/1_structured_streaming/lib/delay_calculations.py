# TODO: Works on assigned and Shared clusters but not on Serverless
from pyspark.sql.functions import col, when, udf, shiftleft, expr
from pyspark.sql.types import LongType
from pyspark.sql import DataFrame
from functools import reduce
import operator

delay_columns = ["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]

def add_dim_delay_id(df: DataFrame, *cols: str) -> DataFrame:
    def map_delay_column(column):
        column = expr(f"try_cast({column} as long)")
        return when(column < 0, 1) \
               .when(column == 0, 2) \
               .when((column > 0) & (column <= 30), 4) \
               .when((column > 30) & (column <= 60), 8) \
               .when(column > 60, 16) \
               .otherwise(0)
    
    def bitwise_or_udf(*cols):
        result = cols[0]
        index = 1
        for col in cols[1:]:
            result = result.bitwiseOR(shiftleft(col.cast(LongType()), 5 * index))
            index += 1
    
        return result

    df = df.withColumn(
        "dim_delay_id",
        bitwise_or_udf(*[map_delay_column(column) for column in cols])
    )

    return df

# Example usage
#result_df = add_dim_delay_id(df, *delay_columns)
#display(result_df)