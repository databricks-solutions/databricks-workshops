# Databricks notebook source
# MAGIC %md
# MAGIC ## Bootstrap dimension tables
# MAGIC **Run this notebook before continuing on to notebook `1_flights_structured_streaming`.** Reviewing the code here is optional.
# MAGIC
# MAGIC This notebook demonstrates batch load to create required dimension tables. Some are a full overwrite and others use the MERGE statement to do an upsert (add new records, update existing records).

# COMMAND ----------

# DBTITLE 1,Generate widgets for input
from lib.utils import get_user_name

user_name_cleaned = get_user_name()

dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
dbutils.widgets.text("catalog", f"{user_name_cleaned}_catalog", "Catalog")
dbutils.widgets.text("database", f"{user_name_cleaned}_workshop", "Database/Schema")
dbutils.widgets.text("file_volume", f"{user_name_cleaned}_volume", "File Path (Volume)")

# COMMAND ----------

# DBTITLE 1,Set key variables
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
catalog =dbutils.widgets.get("catalog") or f"{user_name_cleaned}_catalog"
schema = dbName = database = db = dbutils.widgets.get("database") or f"{user_name_cleaned}_workshop"
volume_name = dbutils.widgets.get("file_volume") or f"{user_name_cleaned}_volume"
static_airlines_table = f"{catalog}.{db}.airlines"

# COMMAND ----------

# DBTITLE 1,If reset_all_date = True, clear existing tables
if reset_all_data:
    spark.sql(f"DROP TABLE IF EXISTS {catalog}.{database}.dim_carrier")
    spark.sql(f"DROP TABLE IF EXISTS {catalog}.{database}.dim_location")
    spark.sql(f"DROP TABLE IF EXISTS {catalog}.{database}.dim_date")
    spark.sql(f"DROP TABLE IF EXISTS {catalog}.{database}.dim_delay")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Carrier Dimension
# MAGIC Steps:
# MAGIC - Get carrier info from spreadsheet
# MAGIC - Calculate code and date fielsds
# MAGIC - Merge any new or updated values into existing dimension table

# COMMAND ----------

# DBTITLE 1,dim_carrier
from pyspark.sql.functions import col, lit, expr, coalesce, row_number, when
from pyspark.sql.window import Window
import pandas as pd
import os

current_dir = os.getcwd()
csv_full_path = f"{current_dir}/data/T_CARRIER_DECODE.csv"

reference_data = pd.read_csv(csv_full_path, header=0)

carrier_reference_df = spark.createDataFrame(reference_data).withColumnRenamed("CARRIER", "carrier_code").withColumn("unique_carrier_code", coalesce("UNIQUE_CARRIER", "carrier_code")).drop("UNIQUE_CARRIER").filter("unique_carrier_code is not null")

window_spec = Window.partitionBy("unique_carrier_code").orderBy(coalesce("end_date_source", lit("2099-12-31")).desc())

carrier_full_df = (
  carrier_reference_df
      .withColumnRenamed("THRU_DATE_SOURCE", "end_date_source")
      .withColumn("row_number", row_number().over(window_spec))
      .withColumn("is_current", when(col("row_number") == 1, True).otherwise(False))
      .drop("row_number")
)

create_table_query = f"""CREATE OR REPLACE TABLE 
  {catalog}.{db}.dim_carrier (
    unique_carrier_code STRING,
    carrier_code STRING,
    airline_id LONG,
    carrier_entity STRING,
    carrier_name STRING,
    unique_carrier_entity STRING,
    unique_carrier_name STRING,
    wac LONG,
    carrier_group LONG,
    carrier_group_new LONG,
    region STRING,
    start_date_source STRING,
    end_date_source STRING,
    is_current BOOLEAN
  )"""
spark.sql(create_table_query)

carrier_full_df.write.mode("overwrite").saveAsTable(f"{catalog}.{db}.dim_carrier")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Location Dimension
# MAGIC Steps:
# MAGIC - Get unique set of origin and destination codes
# MAGIC - Merge any new or updated values into existing dimension table
# MAGIC
# MAGIC ### Customization challenge
# MAGIC If you are looking for additional practice with Databricks ETL, you could try one or more of the following ideas:
# MAGIC - Calculate more descriptive name
# MAGIC - Add external data about each location

# COMMAND ----------

# DBTITLE 1,dim_location
from pyspark.sql.functions import col

location_df = (
  spark.read
    .table(static_airlines_table)
    .select(col("Origin").alias("location_code"))
    .union(
      spark.read
        .table(static_airlines_table)
        .select(col("Dest").alias("location_code"))
    )
    .distinct()
    .where((col("location_code").isNotNull()) & (col("location_code") != lit("Dest")) & (col("location_code") != lit("Origin")))
    .selectExpr(
      "location_code",
      "location_code as location_name"
    )
    .orderBy("location_code")
)

location_df.createOrReplaceTempView("tmp_location")

create_table_query = f"""CREATE TABLE IF NOT EXISTS 
  {catalog}.{db}.dim_location (
    location_code STRING,
    location_name STRING
  )"""
spark.sql(create_table_query)

merge_query = f"""
MERGE INTO {catalog}.{db}.dim_location AS target
USING tmp_location AS source
ON target.location_code = source.location_code
WHEN MATCHED THEN
  UPDATE SET target.location_name = source.location_name
WHEN NOT MATCHED THEN
  INSERT (location_code, location_name)
  VALUES (source.location_code, source.location_name)
"""

spark.sql(merge_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Date Dimension
# MAGIC Steps:
# MAGIC - Calculate every date in range from start_year through end_year
# MAGIC - Calculate various description fields to simplify reporting
# MAGIC - Add placeholder (dummy) date

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, date_format, concat_ws, year, month, dayofmonth, dayofweek, quarter
from pyspark.sql.types import IntegerType, StringType, DateType, StructType, StructField
import datetime

def create_date_dim_table(start_year, end_year):
    # Create a DataFrame with a sequence of dates from start_year to end_year
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)

    date_list = [(start_date + datetime.timedelta(days=x)) for x in range(0, (end_date - start_date).days + 1)]
    date_df = spark.createDataFrame([(d,) for d in date_list], ["Date"])

    # Add required columns
    date_dim_df = date_df \
        .withColumn("Date_id", date_format(col("Date"), "yyyyMMdd").cast(IntegerType())) \
        .withColumn("Year", year(col("Date"))) \
        .withColumn("Month", month(col("Date"))) \
        .withColumn("DayOfMonth", dayofmonth(col("Date"))) \
        .withColumn("DayOfWeek", dayofweek(col("Date"))) \
        .withColumn("MonthName", date_format(col("Date"), "MMMM").cast(StringType())) \
        .withColumn("DayName", date_format(col("Date"), "EEEE").cast(StringType())) \
        .withColumn("Quarter", concat_ws("", lit("Q"), quarter(col("Date"))).cast(StringType())) \
        .withColumn("Date", col("Date").cast(DateType()))

    # Define schema for placeholder row
    schema = StructType([
        StructField("Date_id", IntegerType(), True),
        StructField("Year", IntegerType(), True),
        StructField("Month", IntegerType(), True),
        StructField("DayOfMonth", IntegerType(), True),
        StructField("DayOfWeek", IntegerType(), True),
        StructField("MonthName", StringType(), True),
        StructField("DayName", StringType(), True),
        StructField("Quarter", StringType(), True),
        StructField("Date", DateType(), True)
    ])

    # Add a placeholder row for Null/Unknown date (99990101)
    placeholder_row = [(99990101, 9999, 1, 1, 1, "Unknown", "Unknown", "Q1", None)]
    placeholder_df = spark.createDataFrame(placeholder_row, schema=schema)

    date_dim_df = date_dim_df.unionByName(placeholder_df)

    # Write the DataFrame to a Delta table
    date_dim_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{database}.dim_date")

    print("Date dimension table created successfully.")

# Example usage
create_date_dim_table(1980, 2030)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Delay Dimension
# MAGIC Steps:
# MAGIC - Build out bins/ranges for delay times
# MAGIC - Use bitwise calculation to generate meaningful ID representing delay combination
# MAGIC - Overwrite table `dim_delay`
# MAGIC
# MAGIC Note - the logic for calculating the key is a clever way to combine the possible options. This bitwise calculation is not a very typical ETL pattern but it is useful in this data model.

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.functions import udf, shiftleft
from pyspark.sql.types import LongType
from pyspark.sql.functions import col, expr

#from lib.delay_calculations import delay_columns

delay_columns = ["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]

def generate_combinations():
    from itertools import product
    
    arr_dep_delay_values = ["Early", "No Delay", "1-30m Delay", "31-60m Delay", ">60m Delay"]
    other_delay_values = ["No Delay", "1-30m Delay", "31-60m Delay", ">60m Delay"]
    
    combinations = list(product(arr_dep_delay_values, arr_dep_delay_values, other_delay_values, other_delay_values, other_delay_values, other_delay_values, other_delay_values))
    
    df = spark.createDataFrame(combinations, delay_columns)
    
    return df

combinations_df = generate_combinations()


# Define a mapping dictionary
delay_mapping = {
    "Early": 1,
    "No Delay": 2,
    "1-30m Delay": 4,
    "31-60m Delay": 8,
    ">60m Delay": 16
}

# Create a UDF to map delay values to integers
@udf(LongType())
def map_delay_to_int(delay):
    return delay_mapping[delay]

# Apply the mapping to each column
for column in delay_columns:
    combinations_df = combinations_df.withColumn(f"{column}_Int", map_delay_to_int(col(column)))


# Calculate the bitwise OR for each row
def bitwise_or_udf(*cols):
    result = cols[0]
    for col in cols[1:]:
        result = result.bitwiseOR(col)
    
    return result

combinations_df = combinations_df.withColumn(
    "delay_id",
    bitwise_or_udf(
        shiftleft(col("ArrDelay_Int"), 0),
        shiftleft(col("DepDelay_Int"), 5),
        shiftleft(col("CarrierDelay_Int"), 10),
        shiftleft(col("WeatherDelay_Int"), 15),
        shiftleft(col("NASDelay_Int"), 20),
        shiftleft(col("SecurityDelay_Int"), 25),
        shiftleft(col("LateAircraftDelay_Int"), 30)
    )
).drop("ArrDelay_Int", "DepDelay_Int", "CarrierDelay_Int", "WeatherDelay_Int", "NASDelay_Int", "SecurityDelay_Int", "LateAircraftDelay_Int")

combinations_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{database}.dim_delay")
