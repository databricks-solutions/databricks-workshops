# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
dbutils.widgets.dropdown("size", "small", ["x-small", "small", "medium", "large"], "Size of data volume to load")
data_size = dbutils.widgets.get("size")
dbutils.widgets.dropdown("load_type", "files", ["files", "generate"], "Load from sample dataset files or generate additional data.")
load_type = dbutils.widgets.get("files")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
wc = WorkspaceClient()
user_name = wc.current_user.me().display_name

user_name_cleaned = ''.join(['_' if not c.isalpha() else c for c in user_name.lower()])
display(user_name_cleaned)

# COMMAND ----------

catalog = f"{user_name_cleaned}_catalog"
schema = dbName = database = db = f"{user_name_cleaned}_workshop"
volume_name = f"{user_name_cleaned}_volume"
static_airlines_table = f"{catalog}.{db}.airlines"

# COMMAND ----------

from setup_helper import SetupHelper
SetupHelper.setup_schema(spark, catalog, db, reset_all_data, volume_name)
checkpoint_folder = volume_name + "/_checkpoint"

# COMMAND ----------

def get_catalog_db_checkpoint():
    return catalog, database, checkpoint_folder

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

def delay_history_randomization(df):
    # Delay related columns
    # delay_columns = ["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]

    # Fix ArrDelay and DepDelay - replace NA with 0 and cast to Long
    df = df.withColumn("ArrDelay", F.when(col("ArrDelay") == "NA", 0).otherwise(col("ArrDelay")).cast("int"))
    df = df.withColumn("DepDelay", F.when(col("DepDelay") == "NA", 0).otherwise(col("DepDelay")).cast("int"))

    # To simulate useful data for history...
    # These columns in the datasets are all NA. Replace them with some meaningful delay values
    all_na_cols = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
    for all_na_col in all_na_cols:
        # Condition checks if some delay, col value is NA, and randomness of 20%
        test_condition = (
            ((col("ArrDelay") > 0) | (col("DepDelay") > 0))
          & (col(all_na_col) == "NA")
          & (F.rand() <= 0.2)
        )
        
        # Use condition to simulate historical data
        df = df.withColumn(
               all_na_col, 
               F.when(
                  test_condition, 
                  (F.round(F.rand() * (col("ArrDelay") + col("DepDelay")), 0) + 1))
                .otherwise(0).cast("int")
          )
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean the checkpoint folder if reset_all_data is True

# COMMAND ----------

if reset_all_data:
    dbutils.fs.rm(checkpoint_folder, recurse=True)
    print(f"Removed checkpoint folder {checkpoint_folder}")
    #display(dbutils.fs.ls(volume_folder))
    # dbutils.fs.rm(volume_folder, recurse=True)
    # print(f"Removed volume folder {volume_folder}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the sample airlines data into a Delta table

# COMMAND ----------

from pyspark.sql import functions as F

def get_filter_clause(data_size = "small"):
    if data_size == "x-small":
        filter_clause = F.col("UniqueCarrier") == "AA"
    elif data_size == "small":
        filter_clause = F.col("UniqueCarrier").isin("AA", "DL")
    elif data_size == "medium":
        filter_clause = F.col("UniqueCarrier").isin("AA", "DL", "UA", "WN")
    else:  # large
        filter_clause = F.lit(True)
    return filter_clause

def load_sample_data_into_delta_table(target_table: str, data_size: str = "small"):
    inferred_schema = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("/databricks-datasets/airlines/part-00000").limit(100).schema

    # Modify maxBytesPerTrigger to work with larger batchers (faster full load) or smaller batches (slower full load)
    streaming_df = spark.readStream.format("cloudFiles") \
        .option("cloudFiles.format", "csv") \
        .option("cloudFiles.includeExistingFiles", "true") \
        .schema(inferred_schema) \
        .option("cloudFiles.maxBytesPerTrigger", "2048mb") \
        .load("/databricks-datasets/airlines") \
        .where(get_filter_clause(data_size)) \
        .withColumn("event_date", 
            F.concat(
            F.col("Year").cast("string"),
            F.lit("-"),
            F.lpad(F.col("Month").cast("string"), 2, "0"),
            F.lit("-"),
            F.lpad(F.col("DayOfMonth").cast("string"), 2, "0")
            ).cast("date")
        )

    streaming_cleaned_df = streaming_df.transform(delay_history_randomization)

    (
    streaming_cleaned_df
        .writeStream
        .format("delta")
        .trigger(availableNow=True)
        .option("queryName", f"initialize_delta_table_{target_table}")
        .option("checkpointLocation", f"{checkpoint_folder}/{target_table}")
        .toTable(target_table)
    )

# COMMAND ----------

if load_type == "files":
    load_sample_data_into_delta_table(static_airlines_table, data_size)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start writing new data every N minutes to support streaming

# COMMAND ----------

from pyspark.sql.functions import (
    explode, sequence, to_date, year, month, dayofmonth, dayofweek, expr
)
# from pyspark.sql import SparkSession
from datetime import datetime
import time
import random

def generate_data(static_airlines_table: str, generate_year = None):
    """
    Function to generate sample flight data for the current year
    """
    # Load the static airline data
    static_airlines_df = spark.read.table(static_airlines_table)

    # Determine current year or use passed in value
    if generate_year is None:
        current_year = datetime.now().year
    else:
        current_year = generate_year

    # Construct the start and end of the current year
    start_date = f"{current_year}-01-01"
    days_in_year = 365  # Assuming 365 days for simplicity (not handling leap years here)

    # Create a dynamic event_date column
    final_df = static_airlines_df \
        .limit(100000) \
        .withColumn("event_date", expr(f"date_add(to_date('{start_date}'), cast(rand()*{days_in_year} as int))")) \
        .withColumn("Year", year("event_date")) \
        .withColumn("Month", month("event_date")) \
        .withColumn("DayOfMonth", dayofmonth("event_date")) \
        .withColumn("DayOfWeek", dayofweek("event_date"))

    return final_df  # .transform(delay_history_randomization)

def write_in_intervals(final_data, target_table: str, time_out_seconds: int, batch_interval_seconds: int = 10, data_size: str = "small"):
    """
    Function to write data in intervals

    Parameters:
    - final_data: DataFrame containing the final data to be written
    - target_table: Name of the target table to write the data to
    - time_out_seconds: Maximum time in seconds to run the process
    - batch_interval_seconds: Interval in seconds between each batch write
    - data_size: Size of the data to generate (x-small, small, medium, large)
    """   
    start_time = time.time()

    while True:
        # Check elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_out_seconds:
            print("Time out reached, stopping the process.")
            break

        final_data \
        .sample(withReplacement=True, fraction=0.1, seed=random.randint(0, 100)) \
        .where(get_filter_clause(data_size)) \
        .write.mode("append").saveAsTable(target_table)

# COMMAND ----------

if load_type == "generate":
    # Input parameters for data generation and writing
    try:
    year_after_max_loaded_year = spark.sql(f"SELECT MAX(Year) FROM {static_airlines_table}").collect()[0][0] + 1

    generate_years = [year_after_max_loaded_year]
    for selected_year in generate_years:
        final_data = generate_data(static_airlines_table)
        time_out_seconds = 60  # Run for 5 minutes
        batch_interval_seconds = 30  # Write batch every 30 seconds
        write_in_intervals(final_data, static_airlines_table, time_out_seconds, batch_interval_seconds, data_size)
    except TypeError:
    print("Data generation skipped because no Year value found, check if table has already been created and populated from the file as expected.")
