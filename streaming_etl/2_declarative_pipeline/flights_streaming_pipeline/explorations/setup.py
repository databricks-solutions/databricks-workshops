# Databricks notebook source
# Removing all previously created widgets (if any)
# dbutils.widgets.removeAll()

# COMMAND ----------

# Setting up a widget to control whether all data should be reset
# This dropdown allows users to choose "true" or "false" to reset all data
# Default value is "true"
dbutils.widgets.dropdown("reset_all_data", "true", ["true", "false"], "Reset all data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"

# COMMAND ----------

# Adding widgets to select an existing catalog, database, and volume, or create new ones if left blank
dbutils.widgets.text('catalog', '', 'Catalog (leave blank to create new)')
dbutils.widgets.text('database', '', 'Database (leave blank to create new)')
dbutils.widgets.text('volume_name', '', 'Volume Name (leave blank to create new)')

# Fetching values from widgets
catalog = dbutils.widgets.get('catalog')
database = dbutils.widgets.get('database')
volume_name = dbutils.widgets.get('volume_name')

# COMMAND ----------

# Importing WorkspaceClient from databricks SDK to interact with workspace
from databricks.sdk import WorkspaceClient
wc = WorkspaceClient()

# Fetching and cleaning the current user's display name
user_name = wc.current_user.me().display_name
user_name_cleaned = ''.join(['_' if not c.isalpha() else c for c in user_name.lower()])
display(user_name_cleaned)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

# Assigning default catalog, database, and volume name if not provided in widgets
if len(catalog) < 1:
    catalog = f"{user_name_cleaned}_catalog"
if len(database) < 1:
    database = f"{user_name_cleaned}_workshop"
schema = dbName = db = database

if len(volume_name) < 1:
    volume_name = f"{user_name_cleaned}_volume"

# Defining the static airlines table path
static_airlines_table = f"{catalog}.{db}.airlines"

# COMMAND ----------

# %run ./global-setup

# COMMAND ----------

# SetupHelper.setup_schema(catalog, db, reset_all_data, volume_name)
# checkpoint_folder = volume_name + "/_checkpoint"

# COMMAND ----------

from setup_helper import SetupHelper
SetupHelper.setup_schema(spark, catalog, db, reset_all_data, volume_name)
checkpoint_folder = volume_name + "/_checkpoint"

# COMMAND ----------

# Function to get catalog, database, and checkpoint folder details
def get_catalog_db_checkpoint():
    return catalog, database, checkpoint_folder

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean the checkpoint folder if reset_all_data is True

# COMMAND ----------

# If reset_all_data is set to true, remove the checkpoint folder
if reset_all_data:
    dbutils.fs.rm(checkpoint_folder, recurse=True)
    print(f"Removed checkpoint folder {checkpoint_folder}")

# COMMAND ----------

# Importing necessary libraries for creating the date dimension table
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, date_format, concat_ws, year, month, dayofmonth, dayofweek, quarter
from pyspark.sql.types import IntegerType, StringType, DateType, StructType, StructField
import datetime

def create_date_dim_table(start_year, end_year):
    """
    Function to create a Date Dimension Table from start_year to end_year
    """
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Date Dimension Table").getOrCreate()

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

    # Add a placeholder row for Null/Unknown date (99990101)
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
    placeholder_row = [(99990101, 9999, 1, 1, 1, "Unknown", "Unknown", "Q1", None)]
    placeholder_df = spark.createDataFrame(placeholder_row, schema=schema)

    # Combine with the main DataFrame
    date_dim_df = date_dim_df.unionByName(placeholder_df)

    # Write the DataFrame to a Delta table
    date_dim_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{database}.dim_date")
    print("Date dimension table created successfully.")

# Example usage
create_date_dim_table(1980, 2030)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the sample airlines data into a Delta table

# COMMAND ----------

def load_sample_data_into_delta_table(target_table: str):
    """
    Function to load sample airlines data into a Delta table
    """
    inferred_schema = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("/databricks-datasets/airlines/part-00000").limit(100).schema
    
    streaming_df = spark.readStream.format("cloudFiles") \
        .option("cloudFiles.format", "csv") \
        .option("cloudFiles.includeExistingFiles", "true") \
        .schema(inferred_schema) \
        .load("/databricks-datasets/airlines")
     
    streaming_write_query = (
        streaming_df
            .writeStream
            .format("delta")
            .trigger(availableNow=True)
            .option("queryName", f"initialize_delta_table_{target_table}")
            .option("checkpointLocation", f"{checkpoint_folder}/{target_table}")
            .toTable(target_table)
    )
    
    streaming_write_query.awaitTermination()

# Load sample data into the airlines table
load_sample_data_into_delta_table(static_airlines_table)

# COMMAND ----------

# Install Faker library for data generation
%pip install faker

# COMMAND ----------

# Importing required libraries for data generation
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, explode, sequence, to_date, year, month, dayofmonth, dayofweek, expr, concat, col, lit
from datetime import datetime
import time

def generate_data(static_airlines_table: str, generate_year = None):
    """
    Function to generate sample flight data for the current year
    """
    spark = SparkSession.builder.appName("GenerateFlightsForCurrentYear").getOrCreate()

    # Load the static airline data
    static_airlines_df = spark.read.table(static_airlines_table)

    if generate_year is None:
        # Determine current year
        current_year = datetime.now().year
    else:
        current_year = generate_year

    # Construct the start and end of the current year
    start_date = f"{current_year}-01-01"
    days_in_year = 365  # Assuming 365 days for simplicity (not handling leap years here)

    # Create a dynamic flight_date column
    final_df = static_airlines_df \
        .where(f"Year = {current_year}") \
        .withColumn("flight_date", expr(f"date_add(to_date('{start_date}'), cast(rand()*{days_in_year} as int))")) \
        .withColumn("Year", year("flight_date")) \
        .withColumn("Month", month("flight_date")) \
        .withColumn("DayOfMonth", dayofmonth("flight_date")) \
        .withColumn("DayOfWeek", dayofweek("flight_date"))

    return final_df.limit(1000)

def write_in_intervals(final_data, target_table: str, time_out_seconds: int, batch_interval_seconds: int = 10):
    """
    Function to write data to a Delta table at intervals
    """
    spark = SparkSession.builder.getOrCreate()

    # Define the set of columns that uniquely identify a flight
    unique_key_columns = [
        "UniqueCarrier", "FlightNum", "Year", "Month", "DayOfMonth", "Origin", "CRSDepTime"
    ]

    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_out_seconds:
            print("Time out reached, stopping the process.")
            break
        else:
            print("Running, elapsed time = ", str(elapsed_time))

        # Check if the Delta table exists by trying to read it
        try:
            existing_df = spark.read.table(static_airlines_table)
        except:
            existing_df = spark.createDataFrame([], final_data.schema)

        # Left-anti join to find rows that are not already in the existing table
        ## TMP commented out
        join_condition = [final_data[col_name] == existing_df[col_name] for col_name in unique_key_columns]
        new_rows = broadcast(final_data).join(existing_df, join_condition, "left_anti")

        new_rows = new_rows.withColumn("TailNum", concat(col("TailNum"), lit("A")))
        # Write new rows to the Delta table
        new_rows.limit(1000).write.format("delta").option("mergeSchema", "true").mode("append").saveAsTable(target_table)
        print(f"Wrote new rows to the Delta table.")

        # Sleep for the batch_interval_seconds
        time.sleep(batch_interval_seconds)



# COMMAND ----------

# Input parameters for data generation and writing
# generate_years = [2018, 2019]
generate_years = [spark.sql(f"SELECT MAX(Year) FROM {static_airlines_table}").collect()[0][0]]
for selected_year in generate_years:
    final_data = generate_data(static_airlines_table, selected_year)
    time_out_seconds = 180  # Run for 3 minutes
    batch_interval_seconds = 30  # Write batch every 30 seconds
    write_in_intervals(final_data, static_airlines_table, time_out_seconds, batch_interval_seconds)
