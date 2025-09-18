# Databricks notebook source
# MAGIC %md
# MAGIC # Intro to Unified Processing with Delta Lake
# MAGIC
# MAGIC One of the hallmark innovations of Databricks and the Lakehouse vision is the establishing of a unified method for writing and reading data in a data lake. This unification of batch and streaming jobs provides flexibility, simplicity, and scalability that has been pivotal towards addressing big data needs. Fundamental to building data pipelines (ETL/ELT) is the usage of a multi-hop data architecture known as the medallion architecture.
# MAGIC
# MAGIC <img src="https://github.com/brickmeister/workshop_production_delta/blob/main/img/Multi-Hop%20Delta%20Lake.png?raw=true">
# MAGIC
# MAGIC See below links for more documentation:
# MAGIC * [Delta Lake Docs](https://docs.databricks.com/delta/index.html)
# MAGIC * [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)
# MAGIC * Link to Autoloader guidance

# COMMAND ----------

# DBTITLE 1,Generate widgets for input
from pyspark.sql import functions as F 
from pyspark.sql.functions import col, lit, concat, trim, concat_ws, left, cast
from lib.utils import get_user_name, get_flight_schema

user_name_cleaned = get_user_name()
table_schema = get_flight_schema()

dbutils.widgets.dropdown("reset_all_data", "true", ["true", "false"], "Reset all data")
dbutils.widgets.text("catalog", f"{user_name_cleaned}_catalog", "Catalog")
dbutils.widgets.text("database", f"{user_name_cleaned}_workshop", "Database/Schema")
dbutils.widgets.text("file_volume", f"{user_name_cleaned}_volume", "File Path (Volume)")

# COMMAND ----------

# DBTITLE 1,Set key variables
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
catalog =dbutils.widgets.get("catalog") or f"{user_name_cleaned}_catalog"
schema = dbName = database = db = dbutils.widgets.get("database") or f"{user_name_cleaned}_workshop"
volume_name = dbutils.widgets.get("file_volume") or f"{user_name_cleaned}_volume"

# COMMAND ----------

# DBTITLE 1,Set paths (source, target, checkpoints)
static_airlines_table = f"{catalog}.{db}.airlines"

bronze_table_name = f"{catalog}.{database}.flights_bronze"
silver_table_name = f"{catalog}.{database}.flights_silver"
fact_flight_table_name = f"{catalog}.{database}.fact_flight"

bronze_checkpoint_version = "1.1"
bronze_checkpoint_location = f"/Volumes/{catalog}/{database}/{volume_name}/checkpoint_bronze{bronze_checkpoint_version}"

silver_checkpoint_version = "1.1"
silver_checkpoint_location = f"/Volumes/{catalog}/{database}/{volume_name}/checkpoint_silver{silver_checkpoint_version}"

fact_flight_checkpoint_version = "1.1"
fact_flight_checkpoint_location = f"/Volumes/{catalog}/{database}/{volume_name}/checkpoint_fact_flight{fact_flight_checkpoint_version}"

# COMMAND ----------

# DBTITLE 1,Refresh (Clear existing tables and checkpoints)
if reset_all_data:
    spark.sql(f"DROP TABLE IF EXISTS {bronze_table_name}")

    dbutils.fs.rm(bronze_checkpoint_location, True)
    dbutils.fs.rm(silver_checkpoint_location, True)
    dbutils.fs.rm(fact_flight_checkpoint_location, True)
    print("Refresh All completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process from raw/landing table into bronze
# MAGIC This step starts with a Delta table for simplicity. It represents a scenario where a connector is loading your data incrementally to a Unity Catalog Table as append only records.
# MAGIC
# MAGIC Note - if you are curious about how this data was read from CSV files, you can view the code in the notebook `0_setup/setup`.
# MAGIC
# MAGIC
# MAGIC Steps:
# MAGIC - Read incrementally from Unity Catalog table
# MAGIC - Replace Arrival Delay and Departure Delay NA values with 0
# MAGIC - Cleanup times to be valid format hhmm
# MAGIC - Derive event time and event date from CRSDepTime (departure time)

# COMMAND ----------

# DBTITLE 1,Infer Schema of Airline table
table_schema = spark.read.format("delta").option("inferSchema", "true").table(static_airlines_table).limit(10).schema

# COMMAND ----------

# DBTITLE 1,Read and Transform for Bronze
"""
Read source data incrementally
 Notes:
  1. Modify maxBytesPerTrigger to work with larger batches (faster full load)
     or smaller batches (slower full load)
"""
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit

target_table = f"{catalog}.{db}.flights_bronze"
QUERY_NAME = f"airlines_ingest_bronze"

flights_input_df = (
    spark.readStream.format("delta")
    .schema(table_schema)
    .option("maxBytesPerTrigger", "1024mb")
    .table(static_airlines_table)
)

# COMMAND ----------

flights_bronze_df = (
  flights_input_df
      .withColumn("created_timestamp", F.current_timestamp())
  )

query = (
  flights_bronze_df.writeStream
    .queryName(QUERY_NAME)
    .partitionBy("year")
    .option("checkpointLocation", bronze_checkpoint_location)
    .option("mergeSchema", "true")
    .trigger(availableNow=True) # (processingTime='30 seconds')
    .toTable(bronze_table_name)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process from bronze into silver
# MAGIC This step reads bronze data incrementaly, does additional transformations, and write into silver table.
# MAGIC
# MAGIC Steps:
# MAGIC - Read incrementally from Unity Catalog table
# MAGIC - Creating a FlightNum Column by combining the Unique Carrier & Fligh Num Columns
# MAGIC - Constructs date and timestamp fields from year/month/day and scheduled departure time
# MAGIC - Adds event_date and created_timestamp columns for downstream processing

# COMMAND ----------

"""
Minimal transformations for bronze layer:
- Cleans and normalizes time columns (CRSArrTime, DepTime)
- Constructs date and timestamp fields from year/month/day and scheduled departure time
- Adds event_timestamp and created_timestamp columns for downstream processing
"""

def pad_time_column(col_name):
    return F.lpad(F.replace(col_name, lit("NA"), lit("")), 4, "0")

def build_date_str():
    return F.concat(
        F.col("Year").cast("string"),
        F.lit("-"),
        F.lpad(F.col("Month").cast("string"), 2, "0"),
        F.lit("-"),
        F.lpad(F.col("DayOfMonth").cast("string"), 2, "0")
    )

def build_timestamp_str():
    return F.concat(
        F.expr("substring(CRSDepTime, 1, 2)"),
        F.lit(":"),
        F.expr("substring(CRSDepTime, -2, 2)")
    )

flights_clean_df = (
    flights_bronze_df
    .withColumn("FlightKey", F.concat_ws("-", col("UniqueCarrier"), col("FlightNum"), F.substring(col("event_date"), 1, 10)))
    .withColumn("CRSArrTime", pad_time_column("CRSArrTime"))
    .withColumn("ArrTime", pad_time_column("ArrTime"))
    .withColumn("DepTime", pad_time_column("DepTime"))
    .withColumn("CRSDepTime", pad_time_column("CRSDepTime"))
    .withColumn("timestamp_str", build_timestamp_str())
    .withColumn("event_timestamp", F.expr("try_to_timestamp(concat(event_date, ' ', timestamp_str))"))
    .withColumn("created_timestamp", F.current_timestamp())
    .drop('timestamp_str')
)

# COMMAND ----------

# DBTITLE 1,Streaming Write to Silver
QUERY_NAME = "airlines_silver"

# # Write the cleaned data to the target table in append mode
query = (
  flights_clean_df.writeStream
    .queryName(QUERY_NAME)
    .option("checkpointLocation", silver_checkpoint_location)
    .option("mergeSchema", "true")
    .trigger(availableNow=True) # (processingTime='30 seconds')
    .toTable(silver_table_name)
  )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Process from silver into gold
# MAGIC This step reads silver data incrementally, does additional transformations, and write into gold table fact_flight.
# MAGIC - Creating 2 structs to group time and delay columns

# COMMAND ----------

# DBTITLE 1,Read and Transform for Gold
"""
Read silver data incrementally
 Notes:
  1. Modify maxBytesPerTrigger to work with larger batches (faster full load)
     or smaller batches (slower full load)
"""
from lib.delay_calculations import add_dim_delay_id

QUERY_NAME = f"airlines_fact"

flights_silver_df = (
  spark.readStream.format("delta")
    .schema(table_schema)
    .option("maxBytesPerTrigger", "1024mb")
    .table(silver_table_name)
)

# # add dim_delay_id
try:
    delay_columns = ["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
    flights_silver_df = add_dim_delay_id(flights_silver_df, *delay_columns)
except Exception as e:
  print("Error in delay calculation: " + str(e))
  
flights_silver_df.createOrReplaceTempView("fact_flight_tmp")

fact_transform_query = f"""
    SELECT 
      FlightKey as flight_key,
      UniqueCarrier as carrier_code,
      Origin as origin_location_code,
      Dest as destination_location_code,
      dd.Date_id as dim_date_id,
      event_timestamp,
      FlightNum as flight_num,
      STRUCT (
        CRSArrTime, 
        ArrTime, 
        DepTime,
        CRSDepTime,
        ActualElapsedTime, 
        CRSElapsedTime,
        AirTime
      ) as times,
      dim_delay_id,
      STRUCT (
        ArrDelay,
        DepDelay,
        CarrierDelay,
        WeatherDelay, 
        NASDelay,
        SecurityDelay,
        LateAircraftDelay, 
        IsArrDelayed,
        IsDepDelayed
      ) as delay,
      TaxiIn as taxi_in, 
      TaxiOut as taxi_out, 
      Diverted as diverted,
      Cancelled as cancelled, 
      CancellationCode as cancelled_code,
      Distance as distance,
      current_timestamp() as created_timestamp
    FROM fact_flight_tmp
      INNER JOIN {catalog}.{db}.dim_date dd
        ON fact_flight_tmp.event_date = dd.Date
""" 

fact_clean_df = spark.sql(fact_transform_query)

# COMMAND ----------

# DBTITLE 1,Create fact table (if needed)
def get_fact_table_properties():
  return """
    'delta.autoOptimize.optimizeWrite' = true,
    'schema.autoMerge.enabled' = true,
    'delta.checkpoint.writeStatsAsStruct' = true,
    'delta.columnMapping.mode' = 'name'
  """
  
def create_flights_fact_table(replace=False):
    if replace == False:
      create_str = "CREATE TABLE IF NOT EXISTS"
    else:
      create_str = "CREATE OR REPLACE TABLE"
    
    create_statement = f"""{create_str} {fact_flight_table_name} (
      carrier_code string,
      origin_location_code string,
      destination_location_code string,
      event_date date,
      event_timestamp timestamp
    )
    TBLPROPERTIES ({get_fact_table_properties()})"""

    spark.sql(create_statement)

create_flights_fact_table(reset_all_data)

# COMMAND ----------

# DBTITLE 1,Streaming Write to fact_flight
"""
Write the cleaned data to the target table in append mode
"""
query = (
  fact_clean_df.writeStream
    .queryName(QUERY_NAME)
    .option("checkpointLocation", fact_flight_checkpoint_location)
    .option("mergeSchema", "true")
    .trigger(availableNow=True) 
    .toTable(fact_flight_table_name)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup
# MAGIC You made it through the notebook! 
# MAGIC - Now is a good time to stop the notebook or terminate your cluster to avoid additional charges.
# MAGIC - You may also wish to cleanup the tables created (or you can do that after trying part 2 and part 3 of this workshop).
