# Databricks notebook source
# MAGIC %md
# MAGIC The Data Intelligence Platform provides several features to ensure customers are developing production level pipelines for optimal performance. Today we will be discussing the following features:
# MAGIC
# MAGIC - Predictive Optimization
# MAGIC - Liquid Clustering
# MAGIC - Partitioning
# MAGIC - Analyze
# MAGIC - Deletion Vectors
# MAGIC - Vaccum
# MAGIC - Data Quality Checks w/ DQX
# MAGIC - Iceberg Compatibility

# COMMAND ----------

from databricks.sdk import WorkspaceClient
wc = WorkspaceClient()
user_name = wc.current_user.me().display_name

user_name_cleaned = ''.join(['_' if not c.isalpha() else c for c in user_name.lower()])
display(user_name_cleaned)

# COMMAND ----------

from pyspark.sql import functions as F 
from pyspark.sql.functions import col, lit, concat, trim, concat_ws, left, cast

dbutils.widgets.text("catalog", f"{user_name_cleaned}_catalog", "Catalog")
dbutils.widgets.text("database", f"{user_name_cleaned}_workshop", "Database/Schema")

# COMMAND ----------

catalog =dbutils.widgets.get("catalog") or f"{user_name_cleaned}_catalog"
schema = dbName = database = db = dbutils.widgets.get("database") or f"{user_name_cleaned}_workshop"
silver_table_name = f"{catalog}.{database}.flights_silver"

# COMMAND ----------

print(catalog)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predictive Optimization
# MAGIC
# MAGIC Predictive optimization eliminates the need to manually manage maintenance operations for Unity Catalog managed tables. 
# MAGIC
# MAGIC With this feature enabled, Databricks will automatically do the following:
# MAGIC
# MAGIC - Identify tables that would benefit from maintenance operations and to run the necessary operations
# MAGIC - Collects statistics when data is written to the managed table
# MAGIC
# MAGIC When enabled, predictive optimization runs the following operations
# MAGIC - OPTIMIZE - Triggers incremental clustering for selected tables
# MAGIC - VACCUM - Deletes data files that are no longer referenced by the table
# MAGIC - ANALYZE - Triggers incremental updates to statistics of a table
# MAGIC
# MAGIC ![](/Workspace/Users/benjamin.nwokeleme@databricks.com/streaming_in_production/workshop/3_table_config/predictive_optimization.png)
# MAGIC
# MAGIC _Starting May 7, 2025, Databricks will enable predictive optimization by default for all existing Databricks accounts._

# COMMAND ----------

#spark.sql(f"ALTER SCHEMA {schema} DISABLE PREDICTIVE OPTIMIZATION;")

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog};")
spark.sql(f"ALTER SCHEMA {schema} ENABLE PREDICTIVE OPTIMIZATION;")

# COMMAND ----------

# disable on a table
spark.sql(f"ALTER TABLE {silver_table_name} DISABLE PREDICTIVE OPTIMIZATION;")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Table Performance
# MAGIC
# MAGIC One of the best ways to get fast queries is to organize related data in the same files. This helps the engine read or update a subset of the files. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Liquid Clustering
# MAGIC
# MAGIC Liquid clustering allows your data layout to evolve with ever-changing query patterns. It replaces table partitioning and ZORDER to simplify data layout decisions and optimize query performance. When enabled, you can redefine your clustering columns without having to rewrite any existing data. 
# MAGIC
# MAGIC _What is the problem with hive style partitioning?_
# MAGIC
# MAGIC Partitioning and Z-ordering are not flexible. You have to decide your partitioning columns ahead of time and if your query patterns change (one day your colleague needs to run a query on the customer column), you will have to rewrite the entire dataset to optimize it for that new query.
# MAGIC
# MAGIC _Automatic Liquid Clustering_
# MAGIC
# MAGIC Automatic liquid clustering is a newer feature that when enabled, Databricks intelligently choose the clustering keys to optimize query performance. You enable automatic liquid clustering using the CLUSTER BY AUTO clause. You also need predictive optimization

# COMMAND ----------

# Enable automatic liquid clustering on an existing table. The table must also have Predictive Optimization enabled
# spark.sql(f"ALTER TABLE {silver_table_name} CLUSTER BY AUTO")

# COMMAND ----------

# Disable automatic liquid clustering on an existing table.
spark.sql(f"ALTER TABLE {silver_table_name} CLUSTER BY NONE;")

# COMMAND ----------

# Disable automatic liquid clustering by setting the clustering keys
# to chosen clustering columns or new columns.
spark.sql(f"ALTER TABLE {silver_table_name} CLUSTER BY (Year, Month);")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Partition / Z-Order
# MAGIC
# MAGIC Partitioning is a way to organize a table into smaller subsets to improve performance of queries executed on big data by reducing the data that needs to be read. Z-Ordering is a technique to co-locate related information in the same set of files. This co-locality is automatically used by Delta data-skipping algorithms to dramatically reduce the amount of data that needs to be read.
# MAGIC

# COMMAND ----------

spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT UniqueCarrier, COUNT(1) 
# MAGIC FROM flights_bronze
# MAGIC WHERE 
# MAGIC   event_date <= '1998-06-15'
# MAGIC   AND UniqueCarrier in ('AA')
# MAGIC GROUP BY ALL;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC OPTIMIZE flights_bronze
# MAGIC ZORDER BY UniqueCarrier, event_date
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT UniqueCarrier, COUNT(1) 
# MAGIC FROM flights_bronze 
# MAGIC WHERE 
# MAGIC   event_date <= '1998-06-15'
# MAGIC   AND UniqueCarrier in ('AA')
# MAGIC GROUP BY ALL;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze (if not using PO)
# MAGIC The ANALYZE TABLE statement collects estimated statistics about a specific table or all tables in a specified schema. These statistics are used by the query optimizer to generate an optimal query plan. This is done automatically when predictive optimization is enabled. The statistics collected are stored in the Delta log.

# COMMAND ----------

spark.sql(f"ANALYZE TABLE {silver_table_name} COMPUTE STATISTICS;")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deletion Vectors
# MAGIC
# MAGIC This is a storage optimization feature you can enable on Delta Lake tables. It also improves the performance of DELETE, UPDATE, and MERGE operations on Delta tables.
# MAGIC
# MAGIC Ordinarily, when a single row in a data file is deleted, the entire Parquet file containing the record must be rewritten. With deletion vectors enabled for the table, some Delta operations use deletion vectors to mark these deleted rows as removed without rewriting the Parquet file. **They act as a soft delete mechanism.** When the table is read, the deletions indicated by the deletion vectors are applied the most recent version of the table to resolve its current state.
# MAGIC
# MAGIC These deletes are then applied **physically** when data files are rewritten, as triggered by one of the following events:
# MAGIC - An OPTIMIZE command is run on the table.
# MAGIC - Auto-compaction triggers a rewrite of a data file with a deletion vector.
# MAGIC - REORG TABLE ... APPLY (PURGE) is run against the table.

# COMMAND ----------

spark.sql(f"ALTER TABLE {silver_table_name} SET TBLPROPERTIES ('delta.enableDeletionVectors' = true);")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Vacuum
# MAGIC In order to remove physical traces of deleted data, we run a vacuum command with a retention time

# COMMAND ----------

# Retains the last 14 days of version history

spark.sql(f"VACUUM {silver_table_name} RETAIN 336 HOURS;")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality concepts (databrickslabs/dqx)
# MAGIC
# MAGIC DQX is a Databricks lab project that introduces a Python validation framework for assessing the data quality of PySpark DataFrames. It enables real-time quality validation during data processing rather than relying solely on post-factum monitoring.
# MAGIC
# MAGIC Other frameworks do not provide detailed explanations for specific row or column data quality issues and are primarily designed for complete datasets, making integration into streaming workloads difficult. 
# MAGIC
# MAGIC DQX seamlessly integrates with other Databricks tools and frameworks such as:
# MAGIC
# MAGIC - Lakeflow Pipelines (formerly DLT), where you can use DQX to apply DQX quality checks or to profile input data and generate Lakeflow expectation candidates.
# MAGIC - dbt, where you can apply DQX quality checks on the output of dbt models.
# MAGIC
# MAGIC _For more information, please visit https://databrickslabs.github.io/dqx/_

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table Summary Statistics Generation

# COMMAND ----------

pip install databricks-labs-dqx

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Data loaded as a DataFrame can be profiled to generate summary statistics and candidate data quality rules.

# COMMAND ----------

from databricks.labs.dqx.profiler.profiler import DQProfiler
from databricks.sdk import WorkspaceClient

ws = WorkspaceClient()
profiler = DQProfiler(ws)

summary_stats, profiles = profiler.profile_table(
    table=f"{silver_table_name}",
    columns=["Year"],  # specify columns to profile
)

print("Summary Statistics:", summary_stats)
print("Generated Profiles:", profiles)

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Optional) UC integration + Iceberg Compatibility
# MAGIC
# MAGIC You can enable Iceberg reads on tables stored with Delta Lake in Databricks. Enabling Iceberg reads configures your tables to automatically generate Iceberg metadata, without rewriting data, so that Iceberg clients can read Delta tables written by Databricks. A single copy of the data files serves multiple formats.

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE silver_table_name SET TBLPROPERTIES(
# MAGIC   'delta.columnMapping.mode' = 'name',
# MAGIC   'delta.enableIcebergCompatV2' = 'true',
# MAGIC   'delta.universalFormat.enabledFormats' = 'iceberg');
