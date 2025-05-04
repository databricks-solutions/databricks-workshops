# Databricks notebook source
# MAGIC %md
# MAGIC # 06 AI Gateway Lab Cleanup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Workspace Tidy Up
# MAGIC As we have reached the end of the Lab, it's time to tidy up our workspace.    
# MAGIC
# MAGIC If you will not be using the endpoint(s) you have created, this notebook will help you clean up the endpoints and corresponding payload tables you have created to work through the examples.  

# COMMAND ----------

# DBTITLE 1,Run utils.py to access helper functions & set up configs
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Lab User's Workspace | UC | Endpoint Names
## Update to use your own values
CATALOG_NAME = "<your_UC_catalog_name>"
SCHEMA_NAME = "<your_UC_schema_name>"

pt_endpoint_name = f"systemai_endpt_pt_{endpt_name_suffix}" ## endpt_name_suffix variable is derived in ./utils.py
ENDPOINT_NAME = pt_endpoint_name
print("ENDPOINT_NAME: ", ENDPOINT_NAME)

external_endpoint_name = f"external_endpt_{endpt_name_suffix}" ## endpt_name_suffix variable is derived in ./utils.py
EXT_ENDPOINT_NAME = external_endpoint_name
print("EXT_ENDPOINT_NAME: ", EXT_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1.1 Delete Endpoints & Check Serving Endpoints UI
# MAGIC To avoid unintentional deletes, we commented the code. **Please uncomment lines to run the clean up**

# COMMAND ----------

# DBTITLE 1,Delete Endpoints
## Delete endpoint when no longer needed 
import mlflow.deployments
client = mlflow.deployments.get_deploy_client('databricks')

# client.delete_endpoint(pt_endpoint_name) ## uncomment to execute

# client.delete_endpoint(external_endpoint_name) ## uncomment to execute

# COMMAND ----------

# DBTITLE 1,Check if info available:  pt_endpoint_name
client.get_endpoint(pt_endpoint_name)


# COMMAND ----------

# DBTITLE 1,check if info available: external_endpoint_name
client.get_endpoint(external_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 List UC Tables to check payload tables to be dropped

# COMMAND ----------

# DBTITLE 1,List tables in UC namespace
# List all tables in the specified catalog and schema
tables_df = spark.sql(f"SHOW TABLES IN {CATALOG_NAME}.{SCHEMA_NAME}")
display(tables_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Delete Payload Tables 
# MAGIC
# MAGIC To avoid unintentional deletes, we commented the code. **Please uncomment lines to run the clean up**

# COMMAND ----------

# DBTITLE 1,Delete Payload Tables
## Drop the payload table(s)

# spark.sql("DROP TABLE IF EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.{pt_endpoint_name}_payload") ## uncomment to execute

# spark.sql("DROP TABLE IF EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.{external_endpoint_name}_payload") ## uncomment to execute

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 List UC Tables to verify tables are dropped 

# COMMAND ----------

# DBTITLE 1,Review tables in UC namespace
# List all tables in the specified catalog and schema
tables_df2 = spark.sql(f"SHOW TABLES IN {CATALOG_NAME}.{SCHEMA_NAME}")
display(tables_df2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Thank you
