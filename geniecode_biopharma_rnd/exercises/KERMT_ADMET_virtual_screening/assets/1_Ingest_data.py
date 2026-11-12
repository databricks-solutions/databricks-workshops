# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # KERMT Public Test Data Ingestion
# MAGIC
# MAGIC Loads the ADMET test set CSV (`all_test_cluster_morgan.csv`) from a Unity Catalog volume into the Delta table `main.kermt.data_public_test`.
# MAGIC
# MAGIC Data from https://figshare.com/articles/dataset/Datasets_for_Multitask_finetuning_and_acceleration_of_chemical_pretrained_models_for_small_molecule_drug_property_prediction_/30350548 (Public data test split)

# COMMAND ----------

# DBTITLE 1,Read CSV and write Delta table
from pyspark.sql import functions as F

src_path = "/Volumes/main/kermt/data/all_test_cluster_morgan.csv"
target_table = "main.kermt.data_public_test"

# Read all columns as strings, then cast: `smiles` stays string, all assay endpoints become double
raw = spark.read.option("header", True).csv(src_path).limit(1000)
df = raw.select(
    F.col("smiles"),
    *[F.col(f"`{c}`").cast("double").alias(c) for c in raw.columns if c != "smiles"]
)

# Overwrite so re-running with a different slice is safe
df.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Wrote {spark.table(target_table).count()} rows, {len(df.columns)} columns to {target_table}")

# COMMAND ----------

# DBTITLE 1,Validate ingested table
tbl = spark.table("main.kermt.data_public_test")

# Compare row counts against the source and count non-null labels per endpoint
src_count = spark.read.option("header", True).csv("/Volumes/main/kermt/data/all_test_cluster_morgan.csv").count()
print(f"Source rows: {src_count}, table rows: {tbl.count()}")

display(tbl.select([F.count(F.col(f"`{c}`")).alias(c) for c in tbl.columns]))
display(tbl.limit(10))