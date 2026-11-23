# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # OAS ingestion & QC
# MAGIC Ingests one Observed Antibody Space (OAS) unpaired heavy-chain file (Kim et al. 2020, SRR12326763, IGHG) into `yen.proteinsearch`:
# MAGIC * raw `.csv.gz` → UC Volume `/Volumes/yen/proteinsearch/raw/oas/`
# MAGIC * raw rows → `yen.proteinsearch.oas_raw`
# MAGIC * PySpark QC → `yen.proteinsearch.oas_qc`
# MAGIC * unique AA sequences → `yen.proteinsearch.oas_unique`, and those with ≥3 occurrences → `yen.proteinsearch.oas_unique_min3`
# MAGIC
# MAGIC Occurrences = sum of OAS `Redundancy` (read count per unique nucleotide sequence) across all nucleotide rows that translate to the same `sequence_alignment_aa`.

# COMMAND ----------

# DBTITLE 1,Config and UC objects
CATALOG = "yen"
SCHEMA = "proteinsearch"
VOLUME = "raw"
OAS_URL = "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Kim_2020/csv/SRR12326763_1_Heavy_IGHG.csv.gz"
FILE_NAME = OAS_URL.rsplit("/", 1)[-1]
RAW_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/oas"
RAW_PATH = f"{RAW_DIR}/{FILE_NAME}"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(RAW_PATH)

# COMMAND ----------

# DBTITLE 1,Download raw file to Volume
import os, gzip, json, csv, urllib.request

os.makedirs(RAW_DIR, exist_ok=True)
if not os.path.exists(RAW_PATH):
    urllib.request.urlretrieve(OAS_URL, RAW_PATH)
print(f"{RAW_PATH}: {os.path.getsize(RAW_PATH)/1e6:.1f} MB")

# OAS files start with one CSV-quoted JSON metadata line before the CSV header
with gzip.open(RAW_PATH, "rt") as f:
    meta = json.loads(next(csv.reader([f.readline()]))[0])
meta

# COMMAND ----------

# DBTITLE 1,Load raw CSV into oas_raw
from pyspark.sql import functions as F

raw_df = (
    spark.read.format("csv")
    .option("header", True)
    .option("skipRows", 1)          # skip OAS JSON metadata line
    .option("quote", '"')
    .option("escape", '"')
    .option("inferSchema", False)
    .load(RAW_PATH)
    .withColumn("Redundancy", F.col("Redundancy").cast("int"))
    .withColumn("run_id", F.lit(meta["Run"]))
    .withColumn("subject", F.lit(meta["Subject"]))
    .withColumn("timepoint", F.lit(meta["Longitudinal"]))
    .withColumn("disease", F.lit(meta["Disease"]))
    .withColumn("isotype", F.lit(meta["Isotype"]))
    .withColumn("source_file", F.col("_metadata.file_path"))
)

raw_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{CATALOG}.{SCHEMA}.oas_raw")
oas_raw = spark.table(f"{CATALOG}.{SCHEMA}.oas_raw")
print("rows:", oas_raw.count(), "| total reads (sum Redundancy):", oas_raw.agg(F.sum("Redundancy")).first()[0])
print("metadata says -> unique:", meta["Unique sequences"], "| total:", meta["Total sequences"])
display(oas_raw.select("sequence_alignment_aa", "cdr3_aa", "v_call", "j_call", "productive", "Redundancy").limit(5))

# COMMAND ----------

# DBTITLE 1,QC profile
aa = F.col("sequence_alignment_aa")
qc_flags = oas_raw.select(
    F.count("*").alias("rows"),
    F.sum(F.when(aa.isNull() | (F.length(aa) == 0), 1).otherwise(0)).alias("null_aa"),
    F.sum(F.when(F.col("productive") != "T", 1).otherwise(0)).alias("non_productive"),
    F.sum(F.when(F.col("stop_codon") == "T", 1).otherwise(0)).alias("stop_codon"),
    F.sum(F.when(F.col("vj_in_frame") != "T", 1).otherwise(0)).alias("vj_out_of_frame"),
    F.sum(F.when(aa.rlike("[^ACDEFGHIKLMNPQRSTVWY]"), 1).otherwise(0)).alias("non_canonical_aa"),
    F.sum(F.when(F.col("cdr3_aa").isNull(), 1).otherwise(0)).alias("null_cdr3"),
    F.min(F.length(aa)).alias("min_len"),
    F.percentile_approx(F.length(aa), 0.5).alias("median_len"),
    F.max(F.length(aa)).alias("max_len"),
)
display(qc_flags)

# COMMAND ----------

# DBTITLE 1,QC criteria
# MAGIC %md
# MAGIC ## QC criteria
# MAGIC A row is kept in `oas_qc` only if it passes **all** of the following checks (applied to `sequence_alignment_aa` after trimming and upper-casing):
# MAGIC
# MAGIC | # | Criterion | Rule | Rationale |
# MAGIC | --- | --- | --- | --- |
# MAGIC | 1 | Productive | `productive == 'T'` | Drop non-functional rearrangements |
# MAGIC | 2 | V-J in frame | `vj_in_frame == 'T'` | Drop frameshifted V(D)J junctions |
# MAGIC | 3 | No stop codon | `stop_codon == 'F'` | Drop truncated sequences |
# MAGIC | 4 | Non-null sequence | `sequence_alignment_aa` not null | Need a sequence to embed |
# MAGIC | 5 | Canonical amino acids only | no characters outside `ACDEFGHIKLMNPQRSTVWY` | Drop ambiguous `X` / `*` residues that embedding models handle poorly |
# MAGIC | 6 | VH length | 100–150 aa (`aa_length`) | Typical VH domain range; removes fragments and over-long reads |
# MAGIC | 7 | CDR3 length | 3–35 aa (`cdr3_aa_length`) | Removes implausible or missing CDR3 annotations |
# MAGIC | 8 | Valid read count | `Redundancy` not null and > 0 | Needed to count occurrences |
# MAGIC
# MAGIC **Results on SRR12326763 (IGHG):** 248,957 of 251,928 rows kept (98.8%). Rows failing each check: checks 1–4 = 0; check 5 (non-canonical amino acids) = 1,174; check 6 (VH length) = 1,782; check 7 (CDR3 length) = 25; check 8 = 0. A row can fail more than one check, so these add up to slightly more than the 2,971 rows removed. Raw VH lengths range from 54 to 158 aa (median 111).
# MAGIC
# MAGIC **Deduplication:** `oas_unique` groups by `sequence_alignment_aa`, with `occurrences = sum(Redundancy)` (total reads). OAS files are already one row per amino-acid sequence, so `oas_unique` has the same row count as `oas_qc`. `oas_unique_min3` keeps sequences with `occurrences >= 3` (26,013 sequences).
# MAGIC
# MAGIC *Note: these are simple sequence-level filters. They don't check ANARCI numbering or V-gene identity.*

# COMMAND ----------

# DBTITLE 1,Apply QC filters -> oas_qc
MIN_LEN, MAX_LEN = 100, 150   # typical VH domain length range
MIN_CDR3, MAX_CDR3 = 3, 35

oas_qc = (
    oas_raw
    .withColumn("sequence_alignment_aa", F.upper(F.trim(aa)))
    .filter(F.col("productive") == "T")
    .filter(F.col("vj_in_frame") == "T")
    .filter(F.col("stop_codon") == "F")
    .filter(F.col("sequence_alignment_aa").isNotNull())
    .filter(~F.col("sequence_alignment_aa").rlike("[^ACDEFGHIKLMNPQRSTVWY]"))  # only 20 canonical AAs (no X, *)
    .withColumn("aa_length", F.length("sequence_alignment_aa"))
    .filter(F.col("aa_length").between(MIN_LEN, MAX_LEN))
    .withColumn("cdr3_aa_length", F.length("cdr3_aa"))
    .filter(F.col("cdr3_aa_length").between(MIN_CDR3, MAX_CDR3))
    .filter(F.col("Redundancy").isNotNull() & (F.col("Redundancy") > 0))
)

oas_qc.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{CATALOG}.{SCHEMA}.oas_qc")
oas_qc = spark.table(f"{CATALOG}.{SCHEMA}.oas_qc")
n_raw, n_qc = oas_raw.count(), oas_qc.count()
print(f"QC kept {n_qc:,} / {n_raw:,} rows ({n_qc/n_raw:.1%})")

# COMMAND ----------

# DBTITLE 1,Unique sequences and >=3 occurrences
oas_unique = (
    oas_qc.groupBy("sequence_alignment_aa")
    .agg(
        F.sum("Redundancy").alias("occurrences"),          # total reads
        F.count("*").alias("n_nt_variants"),               # distinct nucleotide sequences
        F.first("cdr3_aa").alias("cdr3_aa"),
        F.first("v_call").alias("v_call"),
        F.first("d_call").alias("d_call"),
        F.first("j_call").alias("j_call"),
        F.first("aa_length").alias("aa_length"),
        F.first("cdr3_aa_length").alias("cdr3_aa_length"),
        F.first("run_id").alias("run_id"),
    )
    .withColumn("seq_id", F.concat(F.lit("OAS_"), F.substring(F.sha2("sequence_alignment_aa", 256), 1, 16)))
    .select("seq_id", "sequence_alignment_aa", "occurrences", "n_nt_variants", "cdr3_aa", "cdr3_aa_length",
            "aa_length", "v_call", "d_call", "j_call", "run_id")
)
oas_unique.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{CATALOG}.{SCHEMA}.oas_unique")
oas_unique = spark.table(f"{CATALOG}.{SCHEMA}.oas_unique")

oas_unique_min3 = oas_unique.filter(F.col("occurrences") >= 3)
oas_unique_min3.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{CATALOG}.{SCHEMA}.oas_unique_min3")
oas_unique_min3 = spark.table(f"{CATALOG}.{SCHEMA}.oas_unique_min3")

print(f"unique AA sequences: {oas_unique.count():,}")
print(f"unique AA sequences with >=3 occurrences: {oas_unique_min3.count():,}")
display(oas_unique_min3.orderBy(F.desc("occurrences")).limit(10))