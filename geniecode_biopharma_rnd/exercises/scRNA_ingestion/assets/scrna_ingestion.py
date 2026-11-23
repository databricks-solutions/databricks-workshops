# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # scRNA-seq h5ad to Delta ingestion (dashboard-ready)
# MAGIC
# MAGIC Reads every `.h5ad` file in the UC Volume `/Volumes/dbacademy/scrna/raw_h5ad/` and writes them to Delta tables with **cspray** (Spark-distributed, so the expression matrix never has to fit in driver memory).
# MAGIC
# MAGIC **Pathway: metadata-only (Pathway A).** These look like pre-processed (CELLxGENE-style) files, so we keep the author's processed `.X` and existing obs annotations as-is. No QC / normalisation / clustering is re-run.
# MAGIC
# MAGIC **Output tables** (in `TABLE_BASE`):
# MAGIC
# MAGIC | Table | Grain | Join keys |
# MAGIC | --- | --- | --- |
# MAGIC | `bronze_obs` | one row per cell (promoted typed columns + `obs_data` VARIANT) | `fp_int`, `cell_idx` |
# MAGIC | `bronze_x` | one row per cell x gene (long format) | `fp_int`, `cell_idx`, `gene_name` |
# MAGIC | `bronze_var` | one row per gene per sample | `fp_int`, `gene_name` |
# MAGIC | `bronze_sam` | one row per sample | `fp_int` |
# MAGIC
# MAGIC Dashboard prep: liquid clustering (`bronze_x` by `gene_name, fp_int`; `bronze_obs` by `disease, tissue`), table/column comments, and lightweight aggregate views for fast widgets.

# COMMAND ----------

# DBTITLE 1,Install cspray
# MAGIC %pip install -q git+https://github.com/databricks-solutions/cspray.git@main anndata
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Configuration
import mlflow
mlflow.autolog(disable=True)  # cspray runs PCA/k-means internally; avoid spurious runs

SOURCE_DIR = "/Volumes/dbacademy/scrna/raw_h5ad/"
# Output schema: your personal writable schema (dbacademy.scrna is read-only source)
CATALOG = "dbacademy"
SCHEMA = spark.sql("SELECT regexp_replace(split(current_user(), '@')[0], '[^a-zA-Z0-9_]', '_')").first()[0]
TABLE_BASE = f"{CATALOG}.{SCHEMA}"
JOIN_CHAR = ".bronze_"
OWNER = spark.sql("SELECT current_user()").first()[0]

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TABLE_BASE}")
print("Source:", SOURCE_DIR)
print("Target:", TABLE_BASE)

# COMMAND ----------

# DBTITLE 1,Discover h5ad files
files = [f for f in dbutils.fs.ls(SOURCE_DIR) if f.path.endswith(".h5ad")]
# h5py needs POSIX paths, not dbfs:-prefixed ones
h5ad_paths = [f.path.replace("dbfs:", "") for f in files]

total_gb = sum(f.size for f in files) / 1e9
print(f"{len(h5ad_paths)} h5ad files, {total_gb:.2f} GB total")
for f in files:
    print(f"  {f.name:60s} {f.size/1e6:10.1f} MB")
assert h5ad_paths, f"No .h5ad files found in {SOURCE_DIR}"

# COMMAND ----------

# DBTITLE 1,Profile obs schemas (backed mode)
import anndata as ad
import pandas as pd

# Lazy read: obs/var only, no expression matrix in memory
profile, dotted = [], {}
for p in h5ad_paths:
    a = ad.read_h5ad(p, backed="r")
    bad = [c for c in a.obs.columns if "." in c]
    if bad:
        dotted[p] = bad
    profile.append({
        "file": p.split("/")[-1], "n_cells": a.n_obs, "n_genes": a.n_vars,
        "has_raw": a.raw is not None, "obsm": list(a.obsm.keys()),
        "n_obs_cols": a.obs.shape[1],
    })
    a.file.close()

display(pd.DataFrame(profile))
if dotted:
    print("Files with dotted obs column names (known cspray promotion bug):")
    for k, v in dotted.items():
        print("  ", k, v)

# COMMAND ----------

# DBTITLE 1,Markdown: ingestion
# MAGIC %md
# MAGIC ## Ingest to bronze Delta tables
# MAGIC `from_raw=False` keeps the processed `.X`. `set_intermediary_persistance(persist=False)` is required on serverless (cspray otherwise issues `CLEAR CACHE` / `PERSIST TABLE`, which Spark Connect blocks). Files with dotted obs column names (e.g. `orig.ident`) are excluded from the batch because they break metadata promotion for the whole batch; they are listed above.

# COMMAND ----------

# DBTITLE 1,Ingest with cspray
from cspray.data import SprayData
import cspray as cs

ingest_paths = [p for p in h5ad_paths if p not in dotted]
skipped = [p for p in h5ad_paths if p in dotted]
print(f"Ingesting {len(ingest_paths)} files; skipping {len(skipped)}")

# Optional Ensembl ID -> symbol reference (guard against BioMart returning an HTML error page)
ensembl_reference_df = None
try:
    gt = cs.utils.get_gene_table()
    if "gene_id" in gt.columns:
        ensembl_reference_df = spark.createDataFrame(gt)
except Exception as e:
    print("BioMart unavailable, trying local reference:", str(e)[:120])
if ensembl_reference_df is None:
    try:
        ref = (spark.read.option("header", True)
               .csv("/Volumes/main/genesis_workbench/scanpy_reference/ensembl_genes_hsapiens.csv")
               .withColumnRenamed("ensembl_gene_id", "gene_id")
               .withColumnRenamed("external_gene_name", "gene_name"))
        ensembl_reference_df = ref
    except Exception as e:
        print("No local Ensembl reference; using identifiers as stored in h5ad")

sdata = SprayData.from_h5ads(
    spark,
    path=ingest_paths,
    force_partitioning=500,
    chunk_size=int(6_000_000),
    from_raw=False,
    broadcast_genes=True,
    ensembl_reference_df=ensembl_reference_df,
    obs_metadata_columns="all",
    var_metadata_columns="all",
)
sdata.set_intermediary_persistance(persist=False)  # REQUIRED on serverless

sdata.to_tables_and_reset(spark, table_base=TABLE_BASE, join_char=JOIN_CHAR)
print("Bronze obs / X / var / sam written")

# COMMAND ----------

# DBTITLE 1,Promote shared metadata
# Promote obs fields shared across files (cell_type, disease, tissue, sex, ...) to typed columns
try:
    cs.md.promote_suggested(sdata, which="both", promote_sam=True)
    # Delta rejects ' ,;{}()\n\t=' in column names (e.g. promoted 'iPSC passage'); replace with '_'
    import re
    sdata.sam = sdata.sam.toDF(*[re.sub(r"[ ,;{}()\n\t=]", "_", c) for c in sdata.sam.columns])
    sdata.to_tables_and_reset(
        spark, table_base=TABLE_BASE, join_char=JOIN_CHAR,
        subset=["obs", "var", "sam"],
    )
    print("Promotion complete")
except Exception as e:
    if "UNRESOLVED_COLUMN" in str(e):
        print("Promotion failed on a dotted column name. Bronze tables are intact; "
              "query extra fields with variant_get(obs_data, '$.field', 'STRING').")
        print(str(e)[:500])
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Markdown: dashboard prep
# MAGIC %md
# MAGIC ## Prepare for dashboards
# MAGIC * Liquid clustering: `bronze_x` on `(gene_name, fp_int)` for gene-centric widgets ("expression of gene X across diseases"); `bronze_obs` on `(disease, tissue)` for explorer-style filters.
# MAGIC * Table/column comments and `TBLPROPERTIES` so Catalog Explorer, dashboards, and Genie understand the model.
# MAGIC * Small aggregate views so summary widgets don't scan the per-cell / per-gene tables.

# COMMAND ----------

T = lambda n: f"{TABLE_BASE}.bronze_{n}"
obs_cols = set(spark.table(T("obs")).columns)

spark.sql(f"ALTER TABLE {T('x')} CLUSTER BY (gene_name, fp_int)")
obs_cluster = [c for c in ["disease", "tissue"] if c in obs_cols] or ["fp_int"]
spark.sql(f"ALTER TABLE {T('obs')} CLUSTER BY ({', '.join(obs_cluster)})")

comments = {
    "obs": "One row per cell from source h5ad obs (pre-processed, metadata-only ingest). Promoted typed columns plus obs_data VARIANT. Join to bronze_x on (fp_int, cell_idx), to bronze_sam on fp_int.",
    "x":   "Expression matrix in long format: one row per (cell, gene) with non-zero processed .X value. Join to bronze_obs on (fp_int, cell_idx).",
    "var": "Gene metadata from source h5ad var, one row per gene per sample (fp_int).",
    "sam": "One row per source h5ad sample (fp_int, file_path, plus any promoted sample-level fields).",
}
for t, c in comments.items():
    spark.sql(f"COMMENT ON TABLE {T(t)} IS '{c}'")
    spark.sql(f"""ALTER TABLE {T(t)} SET TBLPROPERTIES (
        'quality'='bronze', 'data_owner'='{OWNER}', 'domain'='single_cell',
        'source'='{SOURCE_DIR}')""")

# COMMAND ----------

# DBTITLE 1,Clustering, comments, properties
col_comments = {
    "obs": {"fp_int": "Sample key: integer hash of source file_path",
            "cell_idx": "Cell index within sample; (fp_int, cell_idx) uniquely identifies a cell",
            "obs_data": "All original obs fields as VARIANT; read with variant_get(obs_data, '$.field', 'STRING')"},
    "x":   {"fp_int": "Sample key: integer hash of source file_path",
            "cell_idx": "Cell index within sample",
            "gene_idx": "Gene index within sample var",
            "gene_name": "Gene symbol (resolved via Ensembl reference when available; may be NULL)",
            "expression": "Value from processed .X as published (not re-normalised by this pipeline)"},
}
for t, cols in col_comments.items():
    existing = set(spark.table(T(t)).columns)
    for col, c in cols.items():
        if col in existing:
            spark.sql(f"ALTER TABLE {T(t)} ALTER COLUMN {col} COMMENT '{c.replace(chr(39), chr(92) + chr(39))}'")
print("Clustering, comments and properties applied")

# COMMAND ----------

# DBTITLE 1,Dashboard summary views
# Build group-by views from whichever promoted columns exist
dims = [c for c in ["disease", "tissue", "cell_type", "sex", "assay", "development_stage", "donor_id"] if c in obs_cols]
print("Promoted obs dimensions available:", dims)

if dims:
    spark.sql(f"""
      CREATE OR REPLACE VIEW {TABLE_BASE}.vw_cell_counts
      COMMENT 'Cell counts by sample and promoted metadata dimensions (dashboard summary)'
      AS SELECT fp_int, {', '.join(dims)}, COUNT(*) AS n_cells
      FROM {T('obs')} GROUP BY ALL""")

spark.sql(f"""
  CREATE OR REPLACE VIEW {TABLE_BASE}.vw_sample_summary
  COMMENT 'Per-sample summary: source file, cell count, gene count (dashboard summary)'
  AS SELECT s.fp_int, s.file_path, o.n_cells, v.n_genes
  FROM {T('sam')} s
  LEFT JOIN (SELECT fp_int, COUNT(*) AS n_cells FROM {T('obs')} GROUP BY fp_int) o USING (fp_int)
  LEFT JOIN (SELECT fp_int, COUNT(DISTINCT gene_name) AS n_genes FROM {T('var')} GROUP BY fp_int) v USING (fp_int)""")
print("Views created")

# COMMAND ----------

# DBTITLE 1,Validate outputs
summary = []
for t in ["obs", "x", "var", "sam"]:
    df = spark.table(T(t))
    summary.append({"table": T(t), "rows": df.count(), "n_columns": len(df.columns)})
display(pd.DataFrame(summary))

# Cross-check: cells per sample in Delta vs n_obs read from the h5ad files
delta_cells = spark.table(f"{TABLE_BASE}.vw_sample_summary").select("file_path", "n_cells").toPandas()
delta_cells["file"] = delta_cells["file_path"].str.split("/").str[-1]
check = pd.DataFrame(profile)[["file", "n_cells"]].merge(delta_cells[["file", "n_cells"]], on="file", how="left", suffixes=("_h5ad", "_delta"))
check["match"] = check["n_cells_h5ad"] == check["n_cells_delta"]
display(check)
if dims:
    display(spark.table(f"{TABLE_BASE}.vw_cell_counts").orderBy("n_cells", ascending=False).limit(20))

# COMMAND ----------

# DBTITLE 1,Cell 13
def default_write_fn(spark, df, table_name, partition=False, cluster=False):
    writer = default_writer(df)
    if partition:
        writer = add_sample_partition(writer)
    # if cluster: # cannot aprtitions and cluster...
    #     writer = add_cluster_by(writer)
    writer.saveAsTable(table_name)
    if cluster:
        add_zorder(spark, table_name)

# COMMAND ----------

# DBTITLE 1,Markdown: metadata report
# MAGIC %md
# MAGIC ## Metadata report for dashboard ideation
# MAGIC Profiles every column in `bronze_obs` (per cell) and `bronze_sam` (per sample): fill rate, cardinality, how many samples carry it, and top values. It also lists `obs_data` VARIANT keys that were **not** promoted to columns. Each column gets a suggested dashboard role:
# MAGIC * **Filter / breakdown** – low-cardinality category (bar charts, stacked bars, filter widgets)
# MAGIC * **Flag** – boolean / 2-value field (KPI % counters, toggle filters)
# MAGIC * **Measure** – numeric with spread (histograms, box plots, averages)
# MAGIC * **High-cardinality** – search filter or detail table only
# MAGIC * **ID / ontology** – join keys or ontology term IDs (use the paired label column on charts)
# MAGIC
# MAGIC Columns present in only some samples are flagged; charting them across all samples will show NULL groups.

# COMMAND ----------

# DBTITLE 1,Metadata report: obs and sample
from pyspark.sql import functions as F
import pandas as pd

TOP_N, LOW_CARD = 5, 50
SKIP_TYPES = ("variant", "array", "map", "struct", "binary")
KEY_COLS = {"fp_int", "cell_idx", "cell_barcode", "file_path"}

def role(name, dtype, n_distinct):
    if name in KEY_COLS or name.endswith("_ontology_term_id"):
        return "ID / ontology"
    if n_distinct <= 1:
        return "Constant (context only)"
    if dtype == "boolean" or n_distinct <= 2:
        return "Flag"
    if dtype in ("int", "bigint", "double", "float", "smallint") and n_distinct > 20:
        return "Measure"
    return "Filter / breakdown" if n_distinct <= LOW_CARD else "High-cardinality"

def profile_table(name):
    df = spark.table(name)
    fields = [f for f in df.schema.fields if f.dataType.typeName() not in SKIP_TYPES]  # schema read once
    n_rows = df.count()
    n_samples = df.select("fp_int").distinct().count()
    # One pass: non-null count, distinct values, samples covered, per column
    aggs = []
    for i, f in enumerate(fields):
        c = F.col(f"`{f.name}`")
        aggs += [F.count(c).alias(f"nn{i}"),
                 F.countDistinct(c).alias(f"nd{i}"),
                 F.countDistinct(F.when(c.isNotNull(), F.col("fp_int"))).alias(f"ns{i}")]
    r = df.agg(*aggs).first()
    prof = pd.DataFrame([{
        "column": f.name, "type": f.dataType.simpleString(),
        "pct_non_null": round(100 * r[f"nn{i}"] / max(n_rows, 1), 1),
        "n_distinct": r[f"nd{i}"],
        "samples_with_values": f"{r[f'ns{i}']}/{n_samples}",
        "partial_coverage": r[f"ns{i}"] < n_samples,
        "dashboard_role": role(f.name, f.dataType.simpleString(), r[f"nd{i}"]),
    } for i, f in enumerate(fields)])

    # Top values for low-cardinality, non-ID columns (single unpivot query)
    cat = prof[(prof.n_distinct <= LOW_CARD) & (prof.dashboard_role != "ID / ontology")].column.tolist()
    top = pd.DataFrame()
    if cat:
        stack = ", ".join(f"'{c}', CAST(`{c}` AS STRING)" for c in cat)
        top = (df.selectExpr(f"stack({len(cat)}, {stack}) AS (column, value)")
                 .groupBy("column", "value").count()
                 .selectExpr("*", "row_number() OVER (PARTITION BY column ORDER BY count DESC) AS rk")
                 .filter(f"rk <= {TOP_N}").toPandas())
        top = (top.sort_values(["column", "rk"])
                  .assign(v=lambda d: d.value.fillna("NULL") + " (" + d["count"].astype(str) + ")")
                  .groupby("column").v.apply(" | ".join).rename("top_values").reset_index())
        prof = prof.merge(top, on="column", how="left")
    else:
        prof["top_values"] = None
    print(f"{name}: {n_rows:,} rows across {n_samples} samples")
    return prof, n_rows, n_samples

obs_prof, n_cells, n_obs_samples = profile_table(T("obs"))
display(obs_prof)
sam_prof, _, n_sam = profile_table(T("sam"))
display(sam_prof)

# obs_data VARIANT keys not promoted to columns: candidates for extra dashboard fields
variant_keys = spark.sql(f"""
  SELECT k.key, COUNT(DISTINCT s.fp_int) AS n_samples, first(schema_of_variant(k.value)) AS type
  FROM (SELECT fp_int, any_value(obs_data) AS v FROM {T('obs')} GROUP BY fp_int) s,
       LATERAL variant_explode(s.v) AS k
  GROUP BY k.key ORDER BY n_samples DESC, k.key""").toPandas()
unpromoted = variant_keys[~variant_keys.key.isin(set(obs_prof.column) | set(sam_prof.column))]
print(f"obs_data keys not promoted ({len(unpromoted)}) - query with variant_get(obs_data, '$.<key>', '<TYPE>'):")
display(unpromoted)

# COMMAND ----------

# DBTITLE 1,Dashboard ideas from metadata
# Rule-based dashboard suggestions derived from the profiles above
full = lambda p, r: p[(p.dashboard_role == r) & (~p.partial_coverage)].column.tolist()
partial = lambda p: p[p.partial_coverage & (p.dashboard_role != "ID / ontology")].column.tolist()
const = lambda p: p[(p.dashboard_role == "Constant (context only)") & (~p.partial_coverage)].column.tolist()

obs_dims, obs_flags = full(obs_prof, "Filter / breakdown"), full(obs_prof, "Flag")
obs_hicard = full(obs_prof, "High-cardinality")
sam_dims = full(sam_prof, "Filter / breakdown") + full(sam_prof, "Flag")

ideas = []
ideas.append(("KPI counters", f"Total cells ({n_cells:,}), samples ({n_obs_samples}), distinct "
              + ", ".join(c for c in ["cell_type", "donor_id", "tissue", "disease"] if c in obs_prof.column.values)))
if obs_dims:
    ideas.append(("Global filters", ", ".join(obs_dims)))
    ideas.append(("Composition bars", "Cell counts by " + ", ".join(obs_dims[:4]) + " (vw_cell_counts)"))
if "cell_type" in obs_prof.column.values:
    for d in [c for c in ["disease", "tissue", "sex", "donor_id"] if c in obs_prof.column.values]:
        ideas.append(("Cell-type proportions", f"100% stacked bar of cell_type within each {d}"))
    ideas.append(("Gene expression explorer", "Parameter :gene_name, join bronze_x to bronze_obs, box/violin of expression by cell_type, split by disease"))
if obs_flags:
    ideas.append(("Flag KPIs", "% of cells where " + ", ".join(obs_flags) + " is true, per sample"))
if obs_hicard:
    ideas.append(("Detail table / search", ", ".join(obs_hicard)))
if sam_dims:
    ideas.append(("Sample overview table", "vw_sample_summary joined to bronze_sam: " + ", ".join(sam_dims)))
if const(obs_prof) or const(sam_prof):
    ideas.append(("Header text (single value)", ", ".join(const(obs_prof) + const(sam_prof))))
if partial(obs_prof) or partial(sam_prof):
    ideas.append(("Use with care (partial coverage)", ", ".join(partial(obs_prof) + partial(sam_prof))
                  + " - filter to samples that have them or show as 'not reported'"))
if len(unpromoted):
    ideas.append(("Extra fields via variant_get", ", ".join(unpromoted.key.head(10))))

display(pd.DataFrame(ideas, columns=["dashboard_element", "suggestion"]))