# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Hands-on: label TEDDY's tutorial cells with a Census index
# MAGIC
# MAGIC **H1** sends all 3,538 TEDDY tutorial cells (`sample_data.h5ad`, raw counts) through the shared endpoint `teddy_ep_shared`, writes the 512-d embeddings to my table `dbacademy.<me>.teddy_tutorial_cells`, and draws a UMAP coloured by `Cell.Types`. **H2** labels 200 of these cells by the majority type of their 10 nearest Census cells in the shared index `main.owm.teddy_ref_index` and compares them with the expected interneuron types.
# MAGIC
# MAGIC Shared resources in `main.owm` are read only; this notebook writes only to `dbacademy.<me>`. Embeddings are zero-shot model outputs used for clustering and neighbour search, not measurements.

# COMMAND ----------

# DBTITLE 1,Install packages (one install, one restart)
# MAGIC %pip install -q --upgrade "anndata>=0.12" "typing_extensions>=4.15" umap-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config and workspace client
import os, json, time, shutil
import numpy as np, pandas as pd
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
USER = spark.sql("SELECT current_user()").first()[0]     # read the user at run time
SDK_USER = w.current_user.me().user_name
ME = USER.split("@")[0].replace(".", "_").replace("-", "_")

# Shared (read only)
SHARED_CATALOG, SHARED_SCHEMA, SHARED_VOLUME = "main", "owm", "teddy_files"
SHARED_DIR = f"/Volumes/{SHARED_CATALOG}/{SHARED_SCHEMA}/{SHARED_VOLUME}/shared"
ENDPOINT = "teddy_ep_shared"
INDEX = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.teddy_ref_index"
H5AD = f"{SHARED_DIR}/teddy_70m/data/sample_data.h5ad"
VOCAB = f"{SHARED_DIR}/teddy_70m/teddy/models/teddy_g/70M/vocab.txt"
MANIFEST = f"{SHARED_DIR}/teddy_70m/manifest.json"
VALIDATION = f"{SHARED_DIR}/tutorial_validation.json"

# Mine (write here only)
MY_SCHEMA = f"dbacademy.{ME}"
MY_TABLE = f"{MY_SCHEMA}.teddy_tutorial_cells"
MY_VOLUME = f"{MY_SCHEMA}.teddy_handson"
MY_DIR = f"/Volumes/dbacademy/{ME}/teddy_handson"

# Inference controls (same pooling as the reference index)
POOLING, MAX_SEQ_LEN, BATCH, WORKERS, SEED = "mean", "2048", 10, 4, 42
print(f"user={USER}  me={ME}\nmy table={MY_TABLE}\nmy folder={MY_DIR}")

# COMMAND ----------

# DBTITLE 1,Checks: user, schema, shared files, endpoint, index
# 1. Spark and SDK must be the same user
assert USER == SDK_USER, f"Spark user {USER} != SDK user {SDK_USER}: stop"

# 2. My schema exists (do not create it)
assert spark.sql(f"SHOW SCHEMAS IN dbacademy LIKE '{ME}'").count() == 1, f"{MY_SCHEMA} not found: stop"

# 3. Shared files
for p in [H5AD, VOCAB, MANIFEST]:
    assert os.path.exists(p), f"missing shared file {p}"
print("shared files: OK")

# 4. Endpoint (read only)
ep = w.serving_endpoints.get(ENDPOINT)
se = ep.config.served_entities[0]
print(f"endpoint {ENDPOINT}: ready={ep.state.ready.value}, config_update={ep.state.config_update.value}, "
      f"workload={se.workload_type.value}, scale_to_zero={se.scale_to_zero_enabled}")
assert ep.state.ready.value == "READY", "endpoint not READY"

# 5. Index (read only)
ix = w.vector_search_indexes.get_index(INDEX)
print(f"index {INDEX}: ready={ix.status.ready}, rows={ix.status.indexed_row_count}, vs endpoint={ix.endpoint_name}")
assert ix.status.ready, "index not ready"

# 6. Stop and ask if my table exists
assert not spark.catalog.tableExists(MY_TABLE), f"{MY_TABLE} already exists: stop and ask before writing"

# Endpoint model version + Setup's tutorial check
manifest = json.load(open(MANIFEST))
REVISION = manifest["revision"]
MODEL_VERSION = f"{se.entity_name}/{se.entity_version}|TEDDY-G-{manifest['variant']}@{REVISION}|pooling={POOLING}"
print(f"\nendpoint model: {se.entity_name} version {se.entity_version} (HF revision {REVISION})")
print("model_version tag:", MODEL_VERSION)
if os.path.exists(VALIDATION):
    print("\nSetup tutorial_validation.json:")
    print(json.dumps(json.load(open(VALIDATION)), indent=1))
else:
    print("\nSetup tutorial_validation.json: not run")

# COMMAND ----------

# DBTITLE 1,My volume (create if missing) and empty-folder check
spark.sql(f"CREATE VOLUME IF NOT EXISTS {MY_VOLUME}")
existing = os.listdir(MY_DIR)
assert not existing, f"my folder {MY_DIR} already has files {existing}: stop and ask"
print(f"my folder {MY_DIR}: exists and empty")

# COMMAND ----------

# DBTITLE 1,Load tutorial cells (raw counts, vocabulary genes)
import anndata as ad

adata = ad.read_h5ad(H5AD)
vocab = set(l.strip() for l in open(VOCAB) if l.strip())
raw_genes = np.asarray(adata.raw.var_names)
keep = np.isin(raw_genes, list(vocab))
X = adata.raw.X[:, keep].tocsr()            # raw counts; the endpoint applies TEDDY's normalisation
GENES = raw_genes[keep]
obs = adata.obs

assert X.shape[0] == adata.n_obs and obs.index.is_unique
assert np.allclose(X[:100].data, np.round(X[:100].data)), "expected integer counts in .raw.X"
print(f"{adata.n_obs} cells; {len(raw_genes)} raw genes, {keep.sum()} in the TEDDY vocabulary")
display(obs["Cell.Types"].value_counts().rename("cells").to_frame())

# COMMAND ----------

# DBTITLE 1,Embed every cell through the shared endpoint
from concurrent.futures import ThreadPoolExecutor

var_json = pd.DataFrame({"index": GENES}).to_json(orient="split")

def embed_batch(start):
    idx = slice(start, min(start + BATCH, X.shape[0]))
    rec = [{
        "adata_sparsematrix": X[idx].toarray().tolist(),
        "adata_obs": pd.DataFrame({"cell_id": obs.index[idx]}).to_json(orient="split"),
        "adata_var": var_json,
        "pooling": POOLING,
        "max_seq_len": MAX_SEQ_LEN,
    }]
    for attempt in range(3):                     # stop after 3 failed attempts
        try:
            preds = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=rec).predictions
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    n = idx.stop - idx.start
    assert len(preds) == n, f"batch {start}: {len(preds)} predictions for {n} cells"
    assert {p["pooling"] for p in preds} == {POOLING}, "pooling not applied"
    return start, np.asarray([p["embedding"] for p in preds], dtype=np.float32)

t0 = time.time()
embed_batch(0)                                   # warm-up (endpoint scales to zero)
print(f"warm-up {time.time() - t0:.0f} s")
t0 = time.time()
with ThreadPoolExecutor(WORKERS) as pool:
    parts = dict(pool.map(embed_batch, range(0, X.shape[0], BATCH)))
E = np.vstack([parts[s] for s in sorted(parts)])  # predictions matched to cells by position
print(f"embedded {E.shape[0]} cells x {E.shape[1]} dims in {time.time() - t0:.0f} s")
assert E.shape == (adata.n_obs, 512) and np.isfinite(E).all()

# COMMAND ----------

# DBTITLE 1,Write my table
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

schema = StructType([
    StructField("cell_id", StringType(), False),
    StructField("embedding", ArrayType(FloatType()), False),
    StructField("cell_type", StringType(), True),     # tutorial label Cell.Types (e.g. In1_LHX6-PVALB)
    StructField("donor_id", StringType(), True),
    StructField("model_version", StringType(), False),
])
rows = [(cid, emb.tolist(), ct, d, MODEL_VERSION)
        for cid, emb, ct, d in zip(obs.index.astype(str), E, obs["Cell.Types"].astype(str), obs["donor_id"].astype(str))]
cells_df = spark.createDataFrame(rows, schema)
cells_df.write.mode("error").saveAsTable(MY_TABLE)   # fails if the table exists: never overwrite

display(spark.sql(f"""SELECT count(*) AS cells, count(DISTINCT cell_id) AS distinct_ids,
                       min(size(embedding)) AS min_dim, max(size(embedding)) AS max_dim,
                       count(DISTINCT cell_type) AS cell_types, count(DISTINCT donor_id) AS donors,
                       first(model_version) AS model_version FROM {MY_TABLE}"""))

# COMMAND ----------

# DBTITLE 1,UMAP coloured by Cell.Types, clusters labelled
import umap
import matplotlib.pyplot as plt

U = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=SEED).fit_transform(E)
types = obs["Cell.Types"].astype(str).values
order = obs["Cell.Types"].value_counts().index.tolist()
colors = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(10, 8))
for i, t in enumerate(order):
    m = types == t
    ax.scatter(U[m, 0], U[m, 1], s=4, color=colors(i), alpha=0.6, label=f"{t} (n={m.sum()})", rasterized=True)
for i, t in enumerate(order):                     # full type name on each cluster
    cx, cy = np.median(U[types == t], axis=0)
    ax.text(cx, cy, t, fontsize=10, weight="bold", ha="center", va="center", color=colors(i),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colors(i), alpha=0.85))
ax.set_title(f"TEDDY-G 70M tutorial cells (n={len(types)}), {POOLING} pooling, via {ENDPOINT}")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(markerscale=4, fontsize=8, loc="best", frameon=True)
plt.tight_layout()

UMAP_PNG = f"{MY_DIR}/umap_cell_types.png"
fig.savefig("/tmp/umap_cell_types.png", dpi=150)       # write to /tmp, then copy to the volume
shutil.copy("/tmp/umap_cell_types.png", UMAP_PNG)
plt.show()
print("saved", UMAP_PNG)

# COMMAND ----------

# DBTITLE 1,H2 header
# MAGIC %md
# MAGIC ## H2: look up 200 tutorial cells in the Census index
# MAGIC
# MAGIC A seeded, balanced sample of 200 cells (about the same number per `Cell.Types`) is queried against `main.owm.teddy_ref_index` (1,125 Census brain cells, 9 types) with 10 neighbours; each cell gets the most common neighbour type (ties go to the type seen first, i.e. nearest). Expected: In1/In2 = pvalb, In3/In4 = sst, In5/In6 = lamp5, In7 = VIP (approximate) GABAergic cortical interneuron.

# COMMAND ----------

# DBTITLE 1,Sample 200 cells and query the index (10 neighbours)
# H2 needs only the install and config cells above (it reads my table, so H1 need not be re-run)
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

N_QUERY, K = 200, 10
EXPECTED = {
    "In1_LHX6-PVALB": "pvalb GABAergic cortical interneuron",
    "In2_LHX6-PVALB-Chandelier": "pvalb GABAergic cortical interneuron",
    "In3_LHX6-SST": "sst GABAergic cortical interneuron",
    "In4_LHX6-SST-NPY": "sst GABAergic cortical interneuron",
    "In5_LHX6-ADARB2-LAMP5": "lamp5 GABAergic cortical interneuron",
    "In6_ADARB2-LAMP5": "lamp5 GABAergic cortical interneuron",
    "In7_ADARB2-CALB2": "VIP GABAergic cortical interneuron",   # approximate mapping
}
APPROX = {"In7_ADARB2-CALB2"}

# Read back from my table (the persisted embeddings)
tab = spark.table(MY_TABLE).select("cell_id", "embedding", "cell_type", "model_version").toPandas()
assert tab.model_version.nunique() == 1
MODEL_VERSION = tab.model_version.iloc[0]
assert set(tab.cell_type) == set(EXPECTED), "unexpected Cell.Types"

# Balanced seeded sample: 200 // 7 per type, remainder to the first types (sorted)
TYPES = sorted(EXPECTED)
per = {t: N_QUERY // len(TYPES) + (i < N_QUERY % len(TYPES)) for i, t in enumerate(TYPES)}
q = pd.concat([tab[tab.cell_type == t].sample(n=per[t], random_state=SEED) for t in TYPES], ignore_index=True)
assert len(q) == N_QUERY and q.cell_id.is_unique

# Index model must match ours (same revision and pooling)
idx_mv = spark.table("main.owm.teddy_ref_cells").select("model_version").distinct().toPandas().model_version.tolist()
print("index model_version:", idx_mv)
assert len(idx_mv) == 1 and MODEL_VERSION.endswith(idx_mv[0]), "index and query embeddings come from different models/pooling"

def lookup(vec):
    for attempt in range(3):
        try:
            r = w.vector_search_indexes.query_index(index_name=INDEX, columns=["cell_id", "cell_type"],
                                                    query_vector=list(map(float, vec)), num_results=K)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    rows = r.result.data_array
    assert len(rows) == K, f"got {len(rows)} neighbours"
    return [row[1] for row in rows]               # columns in requested order, score last

with ThreadPoolExecutor(8) as pool:
    q["neighbours"] = list(pool.map(lookup, q.embedding))

def vote(nbrs):
    c = Counter(nbrs); top = max(c.values())
    return next(t for t in nbrs if c[t] == top)   # tie -> nearest of the tied types

q["predicted"] = q.neighbours.map(vote)
q["votes"] = [Counter(n)[p] for n, p in zip(q.neighbours, q.predicted)]
q["expected"] = q.cell_type.map(EXPECTED)
q["match"] = q.predicted == q.expected
display(q.cell_type.value_counts().sort_index().rename("query cells").to_frame())

# COMMAND ----------

# DBTITLE 1,Count table and % matching the expected type
PRED_ORDER = ["pvalb GABAergic cortical interneuron", "sst GABAergic cortical interneuron",
              "lamp5 GABAergic cortical interneuron", "VIP GABAergic cortical interneuron"]
PRED_ORDER += sorted(set(q.predicted) - set(PRED_ORDER))
counts = pd.crosstab(q.cell_type, q.predicted).reindex(index=TYPES, columns=PRED_ORDER, fill_value=0)
counts.index.name, counts.columns.name = "tutorial type", "predicted type"
print("Count table (tutorial type x predicted type):")
display(counts.reset_index())

per_type = (q.groupby("cell_type")
             .agg(cells=("match", "size"), matching=("match", "sum"), median_votes=("votes", "median"))
             .reindex(TYPES))
per_type["expected type"] = [EXPECTED[t] + (" (approximate)" if t in APPROX else "") for t in TYPES]
per_type["% matching"] = (100 * per_type.matching / per_type.cells).round(1)
OVERALL = 100 * q.match.mean()
print(f"Overall: {q.match.sum()} / {len(q)} = {OVERALL:.1f}% match the expected type "
      f"(chance with 9 equal-size reference types = 11.1%)")
print(f"Excluding approximate In7: {100 * q[~q.cell_type.isin(APPROX)].match.mean():.1f}%")
display(per_type.reset_index())

# COMMAND ----------

# DBTITLE 1,Confusion-matrix figure (expected type marked)
from matplotlib.patches import Rectangle

M = counts.values
fig, ax = plt.subplots(figsize=(2.0 * M.shape[1] + 5, 0.8 * M.shape[0] + 3))
im = ax.imshow(M, cmap="Blues", vmin=0)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=11,
                color="white" if M[i, j] > M.max() / 2 else "black")
    j_exp = PRED_ORDER.index(EXPECTED[TYPES[i]])
    ax.add_patch(Rectangle((j_exp - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2.5,
                           ls="--" if TYPES[i] in APPROX else "-"))
ax.set_xticks(range(len(PRED_ORDER)))
ax.set_xticklabels([c.replace(" GABAergic cortical interneuron", "") for c in PRED_ORDER], fontsize=10)
ax.set_yticks(range(len(TYPES)))
ax.set_yticklabels([f"{t} (n={per[t]})" for t in TYPES], fontsize=9)
ax.set_xlabel("predicted type: ... GABAergic cortical interneuron (majority of 10 Census neighbours)")
ax.set_ylabel("tutorial type (Cell.Types)")
ax.set_title(f"Census index lookup, {len(q)} tutorial cells: {OVERALL:.1f}% match the expected type\n"
             "red box = expected type (dashed = approximate, In7 = VIP)")
fig.colorbar(im, ax=ax, fraction=0.03, label="cells")
plt.tight_layout()

CM_PNG = f"{MY_DIR}/confusion_matrix.png"
fig.savefig("/tmp/confusion_matrix.png", dpi=150)
shutil.copy("/tmp/confusion_matrix.png", CM_PNG)
plt.show()
print("saved", CM_PNG)

# COMMAND ----------

# DBTITLE 1,Cross-check: exact kNN over the reference table
# Same 200 cells, exact squared-L2 search over all reference embeddings (no approximate index)
ref_emb = spark.table("main.owm.teddy_ref_cells").select("cell_type", "embedding").toPandas()
Rr = np.vstack(ref_emb.embedding.values).astype(np.float64)
Qq = np.vstack(q.embedding.values).astype(np.float64)
d2 = (Qq ** 2).sum(1)[:, None] + (Rr ** 2).sum(1)[None, :] - 2 * Qq @ Rr.T
nn = np.argsort(d2, axis=1)[:, :K]
exact_pred = [vote(list(ref_emb.cell_type.values[r])) for r in nn]
EXACT_MATCH = int(sum(p == e for p, e in zip(exact_pred, q.expected)))
SAME_AS_INDEX = int(sum(p == i for p, i in zip(exact_pred, q.predicted)))
print(f"exact kNN: {EXACT_MATCH}/{len(q)} match the expected type; "
      f"{SAME_AS_INDEX}/{len(q)} cells get the same label as the index lookup")

# COMMAND ----------

# DBTITLE 1,What H2 shows and what it can't
# MAGIC %md
# MAGIC ### What this shows and what it can't
# MAGIC
# MAGIC Using only zero-shot TEDDY-G 70M embeddings, 190 of 200 tutorial interneurons (95.0%; 95.9% without the approximate In7) were labelled with their expected Census subclass by a 10-neighbour vote, against 11.1% by chance over the 9 reference types; per type it ranged from 89.3% (In7 to VIP) to 100% (In2, In5, In6), and the 10 misses stayed among the four interneuron subclasses (mostly cells sent to VIP or lamp5). It cannot show that the labels are biologically right beyond this mapping: In7 = VIP is approximate, the expected types are a hand mapping of tutorial clusters, only 28-29 cells per type were tested, the reference has just 125 cells per type, and some votes were split (median 5 of 10 for In4, 6 of 10 for In1). An exact brute-force nearest-neighbour search over `main.owm.teddy_ref_cells` gave the same 190/200, so the result does not depend on the approximate index search.

# COMMAND ----------

# DBTITLE 1,Report header
# MAGIC %md
# MAGIC ## Optional: HTML report
# MAGIC
# MAGIC Builds `handson_report.html` in my folder with the html-report skill's `build_report` (embedded PNGs, no network, a Print / Save as PDF button). Every number comes from this notebook's cells.

# COMMAND ----------

# DBTITLE 1,Report facts and software versions
import sys, platform, importlib.metadata as im
import anndata as ad

manifest = json.load(open(MANIFEST))
validation = json.load(open(VALIDATION)) if os.path.exists(VALIDATION) else None
ep = w.serving_endpoints.get(ENDPOINT)
se = ep.config.served_entities[0]
ix = w.vector_search_indexes.get_index(INDEX)

adata = ad.read_h5ad(H5AD) if "adata" not in dir() else adata
n_vocab = int(np.isin(np.asarray(adata.raw.var_names), [l.strip() for l in open(VOCAB) if l.strip()]).sum())
ref = spark.sql("SELECT cell_type, count(*) AS n, min(cell_id) AS first_id FROM main.owm.teddy_ref_cells GROUP BY cell_type ORDER BY cell_type").toPandas()
my = spark.sql(f"""SELECT count(*) AS cells, count(DISTINCT cell_id) AS ids, min(size(embedding)) AS dim,
                   count(DISTINCT donor_id) AS donors, count(DISTINCT model_version) AS versions FROM {MY_TABLE}""").first().asDict()

FACTS = {
    "endpoint": ENDPOINT, "workload": se.workload_type.value, "scale_to_zero": se.scale_to_zero_enabled,
    "ready": ep.state.ready.value, "model": se.entity_name, "model_version": se.entity_version,
    "hf_source": manifest["source"], "revision": manifest["revision"], "variant": manifest["variant"],
    "license_model_card": manifest.get("license_model_card"), "license_package": manifest.get("license_package_pyproject"),
    "license_note": manifest.get("license_note"),
    "tutorial_cells": adata.n_obs, "raw_genes": adata.raw.shape[1], "vocab_genes": n_vocab,
    "donors": sorted(adata.obs["donor_id"].astype(str).unique()),
    "tissue": sorted(adata.obs["tissue"].astype(str).unique()), "disease": sorted(adata.obs["disease"].astype(str).unique()),
    "assay": sorted(adata.obs["assay"].astype(str).unique()),
    "index_rows": ix.status.indexed_row_count, "index_endpoint": ix.endpoint_name,
    "ref_types": len(ref), "ref_per_type": sorted(ref.n.unique().tolist()), "ref_id_prefix": sorted({i.split("_")[0] for i in ref.first_id}),
    "my_table": my,
}
for k, v in FACTS.items():
    print(f"{k}: {v}")
print("validation:", validation)

VERSIONS = pd.DataFrame(
    [("python", platform.python_version()), ("environment", os.environ.get("DATABRICKS_RUNTIME_VERSION", "serverless"))]
    + [(p, im.version(p)) for p in ["anndata", "numpy", "pandas", "scipy", "umap-learn", "matplotlib", "databricks-sdk", "pyspark"]
       if p in {d.metadata["Name"].lower() for d in im.distributions()}],
    columns=["software", "version"])
display(VERSIONS)

# COMMAND ----------

# DBTITLE 1,html-report helper (build_report from the html-report skill)
# build_report / display_report from the html-report skill's scripts/html_report.py (skill folder is not
# importable from this compute, so the module is defined here unchanged in behaviour).
import base64, datetime as _dt, html, io, os, re, types

hr = types.SimpleNamespace()
DEFAULT_NOTICE = ("This report was generated by Genie Code in collaboration with a researcher. "
                  "Review and edit it carefully before sharing it with others.")
_CSS = """
:root { --ink:#1b1f24; --muted:#5b6470; --rule:#d7dbe0; --accent:#ff3621; --pass:#1a7f37; --fail:#cf222e; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: #fff; font: 16px/1.55 Georgia, "Times New Roman", serif; }
main { max-width: 880px; margin: 0 auto; padding: 32px 24px 64px; }
header.report h1 { font-size: 30px; line-height: 1.2; margin: 0 0 8px; }
header.report .meta { color: var(--muted); font-size: 14px; margin: 0 0 16px; }
.notice { border: 1px solid var(--rule); border-left: 4px solid var(--accent); background: #fafbfc;
          padding: 10px 14px; font-size: 14px; margin: 16px 0 24px; }
.summary { font-size: 15.5px; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
           padding: 12px 0; margin-bottom: 24px; }
h2 { font-size: 21px; margin: 32px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--rule); }
h3 { font-size: 17px; margin: 22px 0 8px; }
p, ul { margin: 0 0 12px; }
code, .path { font-family: Menlo, Consolas, "DejaVu Sans Mono", monospace; font-size: 13.5px;
              background: #f3f4f6; padding: 1px 4px; border-radius: 3px; overflow-wrap: anywhere; }
.path { display: block; padding: 6px 8px; margin: 0 0 12px; }
figure { margin: 18px 0; text-align: center; }
figure img { max-width: 100%; height: auto; }
figcaption, caption { font-size: 14px; color: var(--muted); text-align: left; margin-top: 6px; }
caption { caption-side: bottom; padding-top: 6px; }
.tbl-wrap { overflow-x: auto; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; font: 14px/1.4 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
th, td { border-bottom: 1px solid var(--rule); padding: 5px 8px; text-align: left; vertical-align: top; }
thead th { border-bottom: 2px solid var(--ink); }
td.pass { color: var(--pass); font-weight: 600; }
td.fail { color: var(--fail); font-weight: 600; }
a { color: #0b5cad; }
.print-btn { position: fixed; top: 14px; right: 14px; z-index: 10; padding: 8px 14px; cursor: pointer;
             font: 600 13px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #fff;
             background: var(--ink); border: 0; border-radius: 6px; }
.print-btn:hover { background: var(--accent); }
footer.report { margin-top: 40px; color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--rule);
                padding-top: 8px; }
@media print {
  @page { size: auto; margin: 18mm 16mm; }
  body { font-size: 11pt; }
  main { max-width: none; padding: 0; }
  .print-btn { display: none !important; }
  h2, h3 { break-after: avoid; page-break-after: avoid; }
  figure, table, .tbl-wrap, .notice, .path, tr { break-inside: avoid; page-break-inside: avoid; }
  .tbl-wrap { overflow: visible; }
  a[href^="http"]::after { content: " (" attr(href) ")"; font-size: 9pt; color: var(--muted); }
}
"""
_BLOCK_KEYS = ("p", "subheading", "bullets", "figure", "table", "checks", "path")

def _inline(text):
    s = html.escape(str(text), quote=True)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', s)
    return s

def _fig_to_b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _image_b64(src, dpi):
    if hasattr(src, "savefig"):
        return _fig_to_b64(src, dpi)
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    elif isinstance(src, str) and os.path.exists(os.path.expanduser(src)):
        with open(os.path.expanduser(src), "rb") as f:
            data = f.read()
    else:
        raise ValueError(f"figure must be a matplotlib Figure, PNG bytes or an existing PNG path, got {str(src)[:80]!r}")
    if not data.startswith(b"\x89PNG"):
        raise ValueError("figure data is not a PNG; save figures as PNG before embedding")
    return base64.b64encode(data).decode("ascii")

def _df_to_html(table, max_rows=200, caption_html=None):
    df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(list(table))
    if len(df) > max_rows:
        raise ValueError(f"table has {len(df)} rows (max {max_rows}); summarise it or raise max_rows")
    t = df.to_html(index=False, escape=True, border=0, na_rep="")
    if caption_html:
        end = t.index(">") + 1
        t = t[:end] + f"<caption>{caption_html}</caption>" + t[end:]
    return t

def _block(b, dpi, counters):
    kinds = [k for k in _BLOCK_KEYS if k in b]
    if len(kinds) != 1:
        raise ValueError(f"each block needs exactly one of {_BLOCK_KEYS}; got keys {sorted(b)}")
    k = kinds[0]
    if k == "p":
        return f"<p>{_inline(b['p'])}</p>"
    if k == "subheading":
        return f"<h3>{_inline(b['subheading'])}</h3>"
    if k == "bullets":
        return "<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in b["bullets"]) + "</ul>"
    if k == "path":
        return f'<span class="path">{html.escape(str(b["path"]))}</span>'
    if k == "figure":
        cap = b.get("caption")
        if not cap:
            raise ValueError("every figure needs a caption (a full sentence)")
        counters["fig"] += 1
        alt = html.escape(b.get("alt") or cap, quote=True)
        width = html.escape(str(b.get("width", "100%")), quote=True)
        img = f'<img alt="{alt}" style="width:{width}" src="data:image/png;base64,{_image_b64(b["figure"], dpi)}">'
        return f"<figure>{img}<figcaption><strong>Figure {counters['fig']}.</strong> {_inline(cap)}</figcaption></figure>"
    if k == "table":
        counters["tab"] += 1
        cap = b.get("caption")
        cap_html = f"<strong>Table {counters['tab']}.</strong> {_inline(cap)}" if cap else None
        return f'<div class="tbl-wrap">{_df_to_html(b["table"], b.get("max_rows", 200), cap_html)}</div>'
    if k == "checks":
        rows = []
        for c in b["checks"]:
            passed = c.get("passed")
            if passed is True:
                cls, word = "pass", "PASS"
            elif passed is False:
                cls, word = "fail", "FAIL"
            else:
                cls, word = "", html.escape(str(c.get("status", "reported")))
            rows.append(f"<tr><td>{_inline(c.get('check', ''))}</td><td>{_inline(c.get('threshold', ''))}</td>"
                        f"<td>{_inline(c.get('result', ''))}</td><td class=\"{cls}\">{word}</td></tr>")
        return ('<div class="tbl-wrap"><table><thead><tr><th>Check</th><th>Threshold</th><th>Result</th>'
                '<th>Status</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")
    raise AssertionError(k)

def _render_html(spec, dpi=150, print_button=True):
    if not spec.get("title"):
        raise ValueError("spec needs a title")
    sections = spec.get("sections") or []
    if not sections:
        raise ValueError("spec needs at least one section")
    counters = {"fig": 0, "tab": 0}
    authors = spec.get("authors") or ""
    authors = ", ".join(authors) if isinstance(authors, (list, tuple)) else str(authors)
    date = spec.get("date") or _dt.date.today().isoformat()
    meta = " \u00b7 ".join(x for x in (html.escape(authors), html.escape(str(date))) if x)
    parts = ["<!DOCTYPE html>", '<html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f"<title>{html.escape(str(spec['title']))}</title><style>{_CSS}</style></head><body>"]
    if print_button:
        parts.append('<button class="print-btn" type="button" onclick="window.print()" '
                     'title="Opens the print dialog; choose Save as PDF">Print / Save as PDF</button>')
    parts.append(f'<main><header class="report"><h1>{_inline(spec["title"])}</h1>'
                 f'<p class="meta">{meta}</p></header>')
    parts.append(f'<div class="notice"><strong>Note:</strong> {_inline(spec.get("notice") or DEFAULT_NOTICE)}</div>')
    if spec.get("summary"):
        parts.append(f'<div class="summary"><strong>Summary.</strong> {_inline(spec["summary"])}</div>')
    for s in sections:
        if not s.get("title"):
            raise ValueError("every section needs a title")
        body = "".join(_block(b, dpi, counters) for b in s.get("blocks", []))
        parts.append(f'<section><h2>{_inline(s["title"])}</h2>{body}</section>')
    footer = spec.get("footer") or ("Generated " + _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M") + " UTC.")
    parts.append(f'<footer class="report">{_inline(footer)}</footer></main></body></html>')
    return "\n".join(parts)

def _check_report(page, max_mb=10.0, require_sections=("Methods", "Code availability", "Data availability")):
    problems = []
    if re.search(r'<(img|script|iframe|source|video|audio)\b[^>]*\bsrc="(?!data:)', page, re.I):
        problems.append("an element loads a file by URL (src not data:); embed it instead")
    if re.search(r'<link\b[^>]*rel="?stylesheet', page, re.I) or re.search(r"@import|url\((?!data:)", page, re.I):
        problems.append("external stylesheet, @import or url() found; the page must not load anything")
    text = re.sub(r"<[^>]+>", " ", page)
    if re.search(r"\bTODO\b|&lt;&lt;[A-Z_ ]+&gt;&gt;|<<[A-Z_ ]+>>", text):
        problems.append("TODO or <<PLACEHOLDER>> text remains; fill it or list it for the user")
    if "window.print()" not in page:
        problems.append("no Print / Save as PDF button")
    if 'class="notice"' not in page:
        problems.append("review notice missing")
    heads = [re.sub(r"<[^>]+>", "", h).strip().lower() for h in re.findall(r"<h2>(.*?)</h2>", page, re.S)]
    for name in require_sections:
        if not any(h.startswith(name.lower()) for h in heads):
            problems.append(f"section '{name}' missing")
    mb = len(page.encode("utf-8")) / 1e6
    if mb > max_mb:
        problems.append(f"file is {mb:.1f} MB (> {max_mb} MB); lower dpi or downscale figures")
    return problems

def _build_report(spec, out_path, dpi=150, print_button=True, overwrite=False, **check_kwargs):
    page = _render_html(spec, dpi=dpi, print_button=print_button)
    problems = _check_report(page, **check_kwargs)
    path = os.path.expanduser(out_path)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; choose a new name or pass overwrite=True after asking the user")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return {"path": path, "bytes": len(page.encode("utf-8")), "sections": len(spec["sections"]),
            "figures": page.count("<figure>"), "tables": page.count('<div class="tbl-wrap">'), "problems": problems}

def _display_report(page_or_path, height=900):
    from IPython.display import HTML, display as _display
    page = str(page_or_path)
    if not page.lstrip().startswith("<") and os.path.exists(os.path.expanduser(page)):
        with open(os.path.expanduser(page), encoding="utf-8") as f:
            page = f.read()
    src = html.escape(page, quote=True)
    _display(HTML(f'<iframe srcdoc="{src}" style="width:100%;height:{int(height)}px;border:1px solid #ddd"></iframe>'))

hr.build_report, hr.display_report, hr.check_report = _build_report, _display_report, _check_report
print("html-report helper ready")

# COMMAND ----------

# DBTITLE 1,Build handson_report.html in my folder
REPORT = f"{MY_DIR}/handson_report.html"
v = validation
n_q, n_match = len(q), int(q.match.sum())
excl = 100 * q[~q.cell_type.isin(APPROX)].match.mean()
lo, hi = per_type["% matching"].idxmin(), per_type["% matching"].idxmax()
miss = q[~q.match]
miss_to = miss.predicted.str.split().str[0].value_counts().to_dict()
sample_in_manifest = any("sample_data.h5ad" in str(f) for f in manifest.get("files", []))
my_t = FACTS["my_table"]
rev7 = FACTS["revision"][:7]

counts_tbl = counts.copy()
counts_tbl.columns = [c.replace(" GABAergic cortical interneuron", "") for c in counts_tbl.columns]
counts_tbl = counts_tbl.reset_index().rename(columns={"tutorial type": "tutorial type (Cell.Types)"})
pt_tbl = per_type.reset_index()[["cell_type", "expected type", "cells", "matching", "% matching", "median_votes"]]
pt_tbl.columns = ["tutorial type", "expected type", "cells", "matching", "% matching", "median votes (of 10)"]

spec = {
  "title": f"TEDDY-G {FACTS['variant']} embeddings label {n_match} of {n_q} tutorial interneurons with their expected Census subclass",
  "authors": USER,
  "summary": (f"All {FACTS['tutorial_cells']:,} cells of the TEDDY tutorial dataset (human prefrontal cortex interneurons) were embedded "
              f"through the shared endpoint `{ENDPOINT}` (`{FACTS['model']}` version {FACTS['model_version']}, TEDDY-G {FACTS['variant']} at "
              f"Hugging Face revision `{rev7}`, mean pooling) and written to `{MY_TABLE}`. In a UMAP the 7 tutorial types form separate clusters. "
              f"A balanced, seeded sample of {n_q} cells was then looked up in `{INDEX}` ({FACTS['index_rows']:,} Census brain cells, "
              f"{FACTS['ref_types']} types) with {K} neighbours and labelled by majority vote: **{n_match}/{n_q} ({OVERALL:.1f}%)** matched the "
              f"expected type, against 11.1% by chance, and {excl:.1f}% without the approximate In7 = VIP mapping."),
  "sections": [
    {"title": "Introduction", "blocks": [
      {"p": "The hands-on exercise asks whether a shared TEDDY endpoint and a shared Census index, used without any training, can recover "
             "the interneuron subclasses annotated in TEDDY's own tutorial data. The tutorial cells were embedded once (H1), then a sample "
             "was labelled by its nearest Census cells and compared with a hand mapping of tutorial types to Census subclasses (H2)."}]},
    {"title": "Results", "blocks": [
      {"subheading": "The seven tutorial interneuron types form separate clusters in TEDDY embedding space"},
      {"p": f"The {my_t['cells']:,} embeddings ({my_t['dim']} dimensions, one per cell) separate the 7 `Cell.Types` into distinct UMAP "
             "clusters; the two lamp5 types (In5, In6) lie next to each other, as do the sst types (In3, In4) and In1 pvalb."},
      {"figure": UMAP_PNG if "UMAP_PNG" in dir() else f"{MY_DIR}/umap_cell_types.png",
       "caption": f"UMAP of all {my_t['cells']:,} tutorial cells (cosine metric, 15 neighbours, min_dist 0.3, seed {SEED}), coloured by "
                  "`Cell.Types` with each cluster labelled by its full type name; the types separate cleanly."},
      {"subheading": f"{OVERALL:.1f}% of {n_q} looked-up cells get their expected Census subclass, against 11.1% by chance"},
      {"p": f"Per type, matching ranged from {per_type.loc[lo, '% matching']}% ({lo}) to {per_type.loc[hi, '% matching']}% ({hi}). "
             f"All {len(miss)} misses were still assigned to one of the four interneuron subclasses "
             f"({', '.join(f'{v} to {k}' for k, v in miss_to.items())})."},
      {"table": counts_tbl, "caption": f"Count table of tutorial type (rows) against predicted Census type (columns, '... GABAergic cortical "
                                        f"interneuron'), majority of {K} neighbours, {n_q} cells."},
      {"figure": CM_PNG if "CM_PNG" in dir() else f"{MY_DIR}/confusion_matrix.png",
       "caption": "Confusion matrix of the same counts; the red box marks the expected type for each row (dashed for the approximate "
                  "In7 = VIP mapping). Most cells fall in the expected box."},
      {"table": pt_tbl, "caption": "Percentage of cells matching the expected type per tutorial type, with the median number of the 10 "
                                    "neighbours that agreed with the winning label."}]},
    {"title": "Discussion", "blocks": [
      {"bullets": [
        "Zero-shot TEDDY-G 70M embeddings carry enough subclass signal for a nearest-neighbour vote against a small Census reference to "
        "recover the pvalb, sst, lamp5 and VIP-like interneuron subclasses of the tutorial data.",
        "Errors stay within GABAergic interneurons; no cell was labelled as a glial or excitatory type."]},
      {"subheading": "Limitations"},
      {"bullets": [
        "In7_ADARB2-CALB2 = VIP GABAergic cortical interneuron is an approximate mapping; the other expected types are also a hand mapping "
        "of tutorial labels, not an independent ground truth.",
        f"Only {n_q} cells (28-29 per type) were looked up, and the reference has {FACTS['ref_per_type'][0]} cells per type, so one cell "
        "changes a per-type percentage by several points.",
        f"Some votes were split (median {per_type.median_votes.min():g} of 10 neighbours for {per_type.median_votes.idxmin()}), so labels near "
        "subclass boundaries are less reliable.",
        "Embeddings are model outputs used for clustering and retrieval; this run is a technical check, not a biological or clinical validation."]}]},
    {"title": "Methods", "blocks": [
      {"subheading": "Data sources"},
      {"p": f"Tutorial data: {FACTS['tutorial_cells']:,} cells x {FACTS['raw_genes']:,} genes from "
             f"{len(FACTS['donors'])} donors ({', '.join(FACTS['donors'])}), {FACTS['tissue'][0]}, "
             f"{' and '.join(FACTS['disease'])}, {FACTS['assay'][0]}, read from:"},
      {"path": H5AD},
      {"p": f"Reference: index `{INDEX}` on Vector Search endpoint `{FACTS['index_endpoint']}`, {FACTS['index_rows']:,} rows synced "
             f"from `main.owm.teddy_ref_cells` ({FACTS['ref_types']} brain cell types x {FACTS['ref_per_type'][0]} cells, IDs prefixed "
             f"`{FACTS['ref_id_prefix'][0]}`), embedded with the same revision and mean pooling."},
      {"subheading": "Preprocessing"},
      {"p": f"Raw counts from `.raw.X` were used. Genes were restricted to the TEDDY vocabulary ({FACTS['raw_genes']:,} to "
             f"{FACTS['vocab_genes']:,} genes); normalisation and tokenisation are done by the served model."},
      {"subheading": "Analysis"},
      {"p": f"Cells were sent to `{ENDPOINT}` ({FACTS['workload']}, scale-to-zero) in requests of {BATCH} cells, {WORKERS} in parallel, "
             f"with pooling `{POOLING}` and max_seq_len {MAX_SEQ_LEN}, sent as input columns and echoed in each response; embedding "
             f"took 615 s on CPU serverless notebook compute. Embeddings were written to `{MY_TABLE}` (cell_id, embedding, cell_type = "
             f"`Cell.Types`, donor_id, model_version). For the lookup, {n_q} cells were sampled with seed {SEED}, per type "
             f"({', '.join(f'{t.split(chr(95))[0]} {n}' for t, n in per.items())}), each was queried with k = {K}, and the most "
             "common neighbour type was taken (ties go to the nearer type)."},
      {"subheading": "Checks"},
      {"checks": [
        {"check": "Setup tutorial check (`tutorial_validation.json`)", "threshold": f"min cosine >= {v['threshold']}",
         "result": f"min {v['min_cosine']:.9f}, median {v['median_cosine']:.9f} over {v['cells']:,} cells ({v['utc'][:10]})",
         "passed": v["result"] == "PASS"} if v else {"check": "Setup tutorial check", "status": "not run"},
        {"check": "My table: one row per tutorial cell", "threshold": f"{FACTS['tutorial_cells']:,} rows, unique cell_id",
         "result": f"{my_t['cells']:,} rows, {my_t['ids']:,} unique", "passed": my_t["cells"] == my_t["ids"] == FACTS["tutorial_cells"]},
        {"check": "Embedding dimension", "threshold": "512", "result": str(my_t["dim"]), "passed": my_t["dim"] == 512},
        {"check": "Index and query model match", "threshold": "same revision and pooling", "result": idx_mv[0],
         "passed": MODEL_VERSION.endswith(idx_mv[0])},
        {"check": "Exact kNN vs index lookup", "threshold": "same labels", "result": f"{SAME_AS_INDEX}/{n_q} identical; exact kNN "
         f"{EXACT_MATCH}/{n_q} match", "passed": SAME_AS_INDEX == n_q}]},
      {"subheading": "Timings and resources"},
      {"p": f"This run created only the table `{MY_TABLE}` and the volume `{MY_VOLUME}` (storage only). The endpoint and index are "
             "shared workshop resources that this run did not create or change; the endpoint scales to zero when idle."},
      {"subheading": "Software"},
      {"table": VERSIONS, "caption": "Software versions on the notebook compute, from importlib.metadata."}]},
    {"title": "Code availability", "blocks": [
      {"p": f"The notebook `teddy_handson_{ME}` (in `/Users/{USER}/teddy_handson/`) and the serving wrapper behind "
             f"`{FACTS['model']}` are workspace-only and not publicly released. The model code and weights are public at "
             f"[Merck/TEDDY on Hugging Face]({FACTS['hf_source']}), revision `{FACTS['revision']}` (licences declared: model card "
             f"{', '.join(FACTS['license_model_card'])}, package {', '.join(FACTS['license_package'])}; {FACTS['license_note']})."}]},
    {"title": "Data availability", "blocks": [
      {"p": (f"The tutorial data `data/sample_data.h5ad` is part of the {'pinned ' if sample_in_manifest else ''}Hugging Face release "
              f"above, copied to the shared volume. ") +
             "Reference cells come from CELLxGENE Census; the Census release is recorded only as the cell-ID prefix above. "
             "Derived data from this run:"},
      {"path": MY_TABLE}, {"path": f"{MY_DIR}/umap_cell_types.png"}, {"path": f"{MY_DIR}/confusion_matrix.png"}]},
    {"title": "References", "blocks": [
      {"bullets": [
        "TEDDY: A Family Of Foundation Models For Understanding Single Cell Biology. [arXiv:2503.03485](https://arxiv.org/abs/2503.03485).",
        f"TEDDY model card, [Merck/TEDDY]({FACTS['hf_source']}).",
        "CELLxGENE Census, [chanzuckerberg.github.io/cellxgene-census](https://chanzuckerberg.github.io/cellxgene-census/)."]}]},
  ],
}

# overwrite=True only replaces this notebook's own earlier build; the folder was checked empty before H1 wrote to it
out = hr.build_report(spec, REPORT, overwrite=True)
print({k: v for k, v in out.items()})
assert not out["problems"], out["problems"]
hr.display_report(out["path"])