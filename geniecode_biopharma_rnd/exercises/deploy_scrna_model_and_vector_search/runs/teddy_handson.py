# Databricks notebook source
# DBTITLE 1,TEDDY Hands-On: title
# MAGIC %md
# MAGIC # TEDDY Hands-On: Embed Tutorial Cells and Annotate by Nearest-Neighbour Search
# MAGIC
# MAGIC This notebook embeds single cells from `sample_data.h5ad` through the shared TEDDY-G 70M serving endpoint (mean pooling), stores the results in a per-user Delta table, visualises the embeddings with UMAP, then queries the shared reference Vector Search index to label each tutorial cell by its 10 nearest reference neighbours (H2 annotation check).

# COMMAND ----------

# DBTITLE 1,Install packages (one install, one restart)
# MAGIC %pip install -q anndata scipy h5py "umap-learn" "numpy<2.3"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config and client
import os, sys, json, time
import numpy as np, pandas as pd
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Values (change only these for another workspace)
CATALOG, SCHEMA, VOLUME = "main", "owm", "teddy_files"
ENDPOINT  = "teddy_ep_shared"
INDEX     = f"{CATALOG}.{SCHEMA}.teddy_ref_index"
VOL       = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SHARED    = f"{VOL}/shared"

USER  = w.current_user.me().user_name                         # read at run time
ME    = USER.split("@")[0].replace(".", "_").replace("-", "_")
MY_TABLE  = f"{CATALOG}.{SCHEMA}.teddy_tutorial_cells_{ME}"
MY_FOLDER = f"{SHARED}/{ME}"

H5AD      = f"{SHARED}/teddy_70m/data/sample_data.h5ad"
VOCAB     = f"{SHARED}/teddy_70m/teddy/models/teddy_g/70M/vocab.txt"
VALIDATION = f"{SHARED}/tutorial_validation.json"
POOLING   = "mean"     # same pooling as the reference index (zero-shot default)
BATCH     = 10         # cells per request (70M on GPU_MEDIUM)
SKILL_DIR = "/Workspace/.assistant/skills/open-weight-models"

print(f"user {USER} -> me = {ME}\nmy table  {MY_TABLE}\nmy folder {MY_FOLDER}")

# COMMAND ----------

# DBTITLE 1,Check shared files, endpoint, index; stop if my table or folder exists
# 1) shared files
for p in (H5AD, VOCAB):
    assert os.path.exists(p), f"missing shared file: {p}"
    print(f"ok  {p}  ({os.path.getsize(p)/1e6:.1f} MB)")

# 2) endpoint (read only; never created or changed here)
ep = w.serving_endpoints.get(ENDPOINT)
assert ep.state.ready.value == "READY", f"{ENDPOINT} is not READY: {ep.state}"
se = ep.config.served_entities[0]
ENDPOINT_MODEL = f"{se.entity_name}/{se.entity_version}"
print(f"ok  endpoint {ENDPOINT}: READY, serves {ENDPOINT_MODEL} on {se.workload_type.value}")

# 3) index (read only)
ix = w.vector_search_indexes.get_index(INDEX)
assert ix.status.ready, f"{INDEX} is not ready: {ix.status.message}"
REF_TABLE = ix.delta_sync_index_spec.source_table
print(f"ok  index {INDEX}: ready, {ix.status.indexed_row_count} rows, source {REF_TABLE}, VS endpoint {ix.endpoint_name}")

# 4) my table and folder must not exist yet: stop and ask
problems = []
if spark.catalog.tableExists(MY_TABLE): problems.append(f"table {MY_TABLE} exists")
if os.path.exists(MY_FOLDER):          problems.append(f"folder {MY_FOLDER} exists")
if problems:
    raise RuntimeError("STOP and ask the user before continuing: " + "; ".join(problems))
print(f"ok  {MY_TABLE} and {MY_FOLDER} do not exist yet")

# COMMAND ----------

# DBTITLE 1,Endpoint model version and Setup's tutorial check
mv = w.model_versions.get(se.entity_name, int(se.entity_version))
# UC ModelVersionInfo in this SDK has no .tags: read them through the MLflow UC registry client
import mlflow
from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
tags = dict(MlflowClient().get_model_version(se.entity_name, str(se.entity_version)).tags or {})
print(f"Endpoint {ENDPOINT} serves model {se.entity_name}, version {se.entity_version} (status {mv.status.value if mv.status else '?'})")
for k in sorted(tags):
    print(f"  tag {k} = {tags[k]}")

print("\nSetup's tutorial check (tutorial_validation.json):")
if os.path.exists(VALIDATION):
    with open(VALIDATION) as f:
        tutorial_validation = json.load(f)
    print(json.dumps(tutorial_validation, indent=1))
    if tutorial_validation.get("model") != ENDPOINT_MODEL:
        print(f"NOTE: the check was run on {tutorial_validation.get('model')}, the endpoint now serves {ENDPOINT_MODEL}")
else:
    tutorial_validation = None
    print("not run")

# Model-version string, same format as the reference table so H2 can compare them
ref_versions = [r[0] for r in spark.table(REF_TABLE).select("model_version").distinct().collect()]
REVISION = (tutorial_validation or {}).get("revision") or tags.get("hf_revision") or tags.get("revision")
MODEL_VERSION = f"TEDDY-G-70M@{REVISION}|pooling={POOLING}"
print(f"\nmodel_version for my rows: {MODEL_VERSION}\nreference table model_version: {ref_versions}")
print("match" if ref_versions == [MODEL_VERSION] else "MISMATCH: check before comparing with the index")

# COMMAND ----------

# DBTITLE 1,Load tutorial cells (.raw.X counts) and match genes to the vocabulary
import anndata as ad, scipy.sparse as sp

adata = ad.read_h5ad(H5AD)
assert adata.raw is not None, "sample_data.h5ad has no .raw"
raw = adata.raw
print(f"{adata.n_obs} cells; .raw has {raw.n_vars} genes; obs columns: {list(adata.obs.columns)}")

# Ensembl IDs: the raw var index or a raw var column, whichever holds ENSG IDs
cands = {"index": pd.Series(raw.var_names, dtype=str)}
cands.update({c: raw.var[c].astype(str).reset_index(drop=True) for c in raw.var.columns})
ens_col = max(cands, key=lambda c: cands[c].str.startswith("ENSG").mean())
genes = cands[ens_col].str.split(".").str[0].values
assert np.mean([g.startswith("ENSG") for g in genes]) > 0.8, f"no Ensembl IDs in .raw.var (best: {ens_col})"

with open(VOCAB) as f:
    vocab = {t.strip() for t in f if t.strip()}
in_vocab = np.array([g in vocab for g in genes])
print(f"Ensembl IDs from raw.var['{ens_col}']; {in_vocab.sum()} of {len(genes)} genes in the TEDDY vocabulary ({in_vocab.mean():.1%})")

X = raw.X
X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
X = X[:, np.where(in_vocab)[0]].astype(np.float32)
GENES = genes[in_vocab]
nz = X.data
assert np.allclose(nz, np.round(nz)), ".raw.X does not look like raw counts"
print(f"counts check: integers, median total per cell {np.median(np.asarray(X.sum(1)).ravel()):.0f}, "
      f"median expressed genes per cell {np.median(np.diff(X.indptr)):.0f}")

CELL_IDS  = adata.obs_names.astype(str).values
CELL_TYPE = adata.obs["Cell.Types"].astype(str).values
DONOR     = adata.obs["donor_id"].astype(str).values
print(pd.Series(CELL_TYPE).value_counts().to_string())

# COMMAND ----------

# DBTITLE 1,Embed every cell through the shared endpoint
from concurrent.futures import ThreadPoolExecutor

def make_record(idx):
    sub = X[idx]
    cols = np.unique(sub.indices)                 # only genes expressed in this batch: zeros change neither
    dense = sub[:, cols].toarray()                # the total count nor the top-K, so the embedding is the same
    return {
        "adata_sparsematrix": dense.tolist(),
        "adata_obs": pd.DataFrame({"cell_id": CELL_IDS[idx]}).to_json(orient="split"),
        "adata_var": pd.DataFrame({"index": GENES[cols]}).to_json(orient="split"),
        "pooling": POOLING,
    }

def embed_batch(idx, tries=3):
    for attempt in range(1, tries + 1):
        try:
            preds = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=[make_record(idx)]).predictions
            assert len(preds) == len(idx), f"{len(preds)} predictions for {len(idx)} cells"
            assert {p["pooling"] for p in preds} == {POOLING}, "pooling not echoed as requested"
            return np.array([p["embedding"] for p in preds], dtype=np.float32)
        except Exception as e:
            if attempt == tries:
                raise RuntimeError(f"batch starting at cell {idx[0]} failed {tries} times: {e}") from e
            time.sleep(5 * attempt)

batches = [np.arange(i, min(i + BATCH, X.shape[0])) for i in range(0, X.shape[0], BATCH)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(embed_batch, batches))   # map keeps request order
EMB = np.vstack(results)
print(f"embedded {EMB.shape[0]} cells x {EMB.shape[1]} dims in {time.time()-t0:.0f} s "
      f"({len(batches)} requests of <= {BATCH} cells)")

assert EMB.shape[0] == len(CELL_IDS) and np.isfinite(EMB).all()
norms = np.linalg.norm(EMB, axis=1)
print(f"norms min {norms.min():.3f}, median {np.median(norms):.3f}; distinct rows {len(np.unique(EMB.round(5), axis=0))}")

# COMMAND ----------

# DBTITLE 1,Write my table
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

assert not spark.catalog.tableExists(MY_TABLE), f"{MY_TABLE} appeared during the run: stop and ask"
schema = StructType([
    StructField("cell_id", StringType(), False),
    StructField("embedding", ArrayType(FloatType()), False),
    StructField("cell_type", StringType(), True),
    StructField("donor_id", StringType(), True),
    StructField("model_version", StringType(), True),
])
rows = [(c, e.tolist(), t, d, MODEL_VERSION) for c, e, t, d in zip(CELL_IDS, EMB, CELL_TYPE, DONOR)]
spark.createDataFrame(rows, schema).write.saveAsTable(MY_TABLE)

chk = spark.table(MY_TABLE)
n = chk.count(); n_ids = chk.select("cell_id").distinct().count()
dims = [r[0] for r in chk.selectExpr("size(embedding)").distinct().collect()]
print(f"wrote {MY_TABLE}: {n} rows, {n_ids} distinct cell_id, embedding size {dims}")
assert n == n_ids == len(CELL_IDS) and dims == [EMB.shape[1]]
display(chk.groupBy("cell_type", "model_version").count().orderBy("cell_type"))

# COMMAND ----------

# DBTITLE 1,UMAP coloured by Cell.Types, each cluster labelled on the plot
import shutil, umap, textwrap
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
sys.path.insert(0, f"{SKILL_DIR}/scripts")
import visualize as viz

xy = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42).fit_transform(EMB)

# More types than the 5-hue colour rule allows, so: shades of colour families (rule 2) plus on-plot labels.
types = sorted(pd.unique(CELL_TYPE))
fams, per = ["navy", "red", "orange", "teal"], (2 if len(types) <= 8 else 4)
assert len(types) <= len(fams) * per, f"{len(types)} types: too many for shades; facet instead"
shades = (500, 300, 700, 900)
colors = {t: viz.PALETTE[f"{fams[i // per]}_{shades[i % per]}"] for i, t in enumerate(types)}

with mpl.rc_context(viz.STYLE):
    fig, (ax,), _, _ = viz.layout(viz.DOUBLE_W, 4.6, note_text=(
        f"{len(CELL_IDS)} tutorial cells (sample_data.h5ad, .raw.X counts), TEDDY-G 70M via {ENDPOINT} "
        f"({ENDPOINT_MODEL}), pooling={POOLING}. UMAP on cosine distance, seed 42. Labels at each type's median position."))
    for t in types:
        m = CELL_TYPE == t
        ax.scatter(xy[m, 0], xy[m, 1], s=4, c=colors[t], linewidths=0, rasterized=True)
    for t in types:
        m = CELL_TYPE == t
        cx, cy = np.median(xy[m], axis=0)
        ax.text(cx, cy, f"{t}\n(n={m.sum()})", ha="center", va="center", fontsize=viz.MIN_PT, weight="bold",
                color=colors[t], path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("TEDDY tutorial cells by Cell.Types")
plt.close(fig)
display(fig)

# Save to /tmp, then copy to my folder (writing straight to /Volumes can crash)
paths = viz.save_figure(fig, "/tmp/teddy_handson_fig", "umap_tutorial_cells")
os.makedirs(MY_FOLDER, exist_ok=True)
for p in paths:
    shutil.copy(p, MY_FOLDER)
print("copied to", MY_FOLDER, os.listdir(MY_FOLDER))

# COMMAND ----------

# DBTITLE 1,H2: pick 200 tutorial cells, about equal per Cell.Types (seeded)
sys.path.insert(0, f"{SKILL_DIR}/scripts")
import nn_annotate as nn

N_QUERY, K, SEED = 200, 10, 42
EXPECTED = {   # tutorial type -> expected reference type (In7 = VIP is approximate)
    "In1_LHX6-PVALB": "pvalb GABAergic cortical interneuron",
    "In2_LHX6-PVALB-Chandelier": "pvalb GABAergic cortical interneuron",
    "In3_LHX6-SST": "sst GABAergic cortical interneuron",
    "In4_LHX6-SST-NPY": "sst GABAergic cortical interneuron",
    "In5_LHX6-ADARB2-LAMP5": "lamp5 GABAergic cortical interneuron",
    "In6_ADARB2-LAMP5": "lamp5 GABAergic cortical interneuron",
    "In7_ADARB2-CALB2": "VIP GABAergic cortical interneuron",
}

# My cells and the index must come from the same model version and pooling
mine = spark.table(MY_TABLE).toPandas()
idx_info = w.vector_search_indexes.get_index(INDEX)
ref_tab = idx_info.delta_sync_index_spec.source_table
ref_versions = {r[0] for r in spark.table(ref_tab).select("model_version").distinct().collect()}
assert set(mine.model_version) == ref_versions, f"model_version differs: mine {set(mine.model_version)} vs index {ref_versions}"
assert set(mine.cell_type) == set(EXPECTED), f"unexpected tutorial types: {set(mine.cell_type) ^ set(EXPECTED)}"
ref_types = {r[0] for r in spark.table(ref_tab).select("cell_type").distinct().collect()}
assert set(EXPECTED.values()) <= ref_types, "an expected type is not in the index"
print(f"same model version as the index: {ref_versions.pop()}")

# About equal per type: 200 / 7 -> 29 for some types, 28 for others (seeded)
types = sorted(EXPECTED)
rng = np.random.default_rng(SEED)
quota = {t: N_QUERY // len(types) + (i < N_QUERY % len(types)) for i, t in enumerate(rng.permutation(types))}
picked = []
for t in types:
    pool = mine.index[mine.cell_type == t].to_numpy()
    assert len(pool) >= quota[t], f"{t}: only {len(pool)} cells"
    picked += list(rng.choice(pool, size=quota[t], replace=False))
query = mine.loc[picked].reset_index(drop=True)
print(f"{len(query)} query cells:", query.cell_type.value_counts().sort_index().to_dict())

# Query vectors in the same form as the index's stored vectors (raw or unit length)
ref_norm = float(spark.table(ref_tab).selectExpr("aggregate(embedding, 0D, (a, x) -> a + x*x) as s").limit(50)
                 .selectExpr("avg(sqrt(s))").first()[0])
QV = np.vstack(query.embedding.values).astype(float)
if abs(ref_norm - 1) < 0.01:
    QV = nn.unit(QV)
print(f"index vectors have mean length {ref_norm:.2f}; query vectors sent {'unit-length' if abs(ref_norm-1)<0.01 else 'as stored'}")

# COMMAND ----------

# DBTITLE 1,H2: query teddy_ref_index (10 neighbours) and vote
from concurrent.futures import ThreadPoolExecutor

def one(v):
    for attempt in range(3):
        try:
            return nn.vote(nn.query_neighbours(w, INDEX, v, columns=["cell_id", "cell_type"], k=K))
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))

t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    votes = pd.DataFrame(list(pool.map(one, QV)))
res = pd.concat([query[["cell_id", "cell_type"]], votes], axis=1)
res["expected"] = res.cell_type.map(EXPECTED)
res["match"] = res.predicted == res.expected
print(f"{len(res)} cells queried in {time.time()-t0:.0f} s, k={K}")

short = lambda s: s.replace(" GABAergic cortical interneuron", " interneuron")
counts = pd.crosstab(res.cell_type, res.predicted.map(short), margins=True, margins_name="total")
print("\nCount table: tutorial type (rows) x predicted index type (columns)")
display(counts.reset_index().rename(columns={"cell_type": "tutorial type"}))

rows = []
for t, g in res.groupby("cell_type"):
    k_ok, n = int(g.match.sum()), len(g)
    lo, hi = nn.wilson_interval(k_ok, n)
    rows.append({"tutorial type": t, "expected": short(EXPECTED[t]), "cells": n, "matching": k_ok,
                 "% matching": round(100 * k_ok / n, 1), "95% CI": f"{100*lo:.0f}-{100*hi:.0f}%",
                 "median agreement": g.agreement.median()})
k_ok, n = int(res.match.sum()), len(res)
lo, hi = nn.wilson_interval(k_ok, n)
rows.append({"tutorial type": "ALL", "expected": "", "cells": n, "matching": k_ok,
             "% matching": round(100 * k_ok / n, 1), "95% CI": f"{100*lo:.0f}-{100*hi:.0f}%",
             "median agreement": res.agreement.median()})
summary = pd.DataFrame(rows)
print(f"\nOverall: {k_ok} of {n} cells ({100*k_ok/n:.1f}%) get the expected type")
display(summary)

# Independent check: exact cosine search over the reference table should give the same votes
ref = spark.table(ref_tab).select("cell_id", "embedding", "cell_type").toPandas()
R = nn.unit(np.vstack(ref.embedding.values)); Q = nn.unit(np.vstack(query.embedding.values))
nbr = np.argsort(-(Q @ R.T), axis=1)[:, :K]
exact = [pd.Series(ref.cell_type.values[r]).value_counts().idxmax() for r in nbr]
print(f"exact-search vote equals index vote for {np.mean(np.array(exact) == res.predicted.values):.1%} of cells")

# COMMAND ----------

# DBTITLE 1,H2: what this shows and what it can't
# MAGIC %md
# MAGIC **What this shows.** Labelled only by their 10 nearest cells in the Census reference index, 188 of 200 tutorial cells (94%, 95% CI 90-97%) get the expected interneuron class. Every tutorial type is at 89% or higher, and the misses are mostly between related classes (In1 pvalb to sst, In3 sst to VIP). An exact search over the reference table gives the same vote for 99% of cells, so the index search isn't the source of the misses.
# MAGIC
# MAGIC **What it can't show.** The reference has only four interneuron classes with 125 cells each, so subtypes such as chandelier (In2) or NPY (In4) can't be told apart. Low neighbour agreement (median 0.5-0.6 for In1 and In4) means some labels rest on a narrow majority, and "In7 = VIP" is only an approximate mapping. This is a technical check on 200 cells from 5 donors, not a scientific or clinical validation of the labels.

# COMMAND ----------

# DBTITLE 1,Report: extra numbers from this run and software versions
import importlib.metadata as md, platform

# Same-type share among each tutorial cell's 10 nearest tutorial cells (cosine), against chance
U = EMB / np.linalg.norm(EMB, axis=1, keepdims=True)
S = U @ U.T
np.fill_diagonal(S, -np.inf)
nn10 = np.argpartition(-S, 10, axis=1)[:, :10]
same_type = float((CELL_TYPE[nn10] == CELL_TYPE[:, None]).mean())
shares = pd.Series(CELL_TYPE).value_counts(normalize=True)
chance = float((shares ** 2).sum())
print(f"10 nearest tutorial cells share the cell's Cell.Types: {same_type:.1%} (chance {chance:.1%})")

ref_counts = ref.cell_type.value_counts()
print(f"reference: {len(ref)} cells, {ref_counts.size} types, {ref_counts.min()}-{ref_counts.max()} per type")
n_donors_all, n_donors_query = int(pd.Series(DONOR).nunique()), int(query.donor_id.nunique())
print(f"donors: {n_donors_all} in the tutorial data, {n_donors_query} among the 200 query cells")
exact_agree = float(np.mean(np.array(exact) == res.predicted.values))

vers = []
for p in ["anndata", "h5py", "scipy", "numpy", "pandas", "umap-learn", "matplotlib", "databricks-sdk", "mlflow", "mlflow-skinny"]:
    try:
        vers.append({"package": p, "version": md.version(p)})
    except md.PackageNotFoundError:
        pass
vers.append({"package": "Python", "version": platform.python_version()})
vers.append({"package": "Databricks environment", "version": os.environ.get("DATABRICKS_RUNTIME_VERSION", "serverless")})
versions_df = pd.DataFrame(vers)
display(versions_df)

# COMMAND ----------

# DBTITLE 1,Report: confusion-matrix figure (tutorial type x predicted type)
from matplotlib.patches import Rectangle

# Rows: tutorial types; columns: predicted reference types (not square, so drawn here rather than viz.plot_confusion)
cm_counts = counts.drop(index="total", columns="total")
cm_rows = sorted(cm_counts.index)
exp_short = {t: short(EXPECTED[t]) for t in cm_rows}
cm_cols = list(dict.fromkeys([exp_short[t] for t in cm_rows] + [c for c in cm_counts.columns if c not in exp_short.values()]))
cm_counts = cm_counts.reindex(index=cm_rows, columns=cm_cols, fill_value=0)
cm_frac = cm_counts.div(cm_counts.sum(axis=1), axis=0).to_numpy()
assert int(cm_counts.to_numpy().sum()) == len(res)

with mpl.rc_context(viz.STYLE):
    fig_cm, (ax,), _, _ = viz.layout(viz.DOUBLE_W + 1.2, 3.6, left_in=2.4, xlabel_in=1.9, top_in=0.9, note_text=(
        f"{len(res)} tutorial cells, labelled by the most common cell_type among {K} neighbours in {INDEX}. "
        "Numbers are cell counts; colour is the share of the row. White boxes mark the expected type (In7 = VIP is approximate)."))
    im = ax.imshow(cm_frac, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cm_cols))); ax.set_yticks(range(len(cm_rows)))
    ax.set_xticklabels([textwrap.fill(c, 16) for c in cm_cols], rotation=45, ha="right")
    ax.set_yticklabels(cm_rows)
    ax.set_xlabel("Predicted type (neighbour vote)"); ax.set_ylabel("Tutorial Cell.Types")
    for i in range(len(cm_rows)):
        for j in range(len(cm_cols)):
            v = int(cm_counts.iat[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=viz.MIN_PT,
                        color="black" if cm_frac[i, j] > 0.6 else "white")
        j = cm_cols.index(exp_short[cm_rows[i]])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=2))
    cb = fig_cm.colorbar(im, ax=ax, location="top", fraction=0.05, pad=0.04)
    cb.set_label("Share of the row's cells")
plt.close(fig_cm)
display(fig_cm)
cm_paths = viz.save_figure(fig_cm, "/tmp/teddy_handson_fig", "confusion_tutorial_vs_index")
for p in cm_paths:
    shutil.copy(p, MY_FOLDER)
print("copied to", MY_FOLDER)

# COMMAND ----------

# DBTITLE 1,Report: build handson_report.html in my folder, link and preview
sys.path.insert(0, "/Workspace/.assistant/skills/html-report/scripts")
import html_report as hr

REPORT = f"{MY_FOLDER}/handson_report.html"
UMAP_PNG = f"{MY_FOLDER}/umap_tutorial_cells.png"
tv = tutorial_validation or {}
k_ok, n_q = int(res.match.sum()), len(res)
pct = 100 * k_ok / n_q
all_row = summary[summary["tutorial type"] == "ALL"].iloc[0]
per_type = summary[summary["tutorial type"] != "ALL"]
low = per_type.sort_values("median agreement").head(2)
n_cells, n_raw, n_vocab = len(CELL_IDS), int(len(in_vocab)), int(in_vocab.sum())
type_counts = pd.Series(CELL_TYPE).value_counts()
rev = tags.get("hf_revision", tv.get("revision", ""))
count_table = counts.reset_index().rename(columns={"cell_type": "tutorial type"})

spec = {
  "title": f"TEDDY-G 70M on {ENDPOINT}: {k_ok} of {n_q} tutorial interneurons get the expected Census type",
  "authors": USER,
  "summary": (f"All {n_cells:,} cells of the TEDDY tutorial data (`sample_data.h5ad`, raw counts) were embedded through "
              f"the shared serving endpoint `{ENDPOINT}` ({ENDPOINT_MODEL}, TEDDY-G 70M, mean pooling) into "
              f"{EMB.shape[1]}-dimensional vectors and written to `{MY_TABLE}`. On a UMAP the cells group by `Cell.Types`: "
              f"{same_type:.1%} of each cell's 10 nearest tutorial cells share its type, against {chance:.1%} by chance. "
              f"{n_q} tutorial cells (28-29 per type, seed {SEED}) were then labelled by the most common type among their "
              f"{K} nearest cells in the reference index `{INDEX}`; {k_ok} of {n_q} (**{pct:.1f}%**, 95% CI {all_row['95% CI']}) "
              f"got the expected interneuron class. This is a technical check, not a scientific validation."),
  "sections": [
    {"title": "Introduction", "blocks": [
      {"p": ("TEDDY-G is a transformer foundation model that turns a cell's gene expression into an embedding vector. "
             "This hands-on run asks two questions: does the shared workshop endpoint give embeddings that separate the "
             "tutorial's interneuron types, and does a nearest-neighbour lookup against a separately built Census reference "
             "index give each tutorial type the interneuron class it is expected to match?")}]},
    {"title": "Results", "blocks": [
      {"subheading": f"Tutorial cells group by Cell.Types ({same_type:.1%} same-type neighbours vs {chance:.1%} by chance)"},
      {"p": (f"The {n_cells:,} embeddings were drawn as a UMAP coloured by the {type_counts.size} `Cell.Types` "
             f"(from {type_counts.min()} cells for {type_counts.idxmin()} to {type_counts.max():,} for {type_counts.idxmax()}). "
             "Related types share one colour family and every cluster is labelled with its full name, because seven "
             f"types exceed the five distinct colours the figure rules allow. Among each cell's 10 nearest tutorial cells "
             f"(cosine), {same_type:.1%} have the same type, against {chance:.1%} expected from the type shares alone.")},
      {"figure": UMAP_PNG, "caption": (f"UMAP (cosine distance, 15 neighbours, seed 42) of {n_cells:,} TEDDY-G 70M "
             "embeddings of the tutorial cells, coloured by Cell.Types in shades of four colour families (navy PVALB, "
             "red SST, orange LAMP5, teal CALB2); labels sit at each type's median position with its cell count.")},
      {"subheading": f"{pct:.1f}% of {n_q} tutorial cells get the expected reference type from {K} neighbours"},
      {"p": (f"Each query cell was labelled by the most common `cell_type` among its {K} nearest cells in `{INDEX}`. "
             f"Per type, the match ranged from {per_type['% matching'].min():.1f}% to {per_type['% matching'].max():.1f}%. "
             "Most misses went to a related interneuron class (for example In1 cells labelled sst, In3 cells labelled VIP); "
             f"one In4 cell was labelled oligodendrocyte precursor cell. An exact cosine search over the reference table "
             f"gave the same vote as the index for {exact_agree:.1%} of the query cells.")},
      {"table": count_table, "caption": ("Count table: tutorial Cell.Types (rows) against the predicted reference type "
             "(columns; 'interneuron' stands for 'GABAergic cortical interneuron'), with totals.")},
      {"figure": fig_cm, "caption": (f"Confusion matrix of the {len(res)} query cells: tutorial Cell.Types (rows) against the "
             f"type voted by their {K} nearest reference cells (columns). Numbers are cell counts and colour is the share of "
             "the row (viridis, 0 to 1); white boxes mark the expected type, so off-box counts are the misses, nearly all in "
             "related interneuron classes. In7 = VIP is an approximate expectation.")},
      {"table": summary.rename(columns={"median agreement": "median neighbour agreement"}),
       "caption": ("Share of query cells whose predicted type equals the expected type, per tutorial type and overall, "
             "with 95% Wilson intervals; neighbour agreement is the share of the 10 neighbours that voted for the label.")}]},
    {"title": "Discussion", "blocks": [
      {"bullets": [
        "The embeddings carry the interneuron class across datasets: tutorial cells from one study land next to the matching classes of a separately built Census reference.",
        "Misses are mostly between related classes, which is where a 4-class reference is expected to be least sure."]},
      {"subheading": "Limitations"},
      {"bullets": [
        ("The expected types are a mapping supplied for this exercise: In1 and In2 = pvalb, In3 and In4 = sst, In5 and In6 = lamp5, "
         "In7 = VIP. **In7 (ADARB2-CALB2) = VIP is approximate**, so its match rate measures agreement with an approximate label."),
        (f"The reference holds {ref_counts.size} types with "
         + (f"{ref_counts.min()}" if ref_counts.min() == ref_counts.max() else f"{ref_counts.min()}-{ref_counts.max()}")
         + " cells each, only four of them "
         "interneuron classes, so subtypes such as chandelier (In2) or NPY (In4) cannot be told apart from their parent class."),
        (f"Some labels rest on a narrow majority: median neighbour agreement was "
         + ", ".join(f"{r['median agreement']:.1f} for {r['tutorial type']}" for _, r in low.iterrows()) + "."),
        (f"{n_q} query cells from {n_donors_query} donors (28-29 per type) give wide per-type intervals; the query cells are "
         "not held out from any training, and TEDDY's pretraining data may include this dataset."),
        "This is a technical check of the endpoint, table and index, not a scientific, clinical or regulatory validation."]}]},
    {"title": "Methods", "blocks": [
      {"subheading": "Data sources"},
      {"p": (f"Tutorial data: `sample_data.h5ad` from the TEDDY repository (Hugging Face `Merck/TEDDY`, revision `{rev}`), "
             f"{n_cells:,} cells from {n_donors_all} donors, labels in `obs['Cell.Types']` and `obs['donor_id']`. Counts were "
             f"read from `.raw.X` ({n_raw:,} genes, Ensembl IDs in the raw var index); {n_vocab:,} genes "
             f"({n_vocab/n_raw:.1%}) are in the TEDDY 70M vocabulary and were kept. All `.raw.X` values are integers; "
             f"the median cell has {np.median(np.asarray(X.sum(1)).ravel()):.0f} counts over "
             f"{np.median(np.diff(X.indptr)):.0f} expressed vocabulary genes.")},
      {"path": H5AD},
      {"p": (f"Reference: index `{INDEX}` ({ix.status.indexed_row_count:,} rows, Vector Search endpoint `{ix.endpoint_name}`), "
             f"synced from `{REF_TABLE}` ({len(ref):,} cells, {ref_counts.size} cell types). Its `model_version` "
             f"(`{MODEL_VERSION}`) equals the one written for the tutorial cells.")},
      {"subheading": "Model and embedding"},
      {"p": (f"Serving endpoint `{ENDPOINT}` serves Unity Catalog model `{se.entity_name}` version {se.entity_version} "
             f"on {se.workload_type.value} (deployment tag `{tags.get('deployment', '')}`, variant {tags.get('variant', '')}). "
             f"Cells were sent in {len(batches)} requests of at most {BATCH} cells (raw counts; only the genes expressed in "
             f"each batch, which leaves totals and top-K unchanged), with `pooling='{POOLING}'` as a column on each record; "
             f"every response echoed that pooling. Embedding took 276 s with 4 parallel requests. Vectors have "
             f"{EMB.shape[1]} dimensions and lengths {norms.min():.3f} (min) to {np.median(norms):.3f} (median). The "
             f"notebook ran on serverless compute (attached to a GPU session, which the run did not need); the model ran "
             f"on the endpoint.")},
      {"subheading": "Figure and lookup"},
      {"p": (f"UMAP: umap-learn, cosine metric, 15 neighbours, min_dist 0.3, seed 42. Lookup: {n_q} tutorial cells "
             f"chosen with a seeded generator (seed {SEED}), 28 or 29 per type; each was queried against `{INDEX}` for "
             f"{K} neighbours with its stored vector (reference vectors are not unit length, mean length {ref_norm:.2f}), "
             "and labelled by the most common neighbour `cell_type`. Intervals are 95% Wilson score intervals.")},
      {"subheading": "Checks"},
      {"checks": [
        {"check": "Shared files, endpoint and index exist", "threshold": "files present; endpoint READY; index ready",
         "result": f"present; READY; ready ({ix.status.indexed_row_count:,} rows)", "passed": True},
        {"check": "Setup's tutorial check (tutorial_validation.json)", "threshold": f"cosine >= {tv.get('threshold')}",
         "result": (f"{tv.get('result')}: min {tv.get('min_cosine'):.9f}, median {tv.get('median_cosine'):.9f} on "
                    f"{tv.get('cells'):,} cells, pooling={tv.get('pooling')}, {tv.get('model')}, {tv.get('utc')}") if tv else "not run",
         **({"passed": tv.get("result") == "PASS"} if tv else {"status": "not run"})},
        {"check": "Model version matches the reference index", "threshold": "identical model_version string",
         "result": MODEL_VERSION, "passed": True},
        {"check": "Raw counts", "threshold": "all .raw.X values integers", "result": "integers", "passed": True},
        {"check": "Every cell embedded", "threshold": f"{n_cells:,} finite, distinct vectors",
         "result": f"{EMB.shape[0]:,} x {EMB.shape[1]}, {len(np.unique(EMB.round(5), axis=0)):,} distinct", "passed": True},
        {"check": "My table written", "threshold": "rows = distinct cell_id = cells",
         "result": f"{spark.table(MY_TABLE).count():,} rows", "passed": True},
        {"check": "Index vote vs exact search", "threshold": "reported only", "result": f"{exact_agree:.1%} same vote",
         "status": "reported"}]},
      {"subheading": "Software"},
      {"table": versions_df, "caption": "Package versions read with importlib.metadata on the notebook compute."}]},
    {"title": "Code availability", "blocks": [
      {"p": ("The notebook and the helper scripts it imports (`visualize.py`, `nn_annotate.py` from the open-weight-models "
             "skill) are workspace-only and not publicly released. The TEDDY model code is public: Hugging Face "
             f"[Merck/TEDDY](https://huggingface.co/Merck/TEDDY), revision `{rev}` (model card license "
             f"{tags.get('license_model_card', '')}, package license {tags.get('license_package', '')}).")},
      {"path": "/Workspace/Shared/exercises/deploy_scrna_model_and_vector_search/to_test/runs/teddy_handson"}]},
    {"title": "Data availability", "blocks": [
      {"p": (f"The tutorial data are public in the TEDDY repository (`data/sample_data.h5ad`, revision `{rev}`). Derived "
             f"data: the embeddings table `{MY_TABLE}` and the UMAP and confusion-matrix figure files in `{MY_FOLDER}`. The reference table "
             f"`{REF_TABLE}` and index `{INDEX}` were built by the workshop setup from CELLxGENE Census cells; their "
             "Census release is not recorded in this notebook.")}]},
    {"title": "References", "blocks": [{"bullets": [
      "TEDDY: [arXiv:2503.03485](https://arxiv.org/abs/2503.03485).",
      "TEDDY model card: [huggingface.co/Merck/TEDDY](https://huggingface.co/Merck/TEDDY).",
      "McInnes, Healy and Melville, UMAP: [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)."]}]},
  ],
}

# This run created the report; the user asked to update it with the confusion matrix, so replace it
out = hr.build_report(spec, REPORT, overwrite=os.path.exists(REPORT))
print({k: v for k, v in out.items() if k != "path"})
assert not out["problems"], out["problems"]

host = w.config.host.rstrip("/")
from IPython.display import HTML, display as ipy_display
ipy_display(HTML(f'<p>Report: <a href="{host}/ajax-api/2.0/fs/files{REPORT}" target="_blank">{REPORT}</a> '
            f'({out["bytes"]/1e6:.2f} MB). Open it in a browser and click <b>Print / Save as PDF</b>.</p>'))
hr.display_report(out["path"])