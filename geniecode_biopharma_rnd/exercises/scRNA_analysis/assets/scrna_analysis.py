# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # scRNA-seq Analysis: 7970bd6b-f752-47a9-8643-2af16855ec49
# MAGIC
# MAGIC Single-sample exploratory analysis of `/Volumes/dbacademy/scrna/raw_h5ad/7970bd6b-f752-47a9-8643-2af16855ec49.h5ad` (~177 MB on disk, serverless is sufficient).
# MAGIC
# MAGIC Workflow: profile (backed read) -> QC metrics -> **data-driven QC thresholds (user confirmation required)** -> normalize / HVG / PCA / UMAP / Leiden -> marker genes -> **LLM-suggested cell types (expert review required)**. Parameters, decisions and plots are tracked in the notebook-scoped MLflow experiment.

# COMMAND ----------

# DBTITLE 1,Install packages
# MAGIC %pip install -q scanpy[leiden] anndata harmonypy harmonypy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup and MLflow parent run
# MAGIC %matplotlib inline
# MAGIC import os, tempfile, time, json
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import anndata as ad
# MAGIC import scanpy as sc
# MAGIC import matplotlib.pyplot as plt
# MAGIC import mlflow
# MAGIC import harmonypy as hm
# MAGIC
# MAGIC sc.set_figure_params(dpi=100, frameon=False)
# MAGIC tmpdir = tempfile.TemporaryDirectory()
# MAGIC
# MAGIC H5AD_PATH = "/Volumes/dbacademy/scrna/raw_h5ad/7970bd6b-f752-47a9-8643-2af16855ec49.h5ad"
# MAGIC SAMPLE_ID = "7970bd6b"
# MAGIC
# MAGIC # MLflow: notebook-scoped experiment (default), parent run for the whole analysis
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC if mlflow.active_run():
# MAGIC     mlflow.end_run()
# MAGIC parent_run = mlflow.start_run(run_name=f"analysis_{SAMPLE_ID}")
# MAGIC mlflow.set_tags({"origin": "agent", "dataset": H5AD_PATH, "pipeline": "scanpy_standard"})
# MAGIC t0 = time.time()
# MAGIC print("Parent run:", parent_run.info.run_id)

# COMMAND ----------

# DBTITLE 1,Profile file (backed read)
# Inspect metadata without loading the expression matrix
bk = ad.read_h5ad(H5AD_PATH, backed="r")
print(f"Shape: {bk.shape[0]:,} cells x {bk.shape[1]:,} genes")
print(f".raw present: {bk.raw is not None}")
print(f"Layers: {list(bk.layers.keys())}")
print(f"obsm keys: {list(bk.obsm.keys())}")
print(f"var columns: {list(bk.var.columns)}")
print(f"First var_names: {bk.var_names[:5].tolist()}")
print(f"\nobs columns ({len(bk.obs.columns)}):")
for col in bk.obs.columns:
    vals = bk.obs[col].dropna().unique()
    print(f"  {col}: {len(vals)} unique - {list(vals[:5])}")
for key in ["tissue", "organism", "assay", "disease", "cell_type", "donor_id"]:
    if key in bk.obs.columns:
        mlflow.set_tag(key, ", ".join(map(str, bk.obs[key].unique()[:5]))[:500])
if "cell_type" in bk.obs.columns:
    print(bk.obs["cell_type"].value_counts())
for key in ["Sample", "donor_id"]:
    if key in bk.obs.columns and "assay" in bk.obs.columns:
        print(pd.crosstab(bk.obs[key], bk.obs["assay"]))
bk.file.close()

# COMMAND ----------

# DBTITLE 1,Full load and gene naming
adata = sc.read_h5ad(H5AD_PATH)
adata.obs_names_make_unique()

# Prefer raw counts if available (CELLxGENE files often store normalized X + raw counts)
if adata.raw is not None:
    print("Using adata.raw as count matrix")
    raw = adata.raw.to_adata()
    raw.obs = adata.obs.copy()
    adata = raw

# Swap Ensembl IDs to gene symbols so MT-/RPS/RPL matching works
if str(adata.var_names[0]).startswith(("ENSG", "ENSMUSG")):
    for col in ["feature_name", "gene_symbols", "gene_name", "symbol"]:
        if col in adata.var.columns:
            adata.var["ensembl_id"] = adata.var_names.copy()
            adata.var_names = adata.var[col].astype(str).values
            adata.var_names_make_unique()
            print(f"Swapped var_names from Ensembl IDs to '{col}'")
            break
    else:
        print("WARNING: Ensembl IDs with no symbol column - MT/ribo QC will be zero")

# Sanity check: counts should be integer-like
sample_vals = adata.X[:200].toarray() if hasattr(adata.X, "toarray") else adata.X[:200]
print("Integer-like counts:", np.allclose(sample_vals, np.round(sample_vals)))
print(adata)

# COMMAND ----------

# DBTITLE 1,QC metrics and data-driven thresholds
adata.var["mt"]   = adata.var_names.str.upper().str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
adata.var["hb"]   = adata.var_names.str.upper().str.contains(r"^HB[^(P)]")
print("n mt/ribo/hb genes:", adata.var.mt.sum(), adata.var.ribo.sum(), adata.var.hb.sum())
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

# This CELLxGENE file has MT- genes removed from the matrix, so pct_counts_mt would be all zero.
# Fall back to the author-computed fraction (obs['percent.mito'], 0-1 scale) converted to %.
if adata.var["mt"].sum() == 0 and "percent.mito" in adata.obs:
    adata.obs["pct_counts_mt"] = adata.obs["percent.mito"].astype(float) * 100
    print("MT genes absent from matrix -> using author percent.mito x 100")

def mad_bounds(s, n_mads=3):
    med = s.median(); mad = np.median(np.abs(s - med))
    return med - n_mads * mad, med + n_mads * mad

obs = adata.obs
qc_profile = {}
for m in ["pct_counts_mt", "n_genes_by_counts", "total_counts", "pct_counts_ribo", "pct_counts_hb"]:
    lo, hi = mad_bounds(obs[m])
    qc_profile[m] = {"median": obs[m].median(), "P1": obs[m].quantile(0.01), "P5": obs[m].quantile(0.05),
                     "P95": obs[m].quantile(0.95), "P99": obs[m].quantile(0.99), "MAD_low": lo, "MAD_high": hi,
                     "min": obs[m].min(), "max": obs[m].max()}
qc_profile_df = pd.DataFrame(qc_profile).T.round(2)
display(qc_profile_df)
print(f"n_cells_raw = {adata.n_obs:,}, n_genes_raw = {adata.n_vars:,}")
if "assay" in obs:
    display(obs.groupby("assay", observed=True)[["pct_counts_mt", "n_genes_by_counts", "total_counts"]].median().round(1))
if "DoubletFinderPrediction" in obs:
    print("Author doublet flags (DoubletFinder non-singlet or Scrublet True):",
          int(((obs.DoubletFinderPrediction != "Singlet") | (obs.ScrubletPrediction == "True")).sum()))
for mc in [3, 5, 10]:
    print(f"genes detected in >= {mc} cells:", int(((adata.X > 0).sum(0) >= mc).sum()))

sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"], jitter=0.3, multi_panel=True, show=False)
plt.savefig(os.path.join(tmpdir.name, "qc_violin_pre.png"), bbox_inches="tight"); plt.show()
sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", show=False)
plt.savefig(os.path.join(tmpdir.name, "qc_scatter_pre.png"), bbox_inches="tight"); plt.show()

# COMMAND ----------

# DBTITLE 1,QC threshold decision (Gate G2)
# MAGIC %md
# MAGIC ## QC thresholds (Gate G2 - confirmed by user)
# MAGIC
# MAGIC * `pct_counts_mt < 10%`: MAD upper bound 9.79%, P95 10.42%. MT- genes are absent from the matrix, so the author's `percent.mito` (x100) is used.
# MAGIC * `600 <= n_genes_by_counts < 2,500`: MAD lower 608 / P1 611; MAD upper 2,420 / P95 2,631. The authors had already filtered at 501 genes.
# MAGIC * Author doublet calls removed (DoubletFinder non-singlet or Scrublet True, 1,009 cells). Genes kept if detected in >= 3 cells.
# MAGIC * Plasmablasts are not disproportionately removed by these cutoffs (3.6% vs 9.4% of memory B cells above the gene cap).

# COMMAND ----------

# DBTITLE 1,Apply QC filters
# Single source of truth: the same dict drives filtering and MLflow logging
qc_params = {
    "max_pct_counts_mt": 10.0,
    "min_genes": 600,
    "max_genes": 2500,
    "remove_author_doublets": True,
    "min_cells": 3,
    "mt_source": "author_percent.mito_x100",
}

n_cells_pre, n_genes_pre = adata.n_obs, adata.n_vars
o = adata.obs
flags = pd.DataFrame({
    "high_mt": o["pct_counts_mt"] >= qc_params["max_pct_counts_mt"],
    "low_genes": o["n_genes_by_counts"] < qc_params["min_genes"],
    "high_genes": o["n_genes_by_counts"] >= qc_params["max_genes"],
    "author_doublet": ((o["DoubletFinderPrediction"] != "Singlet") | (o["ScrubletPrediction"] == "True"))
                      if qc_params["remove_author_doublets"] else False,
}, index=o.index)
keep = ~flags.any(axis=1)
display(flags.sum().rename("cells_flagged").to_frame())

adata = adata[keep.values].copy()
sc.pp.filter_genes(adata, min_cells=qc_params["min_cells"])

retention = (adata.obs["donor_id"].value_counts() / o["donor_id"].value_counts() * 100).round(1)
display(pd.DataFrame({"cells_post_qc": adata.obs["donor_id"].value_counts(), "retention_pct": retention}))
print(f"Cells: {n_cells_pre:,} -> {adata.n_obs:,} ({adata.n_obs / n_cells_pre:.1%}); genes: {n_genes_pre:,} -> {adata.n_vars:,}")

with mlflow.start_run(run_name="gate_g2_qc", nested=True):
    step_dir = os.path.join(tmpdir.name, "gate_g2_qc"); os.makedirs(step_dir, exist_ok=True)
    mlflow.log_params(qc_params)
    mlflow.set_tag("param_source", "data_driven_user_confirmed")
    mlflow.log_metrics({"n_cells_pre_qc": n_cells_pre, "n_cells_post_qc": adata.n_obs,
                        "n_genes_post_qc": adata.n_vars, "filter_retention_pct": 100 * adata.n_obs / n_cells_pre})
    qc_profile_df.to_csv(os.path.join(step_dir, "qc_profile.csv"))
    for f in ["qc_violin_pre.png", "qc_scatter_pre.png"]:
        if os.path.exists(os.path.join(tmpdir.name, f)):
            mlflow.log_artifact(os.path.join(tmpdir.name, f))
    mlflow.log_artifacts(step_dir)

# COMMAND ----------

# DBTITLE 1,Batch key decision (Gate G1)
# MAGIC %md
# MAGIC ## Batch correction (Gate G1 - confirmed by user)
# MAGIC
# MAGIC The file contains 7 donors / 13 libraries from one tonsil study (same tissue, same lineage), so integration is biologically appropriate. Harmony is run on `donor_id` (user confirmed). Assay is nested within donor (only BCP2, 197 cells, is 10x 3' v2), so correcting on donor also absorbs the assay effect. HVGs are selected per donor to avoid donor-specific genes dominating.

# COMMAND ----------

# DBTITLE 1,Normalize, HVG, PCA, Harmony
import harmonypy as hm

analysis_params = {
    "target_sum": 10000,
    "n_top_genes": 2000,
    "hvg_batch_key": "donor_id",
    "n_comps": 50,
    "n_pcs": 30,
    "batch_key": "donor_id",
    "integration_method": "harmony",
    "cluster_resolution": 0.8,
    "random_state": 0,
}

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=analysis_params["target_sum"])
sc.pp.log1p(adata)
adata.raw = adata  # log-normalized, all genes (used for marker tests / plots)

sc.pp.highly_variable_genes(adata, n_top_genes=analysis_params["n_top_genes"],
                            batch_key=analysis_params["hvg_batch_key"])
print("HVGs:", int(adata.var["highly_variable"].sum()))

sc.tl.pca(adata, n_comps=analysis_params["n_comps"], mask_var="highly_variable",
          random_state=analysis_params["random_state"])

ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, vars_use=[analysis_params["batch_key"]],
                    random_state=analysis_params["random_state"])
Z = np.asarray(ho.Z_corr)
if Z.shape[0] != adata.n_obs:  # harmonypy 1.x returns (n_pcs, n_cells)
    Z = Z.T
adata.obsm["X_pca_harmony"] = Z
print("X_pca_harmony:", Z.shape)

with mlflow.start_run(run_name="gate_g1_batch", nested=True):
    mlflow.log_params(analysis_params)
    mlflow.set_tag("param_source", "user_confirmed")
    mlflow.log_metric("n_hvg", int(adata.var["highly_variable"].sum()))

# COMMAND ----------

# DBTITLE 1,Neighbors, UMAP, Leiden
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_pcs=analysis_params["n_pcs"],
                random_state=analysis_params["random_state"])
sc.tl.umap(adata, random_state=analysis_params["random_state"])
sc.tl.leiden(adata, resolution=analysis_params["cluster_resolution"], flavor="igraph",
             n_iterations=2, directed=False, random_state=analysis_params["random_state"])
n_clusters = adata.obs["leiden"].nunique()
print("n_clusters:", n_clusters)

# Compare new clusters with the author annotations and check donor mixing
ct = pd.crosstab(adata.obs["leiden"], adata.obs["author_cell_type"])
cluster_vs_author = (ct.div(ct.sum(1), axis=0) * 100).round(0).astype(int)
cluster_vs_author.insert(0, "n_cells", ct.sum(1))
display(cluster_vs_author)
display(pd.crosstab(adata.obs["leiden"], adata.obs["cell_type"]))
display(pd.crosstab(adata.obs["leiden"], adata.obs["donor_id"], normalize="index").round(2))

with mlflow.start_run(run_name="iter_1_clustering", nested=True):
    step_dir = os.path.join(tmpdir.name, "iter_1_clustering"); os.makedirs(step_dir, exist_ok=True)
    mlflow.log_params({"cluster_resolution": analysis_params["cluster_resolution"]})
    mlflow.log_metric("n_clusters", n_clusters)
    cluster_vs_author.to_csv(os.path.join(step_dir, "cluster_vs_author_pct.csv"))
    sc.pl.umap(adata, color=["leiden", "author_cell_type", "cell_type", "donor_id"], ncols=2,
               legend_loc="right margin", wspace=0.5, show=False)
    plt.savefig(os.path.join(step_dir, "umap_overview.png"), bbox_inches="tight"); plt.show()
    mlflow.log_artifacts(step_dir)

# COMMAND ----------

# DBTITLE 1,Marker genes per cluster
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True)
n_markers = 10
marker_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(n_markers)
display(marker_df.T.rename_axis("cluster"))

# Canonical B-cell / plasma-cell state markers to support annotation
canonical = ["MS4A1", "CD19", "CR2", "FCER2", "IGHD", "IGHM", "CD27", "CCR7", "CD69", "CD83", "NR4A1",
             "MYC", "FCRL4", "FCRL5", "ITGAX", "TBX21", "FCRL2", "FCRL3", "HSPA1A", "DNAJB1", "IFI44L",
             "ISG15", "MKI67", "PRDM1", "XBP1", "JCHAIN", "MZB1", "CD38"]
canonical = [g for g in canonical if g in adata.raw.var_names]

with mlflow.start_run(run_name="markers", nested=True):
    step_dir = os.path.join(tmpdir.name, "markers"); os.makedirs(step_dir, exist_ok=True)
    marker_df.to_csv(os.path.join(step_dir, "top_markers_per_cluster.csv"), index=False)
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, show=False)
    plt.savefig(os.path.join(step_dir, "marker_dotplot.png"), bbox_inches="tight"); plt.show()
    sc.pl.dotplot(adata, canonical, groupby="leiden", standard_scale="var", show=False)
    plt.savefig(os.path.join(step_dir, "canonical_dotplot.png"), bbox_inches="tight"); plt.show()
    mlflow.log_artifacts(step_dir)

# Mean expression of canonical markers per cluster (for annotation review)
expr = sc.get.obs_df(adata, keys=canonical + ["leiden"], use_raw=True)
display(expr.groupby("leiden", observed=True).mean().round(2))

# COMMAND ----------

# DBTITLE 1,Cell type annotation (Gate G4 pending)
# MAGIC %md
# MAGIC ## Cell Type Annotation (LLM-Suggested - Expert Review Required)
# MAGIC
# MAGIC Leiden (resolution 0.8) gives 10 clusters, all B-lineage (MS4A1+, CD19+). Assistant-suggested state labels from the top markers and overlap with `author_cell_type` are pending expert review. The annotation cell will be added after the labels are confirmed or corrected. **Note:** clusters 1, 6 and 9 are ~50% donor BCP6 (12% of cells overall) and cluster 7 has RPS4Y1 among its top markers, so some residual donor effect may remain.

# COMMAND ----------

# DBTITLE 1,Suggested cell type annotations (pending review)
# LLM-suggested cell type annotations - REQUIRES EXPERT REVIEW (not yet written to adata.obs)
suggested_annotations = pd.DataFrame([
    ("0", "KLF2, TXNIP, GPR183, CXCR4, TSC22D3", "Resting memory B (quiescent)", "Medium"),
    ("1", "ID3, VPREB3, CR2, CR1, CD180", "Resting CD21-high memory B", "Medium"),
    ("2", "CD83, MIR155HG, HLA-DQA1, NCL, CCR7, FCER2", "Pre-GC / activated memory B", "Medium"),
    ("3", "JUN, FOS, FOSB, EGR1, CD69, NFKBIA", "Activated memory B (immediate-early)", "Medium"),
    ("4", "HSPA1A, DNAJB1, HSP90AA1, HSPH1", "Heat-shock / stress-response memory B", "Medium"),
    ("5", "CXCR4, RGS1, JUNB, CD69, CD83, IGHM-high", "Activated memory B (IgM-high)", "Medium"),
    ("6", "FCRL4, FCRL5, ITGAX, low CR2", "FCRL4+ atypical memory B", "High"),
    ("7", "HSPA1A/B, DNAJB1, HSPB1, RPS4Y1", "Heat-shock / stress-response memory B", "Low-Medium"),
    ("8", "FCRL5, FCRL2/3, CD22, MS4A1, IKZF3", "FCRL5-high / CD21-low memory B", "Low"),
    ("9", "low CR2, low CXCR4, housekeeping/ribosomal", "CD21-low memory B (possibly low quality)", "Low"),
], columns=["cluster", "key_markers", "suggested_type", "confidence"])

# Add live context from the current clustering: cluster size and dominant author label
sizes = adata.obs["leiden"].value_counts()
ct = pd.crosstab(adata.obs["leiden"], adata.obs["author_cell_type"])
top_author = ct.idxmax(axis=1)
top_pct = (ct.max(axis=1) / ct.sum(axis=1) * 100).round(0).astype(int)
suggested_annotations["n_cells"] = suggested_annotations["cluster"].map(sizes)
suggested_annotations["top_author_label"] = suggested_annotations["cluster"].map(
    lambda c: f"{top_author[c]} ({top_pct[c]}%)")
suggested_annotations["status"] = "llm_suggested_pending_review"

assert set(suggested_annotations["cluster"]) == set(adata.obs["leiden"].cat.categories), \
    "Cluster IDs changed - re-derive suggestions before using this table"

print("WARNING: LLM-suggested annotations - approximate, must be verified by a domain expert.")
print("Tissue context: tonsil, B-cell lineage (all clusters MS4A1+/CD19+).\n")
print(suggested_annotations[["cluster", "n_cells", "key_markers", "suggested_type", "confidence",
                             "top_author_label"]].to_string(index=False))
display(suggested_annotations)

# COMMAND ----------

# DBTITLE 1,TEDDY disease annotation overview
# MAGIC %md
# MAGIC ## Disease annotation by TEDDY reference KNN (exploratory)
# MAGIC
# MAGIC A small stratified sample of cells per Leiden cluster is embedded with the Genesis Workbench **TEDDY** serving endpoint (1024-d), and each embedding is queried against `main.genesis_workbench.teddy_cell_index` (~2M reference cells). Each query cell gets the majority `disease` (and `cell_type` / `tissue`) among its 25 nearest reference neighbors.
# MAGIC
# MAGIC **Scope:** this is reference KNN, not a diagnosis. Labels can only be as good as the reference covers this tissue / cell state, and the known ground truth for this dataset is `disease = normal` (healthy pediatric tonsil), which serves as a sanity check.
# MAGIC
# MAGIC **Status:** embedding and KNN retrieval work (200 cells, 20 per cluster, 25 neighbors each). The index exposes only `cell_id`, and the label table `main.genesis_workbench.teddy_cells` needs `SELECT` that this user does not have, so the disease vote cell stops with a clear message until access is granted. `main.owm.teddy_workshop_cells` was checked as a fallback: it has no `disease` column, uses 512-d embeddings and shares no cell_ids with the index.

# COMMAND ----------

# DBTITLE 1,TEDDY setup and reference index
from collections import Counter
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

w = WorkspaceClient(config=Config(http_timeout_seconds=900))

# Resolve the TEDDY endpoint by slug at runtime (GWB reaps idle endpoints - fail loud)
all_eps = [e.name for e in w.serving_endpoints.list()]
teddy_cands = [n for n in all_eps if "teddy" in n]
if not teddy_cands:
    raise RuntimeError(f"No TEDDY serving endpoint found. Available: {sorted(all_eps)}")
TEDDY_EP = teddy_cands[0]
TEDDY_INDEX = "main.genesis_workbench.teddy_cell_index"
REF_TABLE = "main.genesis_workbench.teddy_cells"
print("TEDDY endpoint:", TEDDY_EP, "| state:", w.serving_endpoints.get(TEDDY_EP).state.ready)

# The source table is not readable by this user (no SELECT on main.genesis_workbench.teddy_cells),
# so reference metadata is read through the Vector Search index instead.
idx = w.vector_search_indexes.get_index(TEDDY_INDEX)
print("Index:", idx.name, "| ready:", idx.status.ready, "| rows:", idx.status.indexed_row_count,
      "| dim:", idx.delta_sync_index_spec.embedding_vector_columns[0].embedding_dimension)
REF_COLS = ["cell_type", "tissue", "disease"]

# COMMAND ----------

# DBTITLE 1,Stratified sample and TEDDY payload probe
teddy_params = {"cells_per_cluster": 20, "knn_k": 25, "chunk_size": 25, "random_state": 0,
                "expression_input": "raw_counts", "gene_ids": "hgnc_symbol"}

# Stratified sample: same number of cells from every Leiden cluster (or all cells if fewer)
sample_idx = (adata.obs.groupby("leiden", observed=True, group_keys=False)
              .apply(lambda d: d.sample(n=min(len(d), teddy_params["cells_per_cluster"]),
                                        random_state=teddy_params["random_state"]))
              .index)
qsub = adata[sample_idx].copy()
print("Sampled cells per cluster:", qsub.obs["leiden"].value_counts().sort_index().to_dict())

# TEDDY ranks genes internally -> send raw counts with HGNC symbols as the var index
Xq = qsub.layers["counts"]
Xq = Xq.toarray() if hasattr(Xq, "toarray") else np.asarray(Xq)
var_q = pd.DataFrame(index=qsub.var_names.astype(str))
var_q["ensembl_id"] = qsub.var["ensembl_id"].astype(str).values
obs_q = pd.DataFrame(index=qsub.obs_names.astype(str))
obs_q["n_counts"] = qsub.obs["total_counts"].astype(float).values

def teddy_embed(rows, obs_df, var_df):
    payload = [{"adata_sparsematrix": np.asarray(rows, dtype=float).tolist(),
                "adata_obs": obs_df.to_json(orient="split"),
                "adata_var": var_df.to_json(orient="split")}]
    preds = w.serving_endpoints.query(name=TEDDY_EP, inputs=payload).predictions
    if isinstance(preds, dict) and "predictions" in preds:
        preds = preds["predictions"]
    return [np.asarray(p["embedding"] if isinstance(p, dict) else p, dtype=float) for p in preds]

# Probe on 2 cells to confirm the contract before the full sample
probe = teddy_embed(Xq[:2], obs_q.iloc[:2], var_q)
print("Probe: n_embeddings =", len(probe), "| dim =", probe[0].shape, "| finite:", bool(np.isfinite(probe[0]).all()))

# COMMAND ----------

# DBTITLE 1,Gene vocabulary check (symbols vs Ensembl)
# Sanity check the gene vocabulary: a mismatched vocab gives near-identical embeddings for different cells.
# Compare HGNC symbols vs Ensembl IDs on 4 cells from 4 different clusters.
probe_rows = [np.flatnonzero(qsub.obs["leiden"].values == c)[0] for c in ["0", "3", "6", "9"]]
var_ens = pd.DataFrame(index=qsub.var["ensembl_id"].astype(str).values)

def mean_offdiag_cos(embs):
    E = np.vstack(embs); E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    return float(S[~np.eye(len(E), dtype=bool)].mean())

vocab_check = {}
for name, vdf in [("hgnc_symbol", var_q), ("ensembl_id", var_ens)]:
    e = teddy_embed(Xq[probe_rows], obs_q.iloc[probe_rows], vdf)
    vocab_check[name] = {"mean_cos_between_cells": round(mean_offdiag_cos(e), 4), "emb_std": round(float(np.std(e)), 4)}
vocab_check_df = pd.DataFrame(vocab_check).T
display(vocab_check_df)

# COMMAND ----------

# DBTITLE 1,Gene vocabulary decision
# MAGIC %md
# MAGIC **Gene vocabulary decision:** with HGNC symbols, TEDDY returned identical embeddings for cells from four different clusters (mean pairwise cosine = 1.0), meaning the genes were not recognized. With Ensembl IDs, the mean pairwise cosine dropped to 0.28. TEDDY therefore expects **Ensembl IDs**, and all embeddings below use `var.ensembl_id`.

# COMMAND ----------

# DBTITLE 1,Check reference metadata availability
# The TEDDY index only exposes cell_id + embedding; neighbor labels must come from a metadata table.
# The canonical source (main.genesis_workbench.teddy_cells) is not readable, so check the accessible
# workshop table and whether its cell_ids overlap with TEDDY KNN neighbors.
from pyspark.sql import functions as F

wk = spark.table("main.owm.teddy_workshop_cells")
print("teddy_workshop_cells columns:", wk.columns)
display(wk.groupBy("model_version").agg(F.count("*").alias("n"), F.countDistinct("cell_type").alias("n_types"),
                                        F.max(F.size("embedding")).alias("dim")))

r = w.vector_search_indexes.query_index(index_name=TEDDY_INDEX, columns=["cell_id"],
                                        query_vector=probe[0].tolist(), num_results=25)
neighbor_ids = [str(x[0]) for x in (r.result.data_array or [])]
print("Probe neighbors:", neighbor_ids[:10])
overlap = wk.filter(F.col("cell_id").cast("string").isin(neighbor_ids)).count()
print(f"Neighbor cell_ids found in teddy_workshop_cells: {overlap} / {len(neighbor_ids)}")

# COMMAND ----------

# DBTITLE 1,TEDDY embed and KNN neighbors
teddy_params["gene_ids"] = "ensembl_id"  # from the vocabulary check above

# 1) Embed the stratified sample in chunks
embs = []
for s in range(0, qsub.n_obs, teddy_params["chunk_size"]):
    sl = slice(s, s + teddy_params["chunk_size"])
    embs.extend(teddy_embed(Xq[sl], obs_q.iloc[sl], var_ens))
E = np.vstack(embs)
assert E.shape == (qsub.n_obs, 1024) and np.isfinite(E).all()

# 2) KNN against the TEDDY reference index. The index only exposes cell_id (+ embedding), so we
#    keep neighbor IDs and similarity scores; labels are joined from the reference metadata afterwards.
nbr_rows = []
for qid, e in zip(qsub.obs_names, E):
    r = w.vector_search_indexes.query_index(index_name=TEDDY_INDEX, columns=["cell_id"],
                                            query_vector=e.tolist(), num_results=teddy_params["knn_k"])
    for rank, x in enumerate(r.result.data_array or []):
        nbr_rows.append({"query_cell": qid, "rank": rank, "ref_cell_id": str(x[0]), "score": float(x[-1])})
neighbors = pd.DataFrame(nbr_rows)
neighbors["leiden"] = neighbors["query_cell"].map(qsub.obs["leiden"])

per_cell = neighbors.groupby("query_cell").agg(n_neighbors=("ref_cell_id", "size"), mean_score=("score", "mean"))
print(f"Query cells: {qsub.n_obs} | with neighbors: {len(per_cell)} | neighbor rows: {len(neighbors):,} "
      f"| unique reference cells: {neighbors.ref_cell_id.nunique():,}")
display(neighbors.groupby("leiden", observed=True)
        .agg(n_query=("query_cell", "nunique"), mean_score=("score", "mean"),
             unique_ref_cells=("ref_cell_id", "nunique")).round(3))

# COMMAND ----------

# DBTITLE 1,Join reference labels and per-cluster disease summary
# Join neighbor IDs to reference metadata (requires SELECT on the TEDDY source table)
from pyspark.errors import AnalysisException
try:
    ref_meta = (spark.table(REF_TABLE)
                .select(F.col("cell_id").cast("string").alias("ref_cell_id"), *REF_COLS)
                .join(spark.createDataFrame(neighbors[["ref_cell_id"]].drop_duplicates()), "ref_cell_id")
                .toPandas())
except AnalysisException as err:
    if "INSUFFICIENT_PERMISSIONS" not in str(err):
        raise
    neighbors.to_csv(os.path.join(tmpdir.name, "teddy_neighbors.csv"), index=False)
    raise RuntimeError(
        f"No SELECT on {REF_TABLE}: the TEDDY index has no label columns, so disease cannot be voted. "
        f"Ask the table owner to run: GRANT SELECT ON TABLE {REF_TABLE} TO `<your user>`; then re-run this cell. "
        f"Neighbor IDs for {neighbors.query_cell.nunique()} cells are kept in `neighbors`.") from None

lab = neighbors.merge(ref_meta, on="ref_cell_id", how="left")
votes = []
for qid, d in lab.groupby("query_cell"):
    row = {"query_cell": qid, "mean_score": d["score"].mean()}
    for col in REF_COLS:
        vals = d[col].dropna().astype(str)
        vc = vals.value_counts()
        row[f"teddy_{col}"] = vc.index[0] if len(vc) else "unknown"
        row[f"teddy_{col}_frac"] = vc.iloc[0] / len(vals) if len(vc) else 0.0
    votes.append(row)
teddy_res = pd.concat([qsub.obs[["leiden", "donor_id", "author_cell_type", "cell_type", "disease"]],
                       pd.DataFrame(votes).set_index("query_cell")], axis=1)

# Per-cluster summary of predicted disease (and context labels)
def top_share(s):
    vc = s.value_counts(normalize=True)
    return f"{vc.index[0]} ({vc.iloc[0]:.0%})"

cluster_summary = (teddy_res.groupby("leiden", observed=True)
    .agg(n_sampled=("disease", "size"),
         predicted_disease=("teddy_disease", top_share),
         mean_disease_vote=("teddy_disease_frac", "mean"),
         predicted_cell_type=("teddy_cell_type", top_share),
         predicted_tissue=("teddy_tissue", top_share),
         mean_knn_score=("mean_score", "mean"),
         known_disease=("disease", top_share))
    .round(3))
cluster_summary.insert(1, "suggested_type", suggested_annotations.set_index("cluster")["suggested_type"])
display(cluster_summary)

# Overall predicted-disease distribution vs ground truth (this dataset is 'normal')
display(pd.crosstab(teddy_res["teddy_disease"], teddy_res["leiden"], margins=True))
agree = (teddy_res["teddy_disease"] == teddy_res["disease"].astype(str)).mean()
print(f"Agreement with known disease label ('normal'): {agree:.1%} of {len(teddy_res)} sampled cells")

with mlflow.start_run(run_name="teddy_disease_knn", nested=True):
    step_dir = os.path.join(tmpdir.name, "teddy_disease_knn"); os.makedirs(step_dir, exist_ok=True)
    neighbors.to_csv(os.path.join(step_dir, "teddy_neighbors.csv"), index=False)
    mlflow.log_params({**teddy_params, "endpoint": TEDDY_EP, "index": TEDDY_INDEX})
    mlflow.log_metrics({"n_cells_annotated": len(teddy_res), "disease_agreement_with_known": float(agree),
                        "vocab_cos_symbol": float(vocab_check["hgnc_symbol"]["mean_cos_between_cells"]),
                        "vocab_cos_ensembl": float(vocab_check["ensembl_id"]["mean_cos_between_cells"])})
    teddy_res.to_csv(os.path.join(step_dir, "teddy_cell_predictions.csv"))
    cluster_summary.to_csv(os.path.join(step_dir, "teddy_cluster_summary.csv"))
    mlflow.log_artifacts(step_dir)

# COMMAND ----------

# DBTITLE 1,Report: session state and software versions
# Report prep: confirm the analysis state is in memory and collect software versions by running code
import importlib.metadata as md, platform, sys
needed = ["adata", "qc_params", "analysis_params", "qc_profile_df", "suggested_annotations",
          "cluster_vs_author", "marker_df", "n_cells_pre"]
print("Missing variables:", [v for v in needed if v not in globals()])
print("adata:", adata.shape, "| clusters:", adata.obs["leiden"].nunique(), "| obsm:", list(adata.obsm.keys()))
print("tmpdir files:", sorted(os.path.relpath(os.path.join(r, f), tmpdir.name)
                               for r, _, fs in os.walk(tmpdir.name) for f in fs))
def _ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None
pkgs = ["scanpy", "anndata", "harmonypy", "numpy", "pandas", "scikit-learn", "igraph", "leidenalg",
        "umap-learn", "matplotlib", "mlflow", "databricks-sdk"]
versions_df = pd.DataFrame([("python", platform.python_version())] +
                           [(p, _ver(p)) for p in pkgs if _ver(p)], columns=["package", "version"])
versions_df.loc[len(versions_df)] = ["databricks environment", os.environ.get("DATABRICKS_RUNTIME_VERSION", "serverless")]
display(versions_df)
print("MLflow parent run:", parent_run.info.run_id, "| experiment:", parent_run.info.experiment_id)

# COMMAND ----------

# DBTITLE 1,Build scientific HTML report to volume
import sys, datetime
_hr_dirs = ["/Workspace/.assistant/skills/html-report/scripts",
            f"/Workspace/Users/{spark.sql('SELECT current_user()').first()[0]}/.assistant/skills/html-report/scripts"]
_hr_dir = next((d for d in _hr_dirs if os.path.exists(os.path.join(d, "html_report.py"))), None)
if _hr_dir is None:
    raise FileNotFoundError(f"html_report.py not found in {_hr_dirs}")
sys.path.insert(0, _hr_dir)
import html_report as hr

REPORT_DIR = "/Workspace/Shared/tests/scrna processing/reports/scrna_7970bd6b"
os.makedirs(REPORT_DIR, exist_ok=True)
report_path = os.path.join(REPORT_DIR, "scrna_7970bd6b_report.html")
if os.path.exists(report_path):  # never overwrite: write a new, timestamped file
    report_path = report_path.replace(".html", f"_{datetime.datetime.now():%Y%m%d_%H%M%S}.html")
author = spark.sql("SELECT current_user()").first()[0]

# ---- Facts computed from the notebook state (no hand-typed numbers) ----
bk = ad.read_h5ad(H5AD_PATH, backed="r"); uns = dict(bk.uns); raw_obs = bk.obs.copy(); n_genes_file = bk.n_vars; bk.file.close()
obs = adata.obs
n_post, g_post = adata.n_obs, adata.n_vars
ret = 100 * n_post / n_cells_pre
n_donors, n_samples = raw_obs["donor_id"].nunique(), raw_obs["Sample"].nunique()
assay_counts = raw_obs["assay"].value_counts()
flag_counts = flags.sum().astype(int)
n_hvg = int(adata.var["highly_variable"].sum())
n_clusters = int(obs["leiden"].nunique())
mt_genes_in_matrix = int(adata.var["mt"].sum())
prof = qc_profile_df
pb_post = int((obs["cell_type"] == "plasmablast").sum())
pc_genes = [g for g in ["PRDM1", "XBP1", "JCHAIN", "MZB1", "SDC1"] if g in adata.raw.var_names]
pc_means = sc.get.obs_df(adata, keys=pc_genes + ["leiden"], use_raw=True).groupby("leiden", observed=True).mean()
pb_by_cluster = pd.crosstab(obs["leiden"], obs["cell_type"])["plasmablast"]
top_pb = pb_by_cluster.sort_values(ascending=False).head(2)
donor_share = pd.crosstab(obs["leiden"], obs["donor_id"], normalize="index")
donor_overall = obs["donor_id"].value_counts(normalize=True)
skew_donor = (donor_share.max(axis=0) / donor_overall).idxmax()
skewed = donor_share[skew_donor][donor_share[skew_donor] >= 2 * donor_overall[skew_donor]]
_c = adata.layers["counts"][:500]
_c = _c.toarray() if hasattr(_c, "toarray") else np.asarray(_c)
counts_ok = bool(np.allclose(_c, np.round(_c)))
author_min_counts, author_min_genes = int(raw_obs["nCount_RNA"].min()), int(raw_obs["nFeature_RNA"].min())
n_nbrs_graph = int(adata.uns["neighbors"]["params"]["n_neighbors"])
_sa = suggested_annotations.set_index("cluster")
fcrl4_cl = _sa.index[_sa["suggested_type"].str.contains("FCRL4")][0]
hsp_cls = list(_sa.index[_sa["suggested_type"].str.contains("Heat-shock")])
ieg_cls = list(_sa.index[_sa["suggested_type"].str.contains("immediate-early")])
_join = lambda xs: " and ".join([", ".join(xs[:-1]), xs[-1]]) if len(xs) > 1 else xs[0]
high_conf = suggested_annotations.query("confidence == 'High'")
top_author_pct = cluster_vs_author.drop(columns="n_cells").max(axis=1)
n_dominant = int((top_author_pct >= 50).sum())
leiden_iters = adata.uns.get("leiden", {}).get("params", {}).get("n_iterations", "not recorded")
n_mads = mad_bounds.__defaults__[0]

# ---- Figures ----
sc.pl.umap(adata, color=["leiden", "author_cell_type"], ncols=2, wspace=0.55, legend_loc="right margin", show=False)
fig_umap = plt.gcf(); fig_umap.set_size_inches(13, 5)
sc.pl.dotplot(adata, canonical, groupby="leiden", standard_scale="var", show=False)
fig_dot = plt.gcf()
sc.pl.umap(adata, color=["donor_id", "assay"], ncols=2, wspace=0.45, show=False)
fig_donor = plt.gcf(); fig_donor.set_size_inches(12, 4.8)
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"], groupby="donor_id", jitter=False,
             multi_panel=True, show=False)
fig_qc = plt.gcf()
plt.close("all")

# ---- Tables ----
annot_tbl = suggested_annotations[["cluster", "n_cells", "key_markers", "suggested_type", "confidence", "top_author_label"]]
pc_tbl = pc_means.round(2).assign(plasmablast_label_cells=pb_by_cluster).reset_index()
flag_tbl = flag_counts.rename("cells_flagged").reset_index().rename(columns={"index": "filter"})
prof_tbl = prof[["median", "P1", "P5", "P95", "P99", "MAD_low", "MAD_high"]].reset_index().rename(columns={"index": "metric"})
skew_tbl = pd.DataFrame({"cluster": skewed.index, f"{skew_donor}_share": skewed.round(2).values,
                         f"{skew_donor}_overall": round(float(donor_overall[skew_donor]), 2)})

summary = (f"We reanalysed a CELLxGENE human tonsil B-cell dataset ({n_cells_pre:,} cells, {n_donors} donors, {n_samples} libraries) "
           f"from raw counts. Data-driven QC kept {n_post:,} cells ({ret:.1f}%). After per-donor highly variable gene selection, Harmony "
           f"integration on donor and Leiden clustering (resolution {analysis_params['cluster_resolution']}), we found {n_clusters} "
           f"memory B-cell state clusters that recover the authors' subtypes, with {len(high_conf)} high-confidence match(es) "
           f"({', '.join(high_conf['suggested_type'])}). The source `plasmablast` label is not supported by plasma-cell gene expression.")

spec = {
  "title": f"{n_clusters} memory B-cell states in healthy pediatric tonsil, with no plasma-cell program behind the source plasmablast label",
  "authors": author,
  "summary": summary,
  "sections": [
    {"title": "Introduction", "blocks": [
      {"p": f"This dataset profiles memory B cells and plasmablasts from human tonsil (`{uns.get('title', 'title not recorded')}`). "
             "The question was whether an independent reanalysis from raw counts recovers the published B-cell states."},
      {"p": "The approach was a standard scanpy workflow with thresholds derived from the data and confirmed by the analyst, "
             "Harmony batch correction on donor, and marker-based cell-state suggestions compared against the author labels."}]},
    {"title": "Results", "blocks": [
      {"subheading": f"Unsupervised clustering resolves {n_clusters} B-cell states that track the authors' subtypes"},
      {"p": f"All {n_clusters} clusters are B-lineage. {n_dominant} clusters have a single dominant author subtype covering at least half of their cells, "
             f"and FCRL4+ memory B cells (cluster {fcrl4_cl}, {int(_sa.loc[fcrl4_cl,'n_cells']):,} cells) are recovered as a distinct population "
             f"({_sa.loc[fcrl4_cl,'top_author_label']} author agreement). Resting, activated, pre-germinal-centre, heat-shock-response "
             "and CD21-low states are separated. The labels are assistant suggestions from marker genes and need expert review."},
      {"figure": fig_umap, "caption": f"UMAP of {n_post:,} QC-passing cells after Harmony correction, coloured by Leiden cluster (left) and the authors' "
             "`author_cell_type` (right). Clusters largely coincide with author subtypes."},
      {"table": annot_tbl, "caption": "Suggested cell state per Leiden cluster, from top Wilcoxon markers and canonical genes, with the dominant "
             "author label and its share of the cluster. Suggestions are pending expert review."},
      {"figure": fig_dot, "caption": "Canonical B-cell, activation, stress and plasma-cell markers per cluster (dot size: fraction of expressing cells; "
             "colour: mean expression scaled per gene). Plasma-cell genes (PRDM1, XBP1, JCHAIN, MZB1) are low in every cluster."},
      {"subheading": "The source plasmablast label is not supported by plasma-cell gene expression"},
      {"p": f"The source `cell_type` column labels {pb_post:,} QC-passing cells as plasmablasts, concentrated in clusters "
             f"{top_pb.index[0]} ({int(top_pb.iloc[0]):,}) and {top_pb.index[1]} ({int(top_pb.iloc[1]):,}). These clusters express immediate-early "
             f"and activation genes, but the highest cluster mean of XBP1 is {pc_means['XBP1'].max():.2f} and of JCHAIN {pc_means['JCHAIN'].max():.2f} "
             "(log-normalized), with no cluster showing a plasma-cell program."},
      {"table": pc_tbl, "caption": "Mean log-normalized expression of plasma-cell genes per cluster and the number of cells carrying the source "
             "`plasmablast` label."}]},
    {"title": "Discussion", "blocks": [
      {"bullets": [
        "The reanalysis reproduces the published memory B-cell state structure, which supports the robustness of those subtypes to an independent pipeline.",
        "The `plasmablast` label in the curated metadata appears to mark activated memory B cells rather than antibody-secreting cells; it should be checked against the original publication before being used downstream.",
        f"Heat-shock (clusters {_join(hsp_cls)}) and immediate-early (cluster {_join(ieg_cls)}) states may partly reflect tissue dissociation stress rather than in vivo biology."]},
      {"subheading": "Limitations"},
      {"bullets": [
        f"Residual donor structure remains after Harmony: donor {skew_donor} makes up {skewed.min():.0%}-{skewed.max():.0%} of clusters "
        f"{', '.join(skewed.index)} against {donor_overall[skew_donor]:.0%} of all cells.",
        "Cell-state labels are LLM suggestions from marker genes, not reference-based annotation, and have not yet been reviewed by an expert.",
        "Mitochondrial genes are absent from the matrix, so the mitochondrial filter relies on the authors' precomputed fraction.",
        f"Only {int(assay_counts.min())} cells come from the second assay, which is too few to assess assay effects separately from donor."]}]},
    {"title": "Methods", "blocks": [
      {"subheading": "Data sources"},
      {"path": H5AD_PATH},
      {"p": f"The h5ad file contains {n_cells_pre:,} cells and {n_genes_file:,} genes from {n_donors} donors and {n_samples} libraries; "
             + "; ".join(f"{k}: {v:,} cells" for k, v in assay_counts.items())
             + ". Raw integer counts were taken from `.raw` (the main matrix is normalized) and Ensembl IDs were mapped to gene symbols with `feature_name`. "
             f"The authors had already filtered cells (minimum {author_min_counts:,} counts and {author_min_genes:,} genes per cell in the file)."},
      {"subheading": "Quality control"},
      {"p": f"QC thresholds were set from this dataset's distributions (median plus or minus {n_mads} MAD, percentiles) and confirmed by the analyst: "
             f"mitochondrial fraction below {qc_params['max_pct_counts_mt']}%, between {qc_params['min_genes']} and {qc_params['max_genes']} detected genes, "
             f"removal of author-flagged doublets (DoubletFinder non-singlet or Scrublet positive), and genes detected in at least "
             f"{qc_params['min_cells']} cells. The matrix contains {mt_genes_in_matrix} MT- genes, so the authors' precomputed `percent.mito` (x100) was used for the mitochondrial filter. "
             f"This kept {n_post:,} of {n_cells_pre:,} cells ({ret:.1f}%) and {g_post:,} genes."},
      {"table": prof_tbl, "caption": "QC metric distributions before filtering, used to derive the thresholds."},
      {"table": flag_tbl, "caption": "Cells flagged by each filter (a cell can fail several)."},
      {"figure": fig_qc, "caption": "Genes detected, total counts and mitochondrial percentage per donor for the QC-passing cells."},
      {"subheading": "Normalization, integration and clustering"},
      {"p": f"Counts were normalized to {analysis_params['target_sum']:,} per cell and log1p-transformed. {n_hvg:,} highly variable genes were "
             f"selected per donor (scanpy `seurat` flavour, batch key `{analysis_params['hvg_batch_key']}`). PCA ({analysis_params['n_comps']} components, "
             f"HVGs only) was corrected with Harmony (harmonypy, batch key `{analysis_params['batch_key']}`, random state {analysis_params['random_state']}). "
             f"A {n_nbrs_graph}-nearest-neighbour graph on the first {analysis_params['n_pcs']} Harmony components fed UMAP and Leiden clustering "
             f"(igraph, resolution {analysis_params['cluster_resolution']}, {leiden_iters} iterations). Marker genes were ranked per cluster with a Wilcoxon rank-sum "
             "test on log-normalized values. Donor was chosen as the batch key because assay is nested within donor. Analysis ran on Databricks serverless CPU."},
      {"figure": fig_donor, "caption": "UMAP coloured by donor (left) and assay (right) after Harmony integration, used to assess donor mixing."},
      {"table": skew_tbl, "caption": f"Clusters where donor {skew_donor} is at least twice its overall share."},
      {"subheading": "Checks"},
      {"checks": [
        {"check": "Raw counts are integer-valued", "threshold": "all integer", "result": str(counts_ok), "passed": counts_ok},
        {"check": "Highly variable genes selected", "threshold": f"= {analysis_params['n_top_genes']}", "result": str(n_hvg), "passed": n_hvg == analysis_params['n_top_genes']},
        {"check": "Harmony embedding shape", "threshold": f"({n_post}, {analysis_params['n_comps']})", "result": str(adata.obsm['X_pca_harmony'].shape),
         "passed": adata.obsm['X_pca_harmony'].shape == (n_post, analysis_params['n_comps'])},
        {"check": "Annotation table matches current clusters", "threshold": "same cluster IDs",
         "result": str(set(annot_tbl.cluster) == set(obs['leiden'].cat.categories)), "passed": set(annot_tbl.cluster) == set(obs['leiden'].cat.categories)},
        {"check": "Cell-state labels reviewed", "threshold": "expert review", "result": "pending", "status": "pending"}]},
      {"subheading": "Software"},
      {"table": versions_df, "caption": "Software versions read with importlib.metadata in the analysis session."}]},
    {"title": "Code availability", "blocks": [
      {"p": "The analysis notebook is in the Databricks workspace and is not publicly released:"},
      {"path": "/Workspace/Shared/tests/scrna processing/scRNA Analysis 7970bd6b.ipynb"},
      {"p": f"Parameters, metrics and artifacts were logged to MLflow (parent run `{parent_run.info.run_id}`, experiment `{parent_run.info.experiment_id}`)."}]},
    {"title": "Data availability", "blocks": [
      {"p": f"The input is a CELLxGENE-schema h5ad file (schema version {uns.get('schema_version', 'not recorded')}) staged on a Unity Catalog volume; "
             f"citation recorded in the file: {uns.get('citation', 'not recorded')}. No derived tables were written."}]},
    {"title": "References", "blocks": [{"bullets": [
      "Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. Genome Biology (2018). [doi:10.1186/s13059-017-1382-0](https://doi.org/10.1186/s13059-017-1382-0)",
      "Korsunsky I, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nature Methods (2019). [doi:10.1038/s41592-019-0619-0](https://doi.org/10.1038/s41592-019-0619-0)",
      "Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports (2019). [doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z)"]}]},
  ],
}

out = hr.build_report(spec, report_path)
print({k: v for k, v in out.items() if k != "problems"})
print("Problems:", out["problems"])
with mlflow.start_run(run_name="report", nested=True):
    mlflow.log_artifact(out["path"])
    mlflow.log_param("report_path", out["path"])
hr.display_report(out["path"])

# COMMAND ----------

# DBTITLE 1,Build slide deck from analysis
# ---- Presentation deck: grounded in session DataFrames, no hand-typed numbers ----
import sys, os, datetime
_deck_dir = "/Workspace/.assistant/skills/presentation-builder/scripts"
assert os.path.exists(os.path.join(_deck_dir, "build_deck.py")), f"build_deck.py not at {_deck_dir}"
if _deck_dir not in sys.path:
    sys.path.insert(0, _deck_dir)
import build_deck

# ---- Ground every number from session state ----
_obs = adata.obs
_n_post = adata.n_obs
_n_clusters = int(_obs["leiden"].nunique())
_n_donors = int(_obs["donor_id"].nunique())
_retention = f"{100 * _n_post / n_cells_pre:.1f}%"
_n_hvg = int(adata.var["highly_variable"].sum())
_res = analysis_params["cluster_resolution"]

# Cells per cluster, labelled with suggested type
_sa = suggested_annotations.set_index("cluster")
_cpc = _obs["leiden"].value_counts().sort_values(ascending=False)
_bars_cluster = [{"label": f"{cl}: {_sa.loc[cl, 'suggested_type']}",
                  "value": int(n)} for cl, n in _cpc.items()]

# Plasma-cell genes per cluster (max mean expression)
_pc_genes = [g for g in ["PRDM1", "XBP1", "JCHAIN", "MZB1"] if g in adata.raw.var_names]
_pc_means = sc.get.obs_df(adata, keys=_pc_genes + ["leiden"], use_raw=True).groupby("leiden", observed=True).mean()
_bars_pc = [{"label": g, "value": round(float(_pc_means[g].max()), 2),
             "display": f"{_pc_means[g].max():.2f}"} for g in _pc_genes]

# High-confidence clusters
_high = suggested_annotations.query("confidence == 'High'")
_pb_post = int((_obs["cell_type"] == "plasmablast").sum())
_pb_by = pd.crosstab(_obs["leiden"], _obs["cell_type"])["plasmablast"].sort_values(ascending=False)
_top2 = _pb_by.head(2)

# Confidence distribution
_conf_vc = suggested_annotations["confidence"].value_counts()
_bars_conf = [{"label": lbl, "value": int(ct)} for lbl, ct in _conf_vc.items()]

# ---- Assemble deck spec ----
spec = {
    "title": f"{_n_clusters} Memory B-Cell States in Healthy Pediatric Tonsil",
    "footer_left": "Databricks · scRNA-seq Reanalysis",
    "slides": [
        # 1 — Cover
        {"type": "cover", "theme": "dark",
         "title": f"{_n_clusters} memory B-cell states in [[healthy pediatric tonsil]]",
         "subtitle": "Independent reanalysis from raw counts with no plasma-cell program behind the source plasmablast label.",
         "meta": [f"{_n_post:,} cells · {_n_donors} donors", f"Sample {SAMPLE_ID}",
                  datetime.date.today().isoformat()],
         "no_footer": True},

        # 2 — Headline metrics
        {"type": "metrics", "kicker": "DATASET",
         "title": "Analysis at a glance",
         "tiles": [
             {"value": f"{n_cells_pre:,}", "label": "Cells before QC",
              "caption": "CELLxGENE tonsil B-cell dataset"},
             {"value": f"{_n_post:,}", "label": "Cells after QC",
              "caption": f"{_retention} retention"},
             {"value": str(_n_donors), "label": "Donors",
              "caption": f"{int(_obs['Sample'].nunique())} libraries"},
             {"value": str(_n_clusters), "label": "Leiden clusters",
              "caption": f"Resolution {_res}, Harmony-corrected"},
         ]},

        # 3 — Cells per cluster
        {"type": "bar_chart", "kicker": "CLUSTERING",
         "title": "Cells per cluster",
         "subtitle": f"Leiden resolution {_res} after Harmony batch correction on donor.",
         "chart": {"heading": "Cells", "unit": "cells", "bars": _bars_cluster},
         "notes": f"Grounded: adata.obs['leiden'].value_counts(), n={_n_post:,}"},

        # 4 — Cluster annotation summary
        {"type": "bullets", "kicker": "ANNOTATION",
         "title": "LLM-suggested cell states recover published subtypes",
         "points": [
             {"heading": f"{len(_high)} high-confidence match",
              "body": f"{', '.join(_high['suggested_type'])} — FCRL4+ atypical memory B cells recovered as a distinct population with {_sa.loc[_high.iloc[0]['cluster'], 'top_author_label']} author agreement."},
             {"heading": "Resting to activated spectrum",
              "body": "Resting quiescent, CD21-high resting, pre-GC, activated (immediate-early and IgM-high) states are cleanly separated."},
             {"heading": "Heat-shock / stress-response clusters",
              "body": "Two clusters (4 and 7) driven by HSPA1A, DNAJB1 — may partly reflect dissociation artifact rather than in vivo biology."},
             {"heading": "Labels pending expert review",
              "body": "All annotations are marker-gene suggestions, not reference-based — expert validation is required before downstream use."},
         ]},

        # 5 — The plasmablast finding
        {"type": "statement", "theme": "dark",
         "eyebrow": "KEY FINDING",
         "statement": f"The source [[plasmablast]] label ({_pb_post:,} cells) is **not supported** by plasma-cell gene expression.",
         "steps": [
             f"Clusters {_top2.index[0]} ({int(_top2.iloc[0]):,} cells) and {_top2.index[1]} ({int(_top2.iloc[1]):,} cells) concentrate the label.",
             "These clusters express immediate-early and activation genes — not antibody-secreting cell markers.",
             "Check the original annotation against the publication before downstream use.",
         ]},

        # 6 — Plasma-cell gene expression (bar chart)
        {"type": "bar_chart", "kicker": "EVIDENCE",
         "title": "Peak cluster-mean of plasma-cell markers",
         "subtitle": "Max mean log-normalised expression across all clusters — all well below plasma-cell levels.",
         "chart": {"heading": "Max cluster mean", "unit": "log-norm", "bars": _bars_pc, "max": 3.0},
         "notes": f"Grounded: sc.get.obs_df(adata, use_raw=True).groupby('leiden').mean().max()"},

        # 7 — Discussion / limitations
        {"type": "bullets", "kicker": "LIMITATIONS",
         "title": "Caveats and next steps",
         "points": [
             {"heading": "Residual donor structure",
              "body": "Donor-specific enrichment persists after Harmony in select clusters — consider alternative correction or higher integration strength."},
             {"heading": "Mitochondrial genes absent",
              "body": "MT filter relies on the authors' precomputed fraction; the matrix itself contains zero MT genes."},
             {"heading": "In-silico annotation only",
              "body": "Cell-state labels are LLM suggestions from Wilcoxon markers and canonical genes — not reference-based mapping or protein-level validation."},
         ]},

        # 8 — Closing
        {"type": "closing", "theme": "dark",
         "eyebrow": "NEXT STEPS",
         "statement": "Expert review → reference mapping → downstream integration",
         "steps": [
             "Validate suggested annotations against the original publication and protein markers.",
             "Run reference-based label transfer (e.g. CellTypist, Azimuth) for independent confirmation.",
             "Integrate with other tonsil datasets to test reproducibility of the B-cell states.",
         ]},
    ],
}

# ---- Build the deck ----
deck_path = os.path.join(REPORT_DIR, f"scrna_{SAMPLE_ID}_deck.html")
if os.path.exists(deck_path):
    deck_path = deck_path.replace(".html", f"_{datetime.datetime.now():%Y%m%d_%H%M%S}.html")
result = build_deck.build_deck(spec, deck_path)
print(result)
from IPython.display import HTML, display
display(HTML(f'<a href="{deck_path}" target="_blank">Open slide deck ({result["slides"]} slides, {result["bytes"]:,} bytes)</a>'))