# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # KERMT ADMET Batch Predictions
# MAGIC
# MAGIC Batch-score every molecule in `main.kermt.data_public_test` via the **KERMT ADMET** serving endpoint, join predictions back with all original columns, and persist the results as a new table `main.kermt.data_public_test_scored`.

# COMMAND ----------

# DBTITLE 1,Score molecules and save to new table
# MAGIC %sql
# MAGIC -- Score every molecule and persist all original columns + the toxicity prediction
# MAGIC -- into a new table (does NOT modify the original)
# MAGIC CREATE TABLE IF NOT EXISTS main.kermt.data_public_test_scored AS
# MAGIC SELECT
# MAGIC   t.*,
# MAGIC   ai_query(
# MAGIC     'gwb_demo_kermt_admet_endpoint',
# MAGIC     t.smiles
# MAGIC   ).toxicity AS predicted_toxicity
# MAGIC FROM main.kermt.data_public_test t

# COMMAND ----------

# DBTITLE 1,Verify the new table
# MAGIC %sql
# MAGIC -- Quick verification: row count, schema, and a sample
# MAGIC DESCRIBE TABLE main.kermt.data_public_test_scored

# COMMAND ----------

display(spark.table("main.kermt.data_public_test_scored"))

# COMMAND ----------

# DBTITLE 1,Setup and compute correlations
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ── Scientific figure defaults ────────────────────────────────────────────────
_FIGURE_STYLE = {
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "figure.titlesize": 8,
    "axes.titleweight": "bold", "figure.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.title_fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.linewidth": 0.8,
    "lines.linewidth": 1.5, "lines.markersize": 5,
    "legend.frameon": False,
    "axes.prop_cycle": mpl.cycler(
        color=["#000000", "#D62728", "#3A6EA5", "#F28C28", "#1BA7A6", "#8C8C8C"]
    ),
    "image.cmap": "viridis",
}
mpl.rcParams.update(_FIGURE_STYLE)

BRAND_DIV = LinearSegmentedColormap.from_list(
    "brand_div", ["#3A6EA5", "#FFFFFF", "#D62728"])

# ── Load data and compute correlations ────────────────────────────────────────
pdf = spark.table("main.kermt.data_public_test_scored").drop("smiles").toPandas()

corr = pdf.corr(method="spearman")["predicted_toxicity"].drop("predicted_toxicity").sort_values()
non_null_counts = pdf.drop(columns="predicted_toxicity").count()
full_corr = pdf.corr(method="spearman")
top4 = corr.abs().nlargest(4).index.tolist()

print(f"Loaded {len(pdf)} molecules, {len(corr)} ADMET endpoints.")
print(f"Top-4 correlated endpoints: {top4}")

# COMMAND ----------

# DBTITLE 1,Spearman correlation bar chart
# ── Rank-correlation of each ADMET endpoint with predicted toxicity ───────────
fig, ax = plt.subplots(figsize=(5.85, 4.0))
colors = ["#D62728" if v > 0 else "#3A6EA5" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, edgecolor="none")
ax.set_xlabel("Spearman ρ with predicted_toxicity")
ax.set_title("Rank-correlation of each ADMET endpoint\nwith predicted toxicity", pad=12)
ax.axvline(0, color="#4D4D4D", lw=0.5)
w, h = fig.get_size_inches()
print(f"Figure size: {w:.2f} x {h:.2f} in")
plt.show()

# COMMAND ----------

# DBTITLE 1,ADMET Spearman heatmap
# ── Full ADMET Spearman correlation heatmap ───────────────────────────────────
mask = np.triu(np.ones_like(full_corr, dtype=bool), k=1)
n = len(full_corr)

# Square heatmap: width-driven sizing so aspect="equal" doesn't leave dead space
fig, ax = plt.subplots(figsize=(5.85, 5.85))
im = ax.imshow(
    full_corr.where(~mask), cmap=BRAND_DIV,
    norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1), aspect="equal"
)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(full_corr.columns, rotation=90)
ax.set_yticklabels(full_corr.columns)
ax.set_title("Full ADMET Spearman correlation heatmap")
cb = plt.colorbar(im, ax=ax, shrink=0.75, label="ρ")
w, h = fig.get_size_inches()
print(f"Figure size: {w:.2f} x {h:.2f} in")
plt.show()

# COMMAND ----------

# DBTITLE 1,Top-4 correlated endpoint scatter plots
# ── Scatter plots: top-4 strongest absolute correlations with toxicity ────────
fig, axes = plt.subplots(2, 2, figsize=(5.85, 5.0))
fig.subplots_adjust(hspace=0.55, wspace=0.4)
for idx, feat in enumerate(top4):
    ax = axes.flat[idx]
    sub = pdf[[feat, "predicted_toxicity"]].dropna()
    ax.scatter(
        sub[feat], sub["predicted_toxicity"],
        s=8, alpha=0.45, color="#000000", rasterized=True,
    )
    rho = sub[feat].corr(sub["predicted_toxicity"], method="spearman")
    ax.set_xlabel(feat)
    ax.set_ylabel("predicted_toxicity")
    ax.set_title(f"ρ = {rho:+.2f}  (n = {len(sub)})")
w, h = fig.get_size_inches()
print(f"Figure size: {w:.2f} x {h:.2f} in")
plt.show()

# COMMAND ----------

# DBTITLE 1,Spearman correlation summary table
# ── Summary table: Spearman correlations with predicted_toxicity ──────────────
summary = pd.DataFrame({
    "spearman_rho": corr,
    "non_null_pairs": non_null_counts,
}).sort_values("spearman_rho", key=abs, ascending=False)
summary.index.name = "endpoint"
display(summary.reset_index())

# COMMAND ----------

# DBTITLE 1,Install mols2grid
# MAGIC %pip install mols2grid

# COMMAND ----------

# DBTITLE 1,Render 3 lowest-toxicity molecules with mols2grid
# ── Render the 3 lowest-toxicity molecules with mols2grid ────────────────────
import mols2grid

# Pull the 3 molecules with the lowest predicted_toxicity
lowest3 = (
    spark.table("main.kermt.data_public_test_scored")
    .where("predicted_toxicity IS NOT NULL")
    .orderBy("predicted_toxicity")
    .limit(3)
    .select("smiles", "predicted_toxicity")
    .toPandas()
)

# Build a static (non-interactive) grid — Databricks doesn't render ipywidgets
grid = mols2grid.MolGrid(
    lowest3, smiles_col="smiles", size=(300, 250), prerender=True
)
html = grid.to_static(subset=["img", "predicted_toxicity"], n_cols=3)
displayHTML(html)