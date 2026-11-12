# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-70M Setup
# MAGIC
# MAGIC Shared resources for the single-cell annotation demo: downloads the pinned TEDDY-70M checkpoint, samples Census reference/query cells, builds a Vector Search index, serves the model, and validates against the tutorial (S0–S5).

# COMMAND ----------

# DBTITLE 1,Install (one cell, one restart)
# MAGIC %pip install -q "mlflow>=3.12" "databricks-sdk>=0.102.0" "transformers==4.41.0" anndata scipy cellxgene-census
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
import os, json, shutil, datetime as dt
import pandas as pd
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
CATALOG, SCHEMA, VOLUME = "main", "owm", "teddy_files"
FQ = f"{CATALOG}.{SCHEMA}"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SHARED = f"{VOL}/shared"
MODEL_DIR_ROOT = f"{SHARED}/teddy_70m"
SENTINEL = f"{MODEL_DIR_ROOT}/.download_complete"
HF_REPO, HF_REV = "Merck/TEDDY", "8c880aa67674042ccdc254bccb5de081092a48bf"
REF_TABLE, VS_ENDPOINT, VS_INDEX = f"{FQ}.teddy_ref_cells", "teddy_vs_workshop", f"{FQ}.teddy_ref_index"
UC_MODEL, SERVE_EP = f"{FQ}.teddy_model_shared", "teddy_ep_shared"
USER = w.current_user.me().user_name
NB_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
CONFIRM_DELETE = False  # set True only after the user confirms the S0 list in chat

STEPS = {}
def now(): return dt.datetime.now(dt.timezone.utc)
def start(s): STEPS[s] = {"start": now()}; print(f"{s} start {STEPS[s]['start']:%Y-%m-%d %H:%M:%S} UTC")
def end(s):
    STEPS[s]["end"] = now(); d = (STEPS[s]["end"] - STEPS[s]["start"]).total_seconds() / 60
    print(f"{s} end {STEPS[s]['end']:%Y-%m-%d %H:%M:%S} UTC ({d:.1f} min)")
print(f"user={USER}\nnotebook={NB_PATH}\nvolume={VOL}")

# COMMAND ----------

# DBTITLE 1,S0a: inventory the schema (no deletes)
start("S0")
def ts(ms): return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M") if ms else None

def inventory():
    rows = []
    for t in w.tables.list(catalog_name=CATALOG, schema_name=SCHEMA):
        kind = "view" if str(t.table_type).endswith("VIEW") else "table"
        rows.append((kind, t.full_name, ts(t.created_at), t.owner))
    for m in w.registered_models.list(catalog_name=CATALOG, schema_name=SCHEMA):
        rows.append(("model", m.full_name, ts(m.created_at), m.owner))
        vs = w.api_client.do("GET", f"/api/2.1/unity-catalog/models/{m.full_name}/versions").get("model_versions", [])
        for v in vs:
            rows.append(("model_version", f"{m.full_name}/{v['version']}", ts(v.get("created_at")), v.get("created_by")))
    for e in w.vector_search_endpoints.list_endpoints():
        idx = [i for i in w.vector_search_indexes.list_indexes(endpoint_name=e.name) if i.name.startswith(FQ + ".")]
        for i in idx:
            rows.append(("vs_index", i.name, None, i.creator))
        if idx:
            rows.append(("vs_endpoint", e.name, ts(e.creation_timestamp), e.creator))
    for ep in w.serving_endpoints.list():
        ents = (ep.config.served_entities or []) if ep.config else []
        pend = w.serving_endpoints.get(ep.name).pending_config
        ents += (pend.served_entities or []) if pend else []
        if any((s.entity_name or "").startswith(FQ + ".") for s in ents):
            rows.append(("serving_endpoint", ep.name, ts(ep.creation_timestamp), ep.creator))
    for f in w.files.list_directory_contents(VOL):
        rows.append(("volume_folder" if f.is_directory else "volume_file", f.path, ts(f.last_modified), None))
    return pd.DataFrame(rows, columns=["type", "name", "created", "owner"])

inv = inventory()
print(inv.type.value_counts().to_string() if len(inv) else "schema is empty")
display(inv)

# COMMAND ----------

# DBTITLE 1,S0b: delete everything listed (needs CONFIRM_DELETE)
if os.path.exists(SENTINEL):
    mt = dt.datetime.fromtimestamp(os.path.getmtime(SENTINEL), dt.timezone.utc)
    assert mt < STEPS["S0"]["start"], f"REFUSED: {SENTINEL} ({mt}) is newer than S0 start; S1 already ran"
    print(f"old sentinel from {mt:%Y-%m-%d %H:%M} UTC (earlier run): will be deleted")
assert CONFIRM_DELETE, "Not confirmed: set CONFIRM_DELETE = True after the user approves the S0 list"

by = lambda k: inv[inv.type == k].name.tolist()
for ep in by("serving_endpoint"):
    w.serving_endpoints.delete(ep); print("deleted serving endpoint", ep)
for ix in by("vs_index"):
    w.vector_search_indexes.delete_index(ix); print("deleted index", ix)
for t in inv[inv.type.isin(["table", "view"])].name:
    spark.sql(f"DROP {'VIEW' if inv.set_index('name').type[t] == 'view' else 'TABLE'} IF EXISTS {t}"); print("dropped", t)
for mv in by("model_version"):
    m, v = mv.rsplit("/", 1); w.model_versions.delete(m, int(v)); print("deleted version", mv)
for m in by("model"):
    w.registered_models.delete(m); print("deleted model", m)
for p in by("volume_folder"):
    shutil.rmtree(p); print("deleted folder", p)
for p in by("volume_file"):
    os.remove(p); print("deleted file", p)
import time
for e in by("vs_endpoint"):
    for _ in range(30):
        left = list(w.vector_search_indexes.list_indexes(endpoint_name=e))
        if not left: break
        time.sleep(10)
    if left: print(f"kept VS endpoint {e}: still holds {len(left)} index(es)")
    else: w.vector_search_endpoints.delete_endpoint(e); print("deleted VS endpoint", e)

# COMMAND ----------

# DBTITLE 1,S0c: check only the schema and an empty volume remain
inv2 = inventory()
left = inv2[inv2.type != "vs_endpoint"]
print("schema exists:", w.schemas.get(FQ).full_name, "| volume exists:", w.volumes.read(f"{FQ}.{VOLUME}").full_name)
display(inv2) if len(inv2) else None
assert left.empty, f"S0 FAILED: {len(left)} objects remain"
print("S0 PASS: only the schema and an empty teddy_files remain")
end("S0")

# COMMAND ----------

# DBTITLE 1,S1a: download TEDDY-70M at the pinned revision, checksum each file
import hashlib, requests
start("S1")
os.makedirs(SHARED, exist_ok=True)
assert not os.path.exists(MODEL_DIR_ROOT), f"{MODEL_DIR_ROOT} exists: not overwriting"
meta = requests.get(f"https://huggingface.co/api/models/{HF_REPO}/revision/{HF_REV}?blobs=true", timeout=60).json()
assert meta["sha"] == HF_REV
SKIP = ("teddy/models/teddy_g/160M/", "teddy/models/teddy_g/400M/")
files = [s for s in meta["siblings"] if not s["rfilename"].startswith(SKIP) and not s["rfilename"].endswith(".DS_Store")]
skipped = sorted(s["rfilename"] for s in meta["siblings"] if s not in files)

DL = "/tmp/teddy_s1/dl"
records = []
for s in files:
    p = s["rfilename"]; dst = f"{DL}/{p}"; os.makedirs(os.path.dirname(dst), exist_ok=True)
    with requests.get(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{p}", stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20): f.write(chunk)
    data = open(dst, "rb").read()
    if s.get("lfs"):
        got, want, kind = hashlib.sha256(data).hexdigest(), s["lfs"]["sha256"], "sha256"
    else:
        got, want, kind = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), s["blobId"], "git-sha1"
    assert got == want, f"checksum mismatch: {p}"
    records.append({"path": p, "bytes": len(data), kind: got})

shutil.copytree(DL, MODEL_DIR_ROOT)
readme = open(f"{DL}/README.md").read(); pyproj = open(f"{DL}/pyproject.toml").read()
lic_card = [l.split(":", 1)[1].strip() for l in readme.splitlines() if l.startswith("license:")]
lic_pkg = [l.split("=", 1)[1].strip().strip('"') for l in pyproj.splitlines() if l.startswith("license")]
manifest = {"source": f"https://huggingface.co/{HF_REPO}", "revision": HF_REV, "variant": "70M",
            "license_model_card": lic_card, "license_package_pyproject": lic_pkg,
            "license_note": "two declarations; confirm with the model owner before redistribution",
            "acquired_utc": now().isoformat(), "acquired_by": USER, "skipped": skipped, "files": records}
json.dump(manifest, open(f"{MODEL_DIR_ROOT}/manifest.json", "w"), indent=1)
open(SENTINEL, "w").write(now().isoformat())
print(f"{len(records)} files, {sum(r['bytes'] for r in records)/1e6:.0f} MB, all checksums match; licences card={lic_card} pkg={lic_pkg}; skipped {len(skipped)}")

# COMMAND ----------

# DBTITLE 1,Embedding code used in S1, S3, S4 (wrapper file + code bundle)
WRAPPER_SRC = r'''
import sys, os, io, json, types
import numpy as np, pandas as pd, torch, mlflow
from scipy.sparse import csr_matrix

CONTROLS = {"pooling": ("mean", {"mean", "tutorial", "first"}), "max_seq_len": ("2048", None)}

class TEDDYEmbedder(mlflow.pyfunc.PythonModel):
    """Raw counts in; TEDDY's own preprocess/tokenize functions; pooling mean|tutorial|first."""
    def load_context(self, context):
        for m in [m for m in sys.modules if m == "teddy" or m.startswith("teddy.")]:
            del sys.modules[m]
        pkg = context.artifacts["teddy_pkg_parent"]
        if pkg in sys.path: sys.path.remove(pkg)
        sys.path.insert(0, pkg)
        from teddy.models.model_directory import get_architecture, model_dict
        from teddy.tokenizer.gene_tokenizer import GeneTokenizer
        from teddy.data_processing.preprocessing.preprocess import normalize_data_inplace, scale_columns_by_median_dict
        from teddy.data_processing.tokenization.tokenization import _check_genes_in_tokenizer, _build_batch_tensors
        self.f = types.SimpleNamespace(norm=normalize_data_inplace, scale=scale_columns_by_median_dict,
                                       check=_check_genes_in_tokenizer, build=_build_batch_tensors)
        md = context.artifacts["model_dir"]
        arch = get_architecture(md)
        self.config = model_dict[arch]["config_cls"].from_pretrained(md)
        self.model = model_dict[arch]["model_cls"].from_pretrained(md, config=self.config)
        self.model.return_all_embs = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).float().eval()
        self.tok = GeneTokenizer.from_pretrained(md)
        self.medians = f"{pkg}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json"
        self.batch = 16 if self.device == "cuda" else 4

    def tokenize(self, counts, genes, max_seq_len=2048):
        layer = csr_matrix(np.asarray(counts, dtype=np.float32))
        data = types.SimpleNamespace(var=pd.DataFrame(index=pd.Index(list(genes))))
        self.f.norm(layer, 10000)                                   # normalized_total 10000
        self.f.scale(layer, data, self.medians, "index")             # divide by TEDDY gene medians
        in_vocab, coding, _ = self.f.check(data, "index", self.tok)  # drop OOV genes
        token_array = torch.tensor(self.tok.encode(coding.tolist(), add_special_tokens=False))
        args = types.SimpleNamespace(max_seq_len=int(max_seq_len), add_cls=False, random_genes=False)
        ids, vals, _, _ = self.f.build(torch.tensor(layer[:, in_vocab].toarray(), dtype=torch.float),
                                       token_array, args)
        ids = ids.clone(); ids[vals <= 0] = self.tok.pad_token_id    # include_zero_genes=False
        return ids, (vals > 0).long()

    @torch.no_grad()
    def embed(self, counts, genes, pooling="mean", max_seq_len=2048):
        counts = np.asarray(counts, dtype=np.float32); out = []
        for s in range(0, counts.shape[0], self.batch):
            ids, mask = self.tokenize(counts[s:s + self.batch], genes, max_seq_len)
            if pooling == "tutorial" and ids.shape[1] < int(max_seq_len):   # tutorial collate pads to max_seq_len
                pad = int(max_seq_len) - ids.shape[1]
                ids = torch.cat([ids, torch.full((ids.shape[0], pad), self.tok.pad_token_id)], 1)
                mask = torch.cat([mask, torch.zeros((mask.shape[0], pad), dtype=torch.long)], 1)
            o = self.model(gene_ids=ids.to(self.device), attention_mask=mask.to(self.device), return_outputs=True)
            H = o["all_embs"].float(); M = mask.to(self.device).unsqueeze(-1).float()
            if pooling == "mean": e = (H * M).sum(1) / M.sum(1)
            elif pooling == "tutorial": e = H.mean(1)
            else: e = o["cell_emb"].float()
            out.append(e.cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def predict(self, context, model_input, params=None):
        rows = []
        for _, r in model_input.iterrows():
            used = {}
            for k, (d, allowed) in CONTROLS.items():
                v = r[k] if k in model_input.columns and isinstance(r[k], str) else (params or {}).get(k, d)
                if allowed and str(v) not in allowed:
                    raise ValueError(f"{k}={v!r} not supported; use one of {sorted(allowed)}")
                used[k] = str(v)
            var = pd.read_json(io.StringIO(r["adata_var"]), orient="split")
            genes = var["index"].tolist() if "index" in var.columns else list(var.index)
            E = self.embed(np.array(r["adata_sparsematrix"], dtype=np.float32), genes, used["pooling"], int(used["max_seq_len"]))
            rows += [{"embedding": e.tolist(), **used} for e in E]
        return pd.DataFrame(rows)

mlflow.models.set_model(TEDDYEmbedder())
'''
import sys, types
CODE_DIR = "/tmp/teddy_code"
MODEL_DIR = f"{MODEL_DIR_ROOT}/teddy/models/teddy_g/70M"
assert all(os.path.exists(f"{MODEL_DIR}/{f}") for f in ("config.json", "model.safetensors", "vocab.txt"))
if not os.path.exists(f"{CODE_DIR}/teddy"):
    shutil.copytree(f"{MODEL_DIR_ROOT}/teddy", f"{CODE_DIR}/teddy",
                    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.pt", "__pycache__", ".DS_Store"))
WRAPPER_PATH = f"{CODE_DIR}/teddy_wrapper.py"
open(WRAPPER_PATH, "w").write(WRAPPER_SRC)
ARTIFACTS = {"model_dir": MODEL_DIR, "teddy_pkg_parent": CODE_DIR}

def load_embedder():
    ns = {}; exec(compile(WRAPPER_SRC, WRAPPER_PATH, "exec"), ns)
    m = ns["TEDDYEmbedder"](); m.load_context(types.SimpleNamespace(artifacts=ARTIFACTS)); return m
print(f"wrapper at {WRAPPER_PATH}; code bundle {sum(len(f) for _,_,f in os.walk(CODE_DIR))} files")

# COMMAND ----------

# DBTITLE 1,S1b: (a) tutorial's own cells, unchanged except paths, 20 cells
import numpy as np, torch, anndata as ad, gc
REPO = "/tmp/teddy_s1/repo"               # tutorial writes data/processed and data/tokenized under the repo root
if os.path.exists(REPO): shutil.rmtree(REPO)   # scratch copy this run made
shutil.copytree(MODEL_DIR_ROOT, REPO, ignore=shutil.ignore_patterns("manifest.json", ".download_complete"))
tut = json.load(open(f"{REPO}/tutorials/model_tutorial.ipynb"))
code = ["".join(c["source"]) for c in tut["cells"] if c["cell_type"] == "code"]
assert code[0].strip() == "cd .."
os.chdir(REPO)                                # = the tutorial's `cd ..` from tutorials/ (path change)
for k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]: del sys.modules[k]
sys.path.insert(0, REPO)
N_CHECK = 20
changed = []
for src in code[1:8]:                         # imports, preprocess, tokenize, model load, collate, dataloader, forward
    if "max_eval_samples=15" in src:
        src = src.replace("max_eval_samples=15", f"max_eval_samples={N_CHECK}"); changed.append("max_eval_samples 15->20")
    exec(src, globals())
exec(code[8], globals())                      # mean pooling cell (all 2,048 positions)
E_tut = pooled_embeddings.copy()
proc = ad.read_h5ad(f"{REPO}/data/processed/sample_data.h5ad", backed="r")
tut_ids = list(proc.obs_names[:N_CHECK]); tut_tokens = [set(x) - {tokenizer.pad_token_id} for x in ds["gene_ids"]]
print(f"tutorial: {E_tut.shape} embeddings on {device}; processed cells {proc.n_obs}; changes besides paths: {changed}")
del model, final_embeddings; gc.collect(); sys.path.remove(REPO); os.chdir("/tmp")

# COMMAND ----------

# DBTITLE 1,S1c: (b) this notebook's embedding code on raw counts, compare per cell
emb = load_embedder()
src_ad = ad.read_h5ad(f"{MODEL_DIR_ROOT}/data/sample_data.h5ad")
raw = src_ad.raw.to_adata()
assert np.allclose(raw.X[:50].toarray() % 1, 0), "raw.X is not integer counts"
sub = raw[tut_ids]
counts = sub.X.toarray(); genes = list(sub.var_names)
E_nb = emb.embed(counts, genes, pooling="tutorial")
ids_nb, mask_nb = emb.tokenize(counts, genes)
same = [set(ids_nb[i][mask_nb[i].bool()].tolist()) == tut_tokens[i] for i in range(N_CHECK)]
cos1 = (E_tut * E_nb).sum(1) / np.linalg.norm(E_tut, axis=1) / np.linalg.norm(E_nb, axis=1)
print(f"device (a)=cpu (b)={emb.device}; same gene set per cell {sum(same)}/{N_CHECK}")
print(f"per-cell cosine (a) vs (b): min {cos1.min():.6f}  median {np.median(cos1):.6f}  max {cos1.max():.6f}")
assert all(same) and cos1.min() >= 0.999, "S1 FAILED: notebook code does not reproduce the tutorial"
print("S1 PASS"); end("S1")

# COMMAND ----------

# DBTITLE 1,S2: sample brain reference and query cells from Census 2025-11-08
import cellxgene_census
start("S2")
CENSUS, EXCL_COLL, SEED, N_REF, N_Q = "2025-11-08", "b953c942-f5d8-434f-9da7-e726ba7c1481", 0, 125, 2
TYPES = ["oligodendrocyte", "astrocyte", "oligodendrocyte precursor cell", "microglial cell",
         "L2/3-6 intratelencephalic projecting glutamatergic neuron", "VIP GABAergic cortical interneuron",
         "pvalb GABAergic cortical interneuron", "sst GABAergic cortical interneuron", "lamp5 GABAergic cortical interneuron"]
REF_H5, Q_H5 = f"{SHARED}/ref_cells.h5ad", f"{SHARED}/query_cells.h5ad"
assert not os.path.exists(REF_H5) and not os.path.exists(Q_H5), "h5ad exists: not overwriting"
VF = ('is_primary_data == True and tissue_general == "brain" and assay == "10x 3\' v3" '
      f'and raw_sum >= 225 and cell_type in {TYPES}')                     # TEDDY QC part 1: >= 225 counts
rng = np.random.default_rng(SEED)
with cellxgene_census.open_soma(census_version=CENSUS) as census:
    dsets = census["census_info"]["datasets"].read().concat().to_pandas()
    excl = set(dsets.loc[dsets.collection_id == EXCL_COLL, "dataset_id"])
    hs = census["census_data"]["homo_sapiens"]
    obs = hs.obs.read(value_filter=VF, column_names=["soma_joinid", "cell_type", "dataset_id", "raw_sum"]).concat().to_pandas()
    obs["cell_type"] = obs.cell_type.astype(str); obs = obs[~obs.dataset_id.astype(str).isin(excl)]
    cand = obs.groupby("cell_type", group_keys=False).apply(lambda g: g.sample(400, random_state=SEED))  # oversample, QC, then sample
    A = cellxgene_census.get_anndata(census, "Homo sapiens", X_name="raw", obs_coords=cand.soma_joinid.sort_values().values,
                                     obs_column_names=["soma_joinid", "cell_type", "dataset_id", "raw_sum", "donor_id"])
mito = A.var.feature_name.str.startswith("MT-").values; assert mito.sum() == 37
A.obs["mito_frac"] = np.asarray(A.X[:, mito].sum(1)).ravel() / A.obs.raw_sum.values
A = A[A.obs.mito_frac <= 0.10].copy()                                    # TEDDY QC part 2: mito <= 10%
A.var_names = A.var.feature_id.values; A.var_names_make_unique()
A.obs["cell_id"] = [f"census{CENSUS}_{j}" for j in A.obs.soma_joinid]; A.obs_names = A.obs.cell_id.values
ref_ids, q_ids = [], []
for i, t in enumerate(TYPES):
    ids = rng.permutation(A.obs_names[A.obs.cell_type == t].values)
    assert len(ids) >= N_REF + N_Q, f"only {len(ids)} {t} cells pass QC"
    ref_ids += list(ids[:N_REF]); q_ids += list(ids[N_REF:N_REF + N_Q]) if i < 5 else []
prov = {"census_version": CENSUS, "value_filter": VF, "excluded_collection": EXCL_COLL, "excluded_datasets": sorted(excl),
        "qc": "raw_sum>=225; mito_frac<=0.10 over 37 MT- genes / raw_sum", "seed": SEED, "X": "raw counts (Census X raw)",
        "created_utc": now().isoformat(), "created_by": USER, "notebook": NB_PATH}
os.makedirs("/tmp/teddy_s2", exist_ok=True)
for ids, dst, role in [(ref_ids, REF_H5, "reference"), (q_ids, Q_H5, "query")]:
    s = A[ids].copy(); s.obs = s.obs[["cell_id", "cell_type", "dataset_id", "donor_id", "raw_sum", "mito_frac"]].astype({"cell_type": str, "dataset_id": str, "donor_id": str})
    s.uns["provenance"] = json.dumps({**prov, "role": role, "n_cells": s.n_obs})
    tmp = f"/tmp/teddy_s2/{os.path.basename(dst)}"; s.write_h5ad(tmp); shutil.copy(tmp, dst)
R, Q = ad.read_h5ad(REF_H5), ad.read_h5ad(Q_H5)
print(f"QC pass {A.n_obs}/{len(cand)} candidates; ref {R.shape}, query {Q.shape}; overlap {len(set(R.obs_names) & set(Q.obs_names))}")
print(R.obs.cell_type.value_counts().to_string()); print(Q.obs.cell_type.value_counts().to_string())
assert R.n_obs == 1125 and Q.n_obs == 10 and not set(R.obs_names) & set(Q.obs_names)
assert not set(R.obs.dataset_id) & excl and not set(Q.obs.dataset_id) & excl and (R.X.data % 1 == 0).all()
print(f"datasets: ref {R.obs.dataset_id.nunique()}, query {Q.obs.dataset_id.nunique()}; S2 PASS"); end("S2")

# COMMAND ----------

# DBTITLE 1,S3a: embed 1,125 reference cells (mean, float32) into teddy_ref_cells with CDF
import time
start("S3")
assert not spark.catalog.tableExists(REF_TABLE), f"{REF_TABLE} exists: not overwriting"
MODEL_VERSION = f"TEDDY-G-70M@{HF_REV}|pooling=mean"
R = ad.read_h5ad(REF_H5)
t0 = time.time()
E_ref = emb.embed(R.X.toarray(), list(R.var_names), pooling="mean")
assert E_ref.dtype == np.float32 and E_ref.shape == (1125, emb.config.d_model) and np.isfinite(E_ref).all()
spark.sql(f"CREATE TABLE {REF_TABLE} (cell_id STRING NOT NULL, embedding ARRAY<FLOAT>, cell_type STRING, model_version STRING) "
          "TBLPROPERTIES (delta.enableChangeDataFeed = true)")
pdf = pd.DataFrame({"cell_id": R.obs.cell_id.values, "embedding": [e.tolist() for e in E_ref],
                    "cell_type": R.obs.cell_type.values, "model_version": MODEL_VERSION})
spark.createDataFrame(pdf, schema="cell_id string, embedding array<float>, cell_type string, model_version string").write.mode("append").saveAsTable(REF_TABLE)
n = spark.table(REF_TABLE).count(); cdf = spark.sql(f"SHOW TBLPROPERTIES {REF_TABLE}").filter("key='delta.enableChangeDataFeed'").first().value
print(f"embedded {E_ref.shape} in {time.time()-t0:.0f}s; {REF_TABLE}: {n} rows, CDF={cdf}")
assert n == 1125 and cdf == "true"

# COMMAND ----------

# DBTITLE 1,S3b: the skill's scripts/vs_index.py, copied unchanged to /tmp
VS_INDEX_PY = r'''"""Create or reuse a Databricks AI Search endpoint and Delta Sync index for per-sample embeddings.

Databricks AI (artificial intelligence) Search is the product formerly called Vector Search; the
`databricks-sdk` still uses the `vector_search_*` names. These helpers are the tested code behind
`references/workflows/embeddings-to-vector-search.md` steps 0b, 3 and 4; the workflow page explains when
and why to call each one.

Rules this module keeps:

- **Read-only unless told otherwise.** Nothing is created unless the caller passes `run_go=True`
  (the user's confirmation, step 0b of the workflow).
- **Never deletes.** If the user chose *replace* for an existing index, the caller deletes it
  explicitly first (the workflow shows that one line); `ensure_index` refuses to touch it.
- **No guessing about sync state.** An existing index reports `ready=True` while it still serves its
  previous snapshot, so `sync_and_wait` tracks the sync pipeline update it started.

Importable without Spark, PyTorch or a GPU: every function takes the `WorkspaceClient` (`w`) and, where
SQL is needed, a `run_sql` callable such as `spark.sql`. Checked against `databricks-sdk` 0.122.0
(serverless environment v6) and the names used by the 0.67.0 / 0.157.0 runs the workflow cites.
"""
import time

from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service.pipelines import UpdateInfoState
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointStatusState,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

_DONE = {UpdateInfoState.COMPLETED}
_FAILED = {UpdateInfoState.FAILED, UpdateInfoState.CANCELED}
CHOICES = ("stop", "reuse", "replace")


# ---------------------------------------------------------------------------------------------------
# Unity Catalog (UC) names
# ---------------------------------------------------------------------------------------------------
def uc_name_parts(name):
    """Split a UC three-part name into lowercase parts, unquoting `...` identifiers (`` = one backtick)."""
    parts, buf, quoted, i = [], [], False, 0
    while i < len(name):
        ch = name[i]
        if ch == "`":
            if quoted and name[i + 1:i + 2] == "`":     # doubled backtick inside a quoted identifier
                buf.append("`")
                i += 2
                continue
            quoted = not quoted
        elif ch == "." and not quoted:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    if quoted or len(parts) != 3 or not all(parts):
        raise ValueError(f"Not a catalog.schema.name identifier: {name!r}")
    return tuple(p.lower() for p in parts)                 # UC identifiers are case-insensitive


def same_uc_name(api_name, ours):
    """Compare a name returned by the API with our configured name, component by component."""
    try:
        theirs = uc_name_parts(api_name)
    except (TypeError, ValueError):                         # None, or not a UC three-part name
        return False
    return theirs == uc_name_parts(ours)


def quote_uc_name(name):
    """Backtick-quote each part of a UC name for SQL (so hyphens and odd characters are safe)."""
    return ".".join("`" + p.replace("`", "``") + "`" for p in uc_name_parts(name))


# ---------------------------------------------------------------------------------------------------
# Step 0b — what already exists, and the one guard every later step goes through
# ---------------------------------------------------------------------------------------------------
def existing_resources(w, *, uc_model=None, table=None, index=None, vs_endpoint=None, table_rows=None):
    """Read-only lookup of resources under the proposed names. Returns a list of plain-language lines.

    table_rows: optional callable(table) -> row count (e.g. lambda t: spark.table(t).count()).
    """
    found = []
    if uc_model:
        # served_entities is the current field; served_models is the older one some endpoints still report.
        for e in w.serving_endpoints.list():
            cfg = e.config
            served = [(s.entity_name, s.entity_version) for s in (cfg.served_entities or [])] if cfg else []
            served += [(s.model_name, s.model_version) for s in (cfg.served_models or [])] if cfg else []
            for name, version in served:
                if same_uc_name(name, uc_model):
                    found.append(f"serving endpoint {e.name}: serves {name} v{version} "
                                 f"({len(served)} served entities), ready={e.state.ready if e.state else None}")
    if table and w.tables.exists(table).table_exists:
        rows = f"{table_rows(table)} rows, " if table_rows else ""
        cols = [c.name for c in (w.tables.get(table).columns or [])]
        found.append(f"table {table}: {rows}columns={cols}")
    if index:
        try:
            idx = w.vector_search_indexes.get_index(index)
            spec = idx.delta_sync_index_spec
            cols = (spec.embedding_vector_columns or []) if spec else []
            found.append(f"index {index}: endpoint={idx.endpoint_name}, "
                         f"source_table={spec.source_table if spec else None}, primary_key={idx.primary_key}, "
                         f"vector columns={[(c.name, c.embedding_dimension) for c in cols]}, "
                         f"ready={idx.status.ready if idx.status else None}")
        except NotFound:
            pass
    if vs_endpoint:
        try:
            ep = w.vector_search_endpoints.get_endpoint(vs_endpoint)
            n = len(list(w.vector_search_indexes.list_indexes(vs_endpoint)))
            found.append(f"AI Search endpoint {vs_endpoint}: {ep.endpoint_type}, "
                         f"{ep.endpoint_status.state if ep.endpoint_status else None}, {n} indexes")
        except NotFound:
            pass
    return found


def on_existing(kind, name, choice, details=""):
    """Return the user's choice for an existing resource; stop if none was made.

    choice: "stop" (default — raise and ask), "reuse" or "replace", set ONLY from the user's answer.
    """
    if choice not in CHOICES:
        raise ValueError(f"on_existing choice must be one of {CHOICES}, got {choice!r}")
    if choice == "stop":
        raise RuntimeError(f"{kind} {name} already exists {details}. Ask the user: reuse it, use a new "
                           "name, or replace it (workflow step 0b) — then pass their choice.")
    print(f"{kind} {name} exists → {choice} (confirmed by the user)")
    return choice


def index_mismatches(idx, *, table, vs_endpoint, emb_dim, primary_key="cell_id", vector_column="embedding",
                     required_columns=None):
    """Why an existing index cannot be reused for this run (an empty list means it can).

    required_columns: columns later queries ask for; on a STANDARD endpoint they must be in columns_to_sync.
    """
    spec = idx.delta_sync_index_spec
    if spec is None:
        return ["not a Delta Sync index"]
    cols = spec.embedding_vector_columns or []
    problems = [] if same_uc_name(spec.source_table, table) else [
        f"source_table={spec.source_table!r}, expected {table!r}"]
    want = {"endpoint": vs_endpoint, "primary_key": primary_key,
            "vector column (name, dim)": [(vector_column, emb_dim)]}
    got = {"endpoint": idx.endpoint_name, "primary_key": idx.primary_key,
           "vector column (name, dim)": [(c.name, c.embedding_dimension) for c in cols]}
    problems += [f"{k}={got[k]!r}, expected {want[k]!r}" for k in want if got[k] != want[k]]
    synced = spec.columns_to_sync
    if required_columns and synced is not None and not set(required_columns) <= set(synced) | {primary_key}:
        problems.append(f"columns_to_sync={synced!r} lacks {sorted(set(required_columns) - set(synced))}")
    return problems


def existing_dim(idx):
    """Embedding dimension of an existing index (None if it has no vector column)."""
    cols = (idx.delta_sync_index_spec.embedding_vector_columns or []) if idx.delta_sync_index_spec else []
    return cols[0].embedding_dimension if cols else None


# ---------------------------------------------------------------------------------------------------
# Waiting — every wait shares one absolute deadline (time.time() seconds)
# ---------------------------------------------------------------------------------------------------
def sleep_until(deadline, poll_s, what):
    """Sleep at most poll_s and never past `deadline`; raise once it has passed."""
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError(f"Deadline reached: {what}")
    time.sleep(min(poll_s, remaining))


def wait_until_ready(w, index_name, deadline, poll_s=15):
    """Poll until the index reports ready; raise at `deadline` instead."""
    while True:
        status = w.vector_search_indexes.get_index(index_name).status
        if status and status.ready:
            return status
        sleep_until(deadline, poll_s, f"{index_name} not ready: {status.message if status else 'no status'}")


def wait_until_queryable(w, index_name, emb_dim, deadline, primary_key="cell_id", poll_s=15):
    """Poll with a 1-result query until the index answers; raise at `deadline`.

    status.ready (and the row count) can be reported before queries work: a run on 2026-11-02 got
    "index is not ready" from query_index right after get_index said ready with every row. Only errors
    that say the index is not ready are retried; anything else propagates.
    """
    probe = [1.0] * emb_dim
    while True:
        try:
            w.vector_search_indexes.query_index(index_name=index_name, columns=[primary_key],
                                                query_vector=probe, num_results=1)
            return
        except Exception as e:                           # the SDK raises different classes for this
            if "not ready" not in str(e).lower():
                raise
            sleep_until(deadline, poll_s, f"{index_name} reports ready but does not answer queries yet")


def pipeline_updates(w, pipeline_id):
    """Every update of the index's sync pipeline (no order is documented, so read all pages)."""
    updates, token = [], None
    while True:
        try:
            page = w.pipelines.list_updates(pipeline_id, page_token=token)
        except PermissionDenied as e:
            raise RuntimeError(
                f"Confirming a fresh sync needs view access to the index's sync pipeline {pipeline_id}. "
                "Ask the index owner or the facilitator to grant it. Not falling back to status.ready, "
                "which can describe the previous snapshot.") from e
        updates += page.updates or []
        token = page.next_page_token
        if not token:
            return updates


def wait_for_row_count(w, index_name, expected, deadline, poll_s=15):
    """Poll until the ready index holds exactly `expected` rows; raise at `deadline`."""
    while True:
        status = w.vector_search_indexes.get_index(index_name).status
        if status and status.ready and status.indexed_row_count == expected:
            return status
        sleep_until(deadline, poll_s, f"{index_name} has {status.indexed_row_count if status else None} "
                                      f"rows, the table has {expected}")


def sync_and_wait(w, index_name, deadline, poll_s=15):
    """Request a sync and return only after THAT sync has completed and the index is ready.

    status.ready alone is not proof: an existing index reports ready=True while it still serves the
    previous snapshot. The SDK's index status has no sync timestamp, and the error for "a sync is
    already running" is not documented, so: wait for any running update to finish, let every
    sync_index error propagate, then track the pipeline update our request started until it completes.
    """
    pipeline_id = w.vector_search_indexes.get_index(index_name).delta_sync_index_spec.pipeline_id
    if not pipeline_id:
        raise RuntimeError(f"{index_name} reports no sync pipeline, so a fresh sync cannot be confirmed — "
                           "ask the user how to proceed")

    # 1. A sync that is already running may predate the rows just written: let it finish first.
    while any(u.state not in _DONE | _FAILED for u in pipeline_updates(w, pipeline_id)):
        sleep_until(deadline, poll_s, f"{index_name}: an earlier sync is still running")

    # 2. Request the sync. No error is suppressed.
    before = {u.update_id for u in pipeline_updates(w, pipeline_id)}
    w.vector_search_indexes.sync_index(index_name)
    print(f"Sync requested on {index_name}")

    # 3. Wait for the update this request started and require it to complete.
    while True:
        new = [u for u in pipeline_updates(w, pipeline_id) if u.update_id not in before]
        failed = [(u.update_id, u.state) for u in new if u.state in _FAILED]
        if failed:
            raise RuntimeError(f"{index_name}: requested sync did not complete: {failed}")
        if new and all(u.state in _DONE for u in new):
            break
        sleep_until(deadline, poll_s, f"{index_name}: the requested sync has not completed")

    return wait_until_ready(w, index_name, deadline)


# ---------------------------------------------------------------------------------------------------
# Steps 3 and 4 — create or reuse (creation only with run_go=True; never deletes)
# ---------------------------------------------------------------------------------------------------
def ensure_endpoint(w, name, *, endpoint_type="STANDARD", emb_dim=None, choice="stop", run_go=False):
    """Return an ONLINE AI Search endpoint `name`, creating it only when run_go is True.

    An existing endpoint is shared compute: it is reused (after the user's choice), never deleted or
    replaced here, and refused if it is the other endpoint type (the index settings depend on it).
    """
    try:
        requested = EndpointType(endpoint_type)                # enum, not a string
    except ValueError:
        raise ValueError(f"This databricks-sdk has no EndpointType {endpoint_type!r}; "
                         f"available: {[t.value for t in EndpointType]}. Upgrade with %pip.") from None
    if requested != EndpointType.STANDARD and emb_dim is not None and emb_dim % 16:
        raise ValueError(f"Storage-optimized endpoints need a dimension divisible by 16, got {emb_dim}")
    try:
        ep = w.vector_search_endpoints.get_endpoint(name)
        on_existing("AI Search endpoint", name, "reuse" if choice == "replace" else choice,
                    f"({ep.endpoint_type}, {ep.endpoint_status.state if ep.endpoint_status else None})")
    except NotFound:
        if not run_go:
            raise RuntimeError(f"AI Search endpoint {name} does not exist. Creating it starts billable compute "
                               "once it hosts an index; confirm with the user, then pass run_go=True.") from None
        print(f"Creating {requested.value} AI Search endpoint {name} (takes several minutes)...")
        ep = w.vector_search_endpoints.create_endpoint_and_wait(name=name, endpoint_type=requested)
    if ep.endpoint_type != requested:
        raise ValueError(f"{name} is {ep.endpoint_type}, but the requested type is {requested.value}. "
                         "Use a matching endpoint or change the type; the index spec depends on it.")
    if ep.endpoint_status is None or ep.endpoint_status.state != EndpointStatusState.ONLINE:
        ep = w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(name)
    print(f"AI Search endpoint {name} ONLINE")
    return ep


def ensure_index(w, index, *, endpoint, table, emb_dim, expected_rows, columns_to_sync, choice="stop",
                 run_go=False, primary_key="cell_id", vector_column="embedding", timeout_s=1800, poll_s=15):
    """Create (or reuse and re-sync) a triggered Delta Sync index; return its status once it holds every row.

    endpoint: the EndpointInfo from ensure_endpoint (its type decides whether columns_to_sync is allowed).
    expected_rows: the source table's row count, read by the caller after writing it.
    The index is never deleted here: for "replace", delete it first (the workflow shows how), then call again.
    """
    deadline = time.time() + timeout_s                  # one absolute deadline for every wait below
    try:
        idx = w.vector_search_indexes.get_index(index)
    except NotFound:
        idx = None

    if idx is not None:
        chosen = on_existing("Index", index, choice, f"(endpoint={idx.endpoint_name}, dim={existing_dim(idx)})")
        if chosen == "replace":
            raise RuntimeError(f"{index} still exists. The user chose replace: delete it explicitly first "
                               "(w.vector_search_indexes.delete_index), wait until it is gone, then call again.")
        problems = index_mismatches(idx, table=table, vs_endpoint=endpoint.name, emb_dim=emb_dim,
                                    primary_key=primary_key, vector_column=vector_column,
                                    required_columns=columns_to_sync)
        if problems:
            raise RuntimeError(f"{index} cannot be reused: {problems}. Ask the user for a new index name or replace.")
        sync_and_wait(w, index, deadline, poll_s)      # existing index: wait for the sync requested here
    else:
        if not run_go:
            raise RuntimeError(f"Index {index} does not exist. Confirm creating it with the user, then pass "
                               "run_go=True.")
        w.vector_search_indexes.create_index(
            name=index,
            endpoint_name=endpoint.name,
            primary_key=primary_key,
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=table,
                embedding_vector_columns=[EmbeddingVectorColumn(name=vector_column, embedding_dimension=emb_dim)],
                pipeline_type=PipelineType.TRIGGERED,
                # Storage-optimized endpoints reject columns_to_sync and sync every source column instead.
                columns_to_sync=list(columns_to_sync) if endpoint.endpoint_type == EndpointType.STANDARD else None,
            ),
        )
        print(f"Created {index}; the initial sync runs as part of creation")
        wait_until_ready(w, index, deadline, poll_s)   # a new index has no earlier snapshot to confuse

    status = wait_for_row_count(w, index, expected_rows, deadline, poll_s)
    wait_until_queryable(w, index, emb_dim, deadline, primary_key=primary_key, poll_s=poll_s)
    print(f"Index ready and answering queries: {status.indexed_row_count} rows, matching the table")
    return status


def grant_select(run_sql, objects, principal):
    """GRANT SELECT on tables and indexes to a user or group (idempotent).

    Querying an AI Search index needs USE CATALOG, USE SCHEMA and SELECT on the index itself; an index is
    a UC table-type securable, so the statement is GRANT ... ON TABLE <index>. run_sql: e.g. spark.sql.
    """
    for obj in objects:
        run_sql(f"GRANT SELECT ON TABLE {quote_uc_name(obj)} TO `{principal.replace('`', '``')}`")
        print(f"GRANT SELECT ON TABLE {obj} TO {principal}")


def index_sanity_check(w, index, *, columns, cell_id, vector, k=10):
    """After a rebuild: a cell already in the table must find itself first, with k rows, scores not increasing.

    columns: columns to return; must start with the primary key column (the score is appended last).
    """
    hits = w.vector_search_indexes.query_index(index_name=index, columns=list(columns),
                                               query_vector=[float(v) for v in vector], num_results=k)
    rows = (hits.result.data_array or []) if hits.result else []
    scores = [float(r[-1]) for r in rows]
    problems = []
    if len(rows) != k:
        problems.append(f"{len(rows)} rows returned, expected {k}")
    if rows and str(rows[0][0]) != str(cell_id):
        problems.append(f"top-1 is {rows[0][0]!r}, not the query cell {cell_id!r}")
    if any(b > a + 1e-9 for a, b in zip(scores, scores[1:])):
        problems.append(f"scores increase down the list: {scores}")
    if problems:
        raise RuntimeError(f"{index} failed the sanity check: {problems}")
    print(f"Sanity check passed: {cell_id} is its own top-1 (score {scores[0]:.4f}), {k} rows, scores not increasing")
    return scores
'''
os.makedirs("/tmp/teddy_vs", exist_ok=True)
open("/tmp/teddy_vs/vs_index.py", "w").write(VS_INDEX_PY)
if "/tmp/teddy_vs" not in sys.path: sys.path.insert(0, "/tmp/teddy_vs")
sys.modules.pop("vs_index", None); import vs_index
print("vs_index helpers:", [f for f in dir(vs_index) if f.startswith(("ensure", "index_"))])

# COMMAND ----------

# DBTITLE 1,S3c: create VS endpoint + Delta Sync index, check 20 cells find themselves
if "VS_TIMES" not in globals():                      # create only once per run (this run's own resources)
    VS_TIMES = {}
    print("existing:", vs_index.existing_resources(w, index=VS_INDEX, vs_endpoint=VS_ENDPOINT) or "none")
    t = time.time()
    vs_ep = vs_index.ensure_endpoint(w, VS_ENDPOINT, endpoint_type="STANDARD", emb_dim=E_ref.shape[1], run_go=True)  # user approved in chat
    VS_TIMES["endpoint_create_to_online_min"] = (time.time() - t) / 60
    n_rows = spark.table(REF_TABLE).count()
    T_INDEX0 = time.time()
    vs_index.ensure_index(w, VS_INDEX, endpoint=vs_ep, table=REF_TABLE, emb_dim=E_ref.shape[1], expected_rows=n_rows,
                          columns_to_sync=["cell_id", "cell_type", "model_version"], run_go=True, timeout_s=2700)
    VS_TIMES["index_create_to_queryable_min"] = (time.time() - T_INDEX0) / 60

def check_with_retry(cid, vec, deadline):
    while True:                                      # a new index can flap to "not ready" right after its first answer
        try: return vs_index.index_sanity_check(w, VS_INDEX, columns=["cell_id", "cell_type"], cell_id=cid, vector=vec, k=10)
        except Exception as e:
            if "not ready" not in str(e).lower(): raise
            vs_index.sleep_until(deadline, 15, f"{VS_INDEX} not ready for self-retrieval")
tab = spark.table(REF_TABLE).toPandas().set_index("cell_id")
pick = tab.sample(20, random_state=1)
dl = time.time() + 900
for cid, row in pick.iterrows():
    check_with_retry(cid, row.embedding, dl)
VS_TIMES["index_create_to_20_checks_min"] = (time.time() - T_INDEX0) / 60
print(f"20/20 cells are their own top-1. Endpoint create->ONLINE {VS_TIMES['endpoint_create_to_online_min']:.1f} min; "
      f"index create->first query answered {VS_TIMES['index_create_to_queryable_min']:.1f} min; "
      f"->20 checks passed {VS_TIMES['index_create_to_20_checks_min']:.1f} min")
end("S3")

# COMMAND ----------

# DBTITLE 1,S4a: register shared/teddy_70m as teddy_model_shared (express, env_pack), check the loaded version
import mlflow, importlib.metadata as im
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import DataType
from mlflow.utils.env_pack import EnvPackConfig
start("S4")
assert not any(m.full_name == UC_MODEL for m in w.registered_models.list(catalog_name=CATALOG, schema_name=SCHEMA)), f"{UC_MODEL} exists"
mlflow.set_registry_uri("databricks-uc")
def cos(A, B): A, B = np.asarray(A), np.asarray(B); return (A * B).sum(1) / np.linalg.norm(A, axis=1) / np.linalg.norm(B, axis=1)
def records_for(adata, **controls):            # one row = one batch of cells; all genes (library size uses every gene)
    return [{"adata_sparsematrix": adata.X.toarray().tolist(),
             "adata_obs": pd.DataFrame({"cell_id": list(adata.obs_names)}).to_json(orient="split"),
             "adata_var": pd.DataFrame({"index": list(adata.var_names)}).to_json(orient="split"), **controls}]

c0 = R[:1]; nz = np.asarray(c0.X.toarray()[0] > 0)       # one-cell example, its expressed genes only (~UI-friendly size)
ex_df = pd.DataFrame(records_for(c0[:, nz], pooling="mean", max_seq_len="2048"))
default_params = {"max_seq_len": "2048", "pooling": "mean"}
out_ex = pd.DataFrame({"embedding": [[0.0] * emb.config.d_model]}).assign(pooling="mean", max_seq_len="2048")
sig0 = infer_signature(ex_df, out_ex, params=default_params)
signature = ModelSignature(inputs=Schema([ColSpec(c.type, c.name, required=c.name not in default_params) for c in sig0.inputs.inputs]),
                           outputs=sig0.outputs, params=sig0.params)
assert {p.name: p.dtype for p in signature.params.params} == {"max_seq_len": DataType.string, "pooling": DataType.string}
PKGS = ["mlflow", "databricks-sdk", "transformers", "torch", "numpy", "pandas", "scipy", "anndata", "scanpy", "scikit-learn", "datasets", "tqdm"]
PIP_REQS = [f"{p}=={im.version(p).split('+')[0]}" for p in PKGS]
TAGS = {"hf_repo": HF_REPO, "hf_revision": HF_REV, "variant": "70M", "license_model_card": "apache-2.0",
        "license_package": "MIT", "paper": "arXiv:2503.03485", "default_pooling": "mean", "deployment": "express"}
t = time.time()
with mlflow.start_run(run_name="teddy_model_shared") as run:
    mlflow.set_tags(TAGS)
    mlflow.log_dict(manifest, "provenance/manifest.json")
    info = mlflow.pyfunc.log_model(name="model", python_model=WRAPPER_PATH, artifacts=ARTIFACTS, signature=signature,
                                   input_example=(ex_df, default_params), pip_requirements=PIP_REQS)
mv = mlflow.register_model(info.model_uri, UC_MODEL, env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=True))
for k, v in TAGS.items(): mlflow.MlflowClient().set_model_version_tag(UC_MODEL, mv.version, k, v)
print(f"registered {UC_MODEL} v{mv.version} in {time.time()-t:.0f}s; reqs {PIP_REQS}")

loaded = mlflow.pyfunc.load_model(f"models:/{UC_MODEL}/{mv.version}")
check = R[pick.index.tolist()]
E_loaded = np.array(loaded.predict(pd.DataFrame(records_for(check)))["embedding"].tolist())
c_tab = cos(E_loaded, np.stack(pick.embedding.values))
print(f"loaded v{mv.version} vs table, 20 cells: min cosine {c_tab.min():.6f}")
assert c_tab.min() >= 0.999, "S4 FAILED: registered model does not match the table"
del loaded; gc.collect(); torch.cuda.empty_cache()

# COMMAND ----------

# DBTITLE 1,S4b: serve as teddy_ep_shared (GPU_MEDIUM, scale to zero, AI Gateway inference table), wait for READY
from databricks.sdk.service.serving import (EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType,
                                            EndpointStateReady, EndpointStateConfigUpdate, EndpointTag,
                                            AiGatewayConfig, AiGatewayUsageTrackingConfig, AiGatewayInferenceTableConfig)
from databricks.sdk.errors import NotFound
try: w.serving_endpoints.get(SERVE_EP); raise RuntimeError(f"{SERVE_EP} exists: stop and ask")
except NotFound: pass
entity = ServedEntityInput(name=f"teddy-70m-v{mv.version}", entity_name=UC_MODEL, entity_version=str(mv.version),
                           workload_type=ServingModelWorkloadType.GPU_MEDIUM, workload_size="Small", scale_to_zero_enabled=True)
PAYLOAD_PREFIX = SERVE_EP                                     # inference table -> f"{FQ}.{SERVE_EP}_payload"
gateway = AiGatewayConfig(usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
                          inference_table_config=AiGatewayInferenceTableConfig(enabled=True, catalog_name=CATALOG,
                                                                               schema_name=SCHEMA, table_name_prefix=PAYLOAD_PREFIX))
T_EP0 = time.time()
w.serving_endpoints.create(name=SERVE_EP, config=EndpointCoreConfigInput(name=SERVE_EP, served_entities=[entity]),
                           ai_gateway=gateway,
                           tags=[EndpointTag(key="deployment", value="express"), EndpointTag(key="hf_revision", value=HF_REV)])
while True:
    ep = w.serving_endpoints.get(SERVE_EP)
    if ep.state.config_update == EndpointStateConfigUpdate.UPDATE_FAILED:
        raise RuntimeError(f"S4 FAILED: endpoint update failed: {ep.state}")
    if ep.state.ready == EndpointStateReady.READY and ep.state.config_update == EndpointStateConfigUpdate.NOT_UPDATING: break
    assert time.time() - T_EP0 < 45 * 60, "S4 FAILED: endpoint not READY after 45 min"
    time.sleep(20)
EP_READY_MIN = (time.time() - T_EP0) / 60
assert ep.ai_gateway and ep.ai_gateway.usage_tracking_config.enabled and ep.ai_gateway.inference_table_config.enabled, \
    "S4 FAILED: AI Gateway usage tracking / inference table not on"
assert w.tables.exists(f"{FQ}.{PAYLOAD_PREFIX}_payload").table_exists, "S4 FAILED: inference table not created"
print(f"{SERVE_EP} READY, serving {UC_MODEL} v{mv.version}; create->READY {EP_READY_MIN:.1f} min")
print(f"AI Gateway: usage tracking on, inference table {FQ}.{PAYLOAD_PREFIX}_payload (rows can take up to an hour)")

# COMMAND ----------

# DBTITLE 1,S4c: endpoint matches the table; a non-default setting changes the output
def invoke(records, params=None):
    body = {"dataframe_records": records, **({"params": params} if params else {})}
    return w.api_client.do("POST", f"/serving-endpoints/{SERVE_EP}/invocations", body=body)["predictions"]
def via_endpoint(adata, batch=10, **controls):
    preds = []
    for s in range(0, adata.n_obs, batch):
        preds += w.serving_endpoints.query(name=SERVE_EP, dataframe_records=records_for(adata[s:s + batch], **controls)).predictions
    assert len(preds) == adata.n_obs
    return np.array([p["embedding"] for p in preds]), {p["pooling"] for p in preds}
E_ep, pool_used = via_endpoint(check)                                  # default controls
c_ep = cos(E_ep, np.stack(pick.embedding.values))
print(f"endpoint (default, echo {pool_used}) vs table, 20 cells: min {c_ep.min():.6f} median {np.median(c_ep):.6f}")
E_first, pool_first = via_endpoint(check, pooling="first")             # non-default, column route
c_first = cos(E_first, E_ep)
p = invoke(records_for(check[:10]), params={"max_seq_len": "32"})     # non-default, params route
c_32 = cos(np.array([x["embedding"] for x in p]), E_ep[:10])
print(f"pooling=first (column, echo {pool_first}): median cosine to default {np.median(c_first):.3f}; "
      f"max_seq_len=32 (params, echo {p[0]['max_seq_len']}): median {np.median(c_32):.3f}")
assert pool_used == {"mean"} and c_ep.min() >= 0.999, "S4 FAILED: endpoint does not match the table"
assert pool_first == {"first"} and c_first.max() < 0.99 and p[0]["max_seq_len"] == "32" and c_32.max() < 0.99, "S4 FAILED: control not applied"
print(f"S4 PASS; create->READY {EP_READY_MIN:.1f} min"); end("S4")

# COMMAND ----------

# DBTITLE 1,S5a: the tutorial's steps on all sample cells (GPU)
start("S5")
VAL_JSON = f"{SHARED}/tutorial_validation.json"
assert not os.path.exists(VAL_JSON), f"{VAL_JSON} exists: not overwriting"
REPO5 = "/tmp/teddy_s5/repo"
if os.path.exists(REPO5): shutil.rmtree(REPO5)                 # scratch copy this run made
shutil.copytree(MODEL_DIR_ROOT, REPO5, ignore=shutil.ignore_patterns("manifest.json", ".download_complete"))
code5 = ["".join(c["source"]) for c in json.load(open(f"{REPO5}/tutorials/model_tutorial.ipynb"))["cells"] if c["cell_type"] == "code"]
os.chdir(REPO5)
for k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]: del sys.modules[k]
sys.path.insert(0, REPO5)
EDITS = {"max_eval_samples=15": "max_eval_samples=10**9", "batch_size=1": "batch_size=16",
         'device = torch.device("cpu")': 'device = torch.device("cuda"); model.to(device)'}
t = time.time(); applied = []
for src in code5[1:9]:                                      # imports .. mean pooling, as in S1b
    for a, b in EDITS.items():
        if a in src: src = src.replace(a, b); applied.append(f"{a} -> {b}")
    exec(src, globals())
E_tut5 = pooled_embeddings.copy()
proc5 = ad.read_h5ad(f"{REPO5}/data/processed/sample_data.h5ad", backed="r"); ids5 = list(proc5.obs_names)
print(f"tutorial: {E_tut5.shape} in {time.time()-t:.0f}s; changes besides paths: {applied}")
assert E_tut5.shape[0] == len(ids5) == 3538
del model, final_embeddings, all_embeddings; gc.collect(); torch.cuda.empty_cache(); sys.path.remove(REPO5); os.chdir("/tmp")

# COMMAND ----------

# DBTITLE 1,S5b: same cells through teddy_ep_shared (pooling=tutorial), compare, figure, save result
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
raw5 = ad.read_h5ad(f"{MODEL_DIR_ROOT}/data/sample_data.h5ad").raw.to_adata()[ids5]   # .raw.X integer counts
chunks = [raw5[s:s + 10] for s in range(0, raw5.n_obs, 10)]
t = time.time()
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(lambda a: via_endpoint(a, pooling="tutorial"), chunks))
E_ep5 = np.concatenate([r[0] for r in res]); echo = set().union(*[r[1] for r in res])
c5 = cos(E_tut5, E_ep5)
print(f"endpoint: {E_ep5.shape} in {time.time()-t:.0f}s, echo {echo}; cosine min {c5.min():.6f} median {np.median(c5):.6f}; "
      f"cells below 0.999: {(c5 < 0.999).sum()}")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.scatter(np.arange(len(c5)), c5, s=4); ax.axhline(0.999, color="red", ls="--", label="0.999")
ax.set(xlabel="cell (tutorial order)", ylabel="cosine (tutorial vs endpoint)", ylim=(min(0.998, c5.min() - 1e-4), 1.0001),
       title=f"TEDDY-70M tutorial vs {SERVE_EP}, {len(c5)} cells"); ax.legend(); plt.tight_layout(); plt.show()
result = {"endpoint": SERVE_EP, "model": f"{UC_MODEL}/{mv.version}", "revision": HF_REV, "cells": int(len(c5)),
          "pooling": "tutorial", "min_cosine": float(c5.min()), "median_cosine": float(np.median(c5)), "threshold": 0.999,
          "result": "PASS" if c5.min() >= 0.999 and echo == {"tutorial"} else "FAIL", "utc": now().isoformat()}
json.dump(result, open(VAL_JSON, "w"), indent=1)
print(json.load(open(VAL_JSON)))
assert result["result"] == "PASS", "S5 FAILED"
end("S5")

# COMMAND ----------

# DBTITLE 1,Summary: names created, step times, what is still billing
display(inventory())
print(pd.DataFrame([{"step": s, "start_utc": f"{v['start']:%Y-%m-%d %H:%M:%S}", "end_utc": f"{v['end']:%Y-%m-%d %H:%M:%S}",
                     "minutes": round((v["end"] - v["start"]).total_seconds() / 60, 1)} for s, v in STEPS.items()]).to_string(index=False))
print(f"VS: endpoint create->ONLINE {VS_TIMES['endpoint_create_to_online_min']:.1f} min, index create->answering "
      f"{VS_TIMES['index_create_to_queryable_min']:.1f} min; serving create->READY {EP_READY_MIN:.1f} min")
se = w.serving_endpoints.get(SERVE_EP); vse = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
print(f"still billing: serving endpoint {SERVE_EP} ({se.state.ready.value}, GPU_MEDIUM, scale to zero after idle); "
      f"Vector Search endpoint {VS_ENDPOINT} ({vse.endpoint_status.state.value}, STANDARD, bills while it exists); "
      f"storage for {REF_TABLE}, index, model and volume files")

# COMMAND ----------

# DBTITLE 1,Demo: query teddy_ref_index with the 10 query cells (embedded by teddy_ep_shared)
import anndata as ad, numpy as np
from collections import Counter
Q = ad.read_h5ad(f"{SHARED}/query_cells.h5ad")
def one_cell(i):                       # expressed genes only: zeros change neither the total nor top-K (small request)
    x = Q.X[i].toarray()[0]; nz = x > 0
    return [{"adata_sparsematrix": [x[nz].tolist()],
             "adata_obs": pd.DataFrame({"cell_id": [Q.obs_names[i]]}).to_json(orient="split"),
             "adata_var": pd.DataFrame({"index": list(Q.var_names[nz])}).to_json(orient="split")}]
preds = [w.serving_endpoints.query(name=SERVE_EP, dataframe_records=one_cell(i)).predictions[0] for i in range(Q.n_obs)]  # default pooling=mean, as the index
assert len(preds) == Q.n_obs and {p["pooling"] for p in preds} == {"mean"}
K, rows = 10, []
for (cid, true), p in zip(Q.obs[["cell_type"]].itertuples(), preds):
    hits = w.vector_search_indexes.query_index(index_name=VS_INDEX, columns=["cell_id", "cell_type"],
                                               query_vector=p["embedding"], num_results=K).result.data_array
    assert len(hits) == K and cid not in {h[0] for h in hits}          # query cells are not in the index
    vote, n = Counter(h[1] for h in hits).most_common(1)[0]
    rows.append({"cell_id": cid, "true_type": true, "vote_type": vote, "vote_share": n / K,
                 "same_type_neighbours": sum(h[1] == true for h in hits) / K, "top1_score": round(float(hits[0][-1]), 4),
                 "correct": vote == true})
res = pd.DataFrame(rows); display(res)
print(f"{res.correct.sum()}/{len(res)} query cells get the right type by a {K}-neighbour vote; mean same-type share "
      f"{res.same_type_neighbours.mean():.2f} (chance 125/1125 = 0.11 per type)")
# cross-check: exact L2 search over the table gives the same votes (index score = 1/(1+L2^2))
tab = spark.table(REF_TABLE).toPandas(); X = np.stack(tab.embedding.values); P = np.array([p["embedding"] for p in preds])
d2 = ((P[:, None, :] - X[None]) ** 2).sum(-1); nn = np.argsort(d2, 1)[:, :K]
exact = [Counter(tab.cell_type.values[r]).most_common(1)[0][0] for r in nn]
print(f"exact search agrees with the index on {sum(a == b for a, b in zip(exact, res.vote_type))}/{len(res)} votes; "
      f"top-1 1/(1+d2) = {[round(float(1 / (1 + d2[i, nn[i, 0]])), 4) for i in range(3)]} vs index {res.top1_score.tolist()[:3]}")