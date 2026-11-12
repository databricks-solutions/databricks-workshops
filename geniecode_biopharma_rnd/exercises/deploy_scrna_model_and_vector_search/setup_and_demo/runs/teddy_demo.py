# Databricks notebook source
# DBTITLE 1,TEDDY-70M demo
# MAGIC %md
# MAGIC # TEDDY-70M demo: deploy, index, label
# MAGIC Prompt 1 packages TEDDY-70M from Hugging Face (reusing Setup's copy only after every file passes its checksum) and starts an express-deployment serving endpoint; prompt 2 builds a Vector Search index of the reference brain cells, and prompt 3 labels 10 unseen cells by a vote of their 10 nearest labelled reference cells.

# COMMAND ----------

# DBTITLE 1,Install packages (one install, one restart)
# MAGIC %pip install -q "mlflow>=3.12" "databricks-sdk>=0.102.0" "transformers==4.41.0" anndata umap-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config and workspace client
import os, sys, json, time, datetime as dt
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
USER = w.current_user.me().user_name                      # read at run time
NB_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
NB_DIR = os.path.dirname(NB_PATH)                         # workspace folder of this notebook

CATALOG, SCHEMA, VOLUME = "main", "owm", "teddy_files"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SHARED_SNAP = f"{VOLUME_PATH}/shared/teddy_70m"           # Setup's copy (read-only)
DEMO_DIR    = f"{VOLUME_PATH}/demo"
DEMO_SNAP   = f"{DEMO_DIR}/teddy_70m"                     # used only if the shared copy fails a check
QUERY_H5AD  = f"{VOLUME_PATH}/shared/query_cells.h5ad"

HF_REPO, HF_URL = "Merck/TEDDY", "https://huggingface.co/Merck/TEDDY"
HF_REVISION = "8c880aa67674042ccdc254bccb5de081092a48bf"
VARIANT = "70M"

UC_MODEL    = f"{CATALOG}.{SCHEMA}.teddy_model_demo"
ENDPOINT    = "teddy_ep_demo"
INF_PREFIX  = "teddy_ep_demo"                             # inference tables in CATALOG.SCHEMA
EXPERIMENT  = f"{NB_DIR}/teddy_demo_experiment"
REF_TABLE   = f"{CATALOG}.{SCHEMA}.teddy_ref_cells"
CELLS_TABLE = f"{CATALOG}.{SCHEMA}.teddy_demo_cells"
INDEX       = f"{CATALOG}.{SCHEMA}.teddy_demo_index"
VS_ENDPOINT = "teddy_vs_workshop"

SKILL_DIR = "/Workspace/.assistant/skills/open-weight-models"
RUN_DIR   = "/tmp/teddy_demo"
os.makedirs(RUN_DIR, exist_ok=True)
STEP_TIMES = {}                                           # step -> seconds
print(f"user={USER}\nnotebook={NB_PATH}\nexperiment={EXPERIMENT}\nvolume={VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Check compute: serverless GPU A10, torch importable
import subprocess
info = {"python": sys.version.split()[0]}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except ImportError as e:
    info["torch"] = f"NOT importable: {e}"
    info["cuda_available"], info["gpu"] = False, None
info["env_vars"] = {k: v for k, v in os.environ.items()
                    if k.startswith(("DATABRICKS_RUNTIME", "DATABRICKS_SERVERLESS", "DB_ENV")) or "ENVIRONMENT_VERSION" in k}
print(json.dumps(info, indent=1))
ok = bool(info["cuda_available"]) and "A10" in (info["gpu"] or "")
assert ok, ("STOP: this notebook is not on serverless GPU A10 (AI environment v3+). "
            "Switch compute to Serverless GPU -> A10, then re-run from the install cell.")
print("OK: GPU visible and torch importable ->", info["gpu"])

# COMMAND ----------

# DBTITLE 1,Check names are free (overwrite nothing)
import mlflow
from databricks.sdk.errors import NotFound
mlflow.set_registry_uri("databricks-uc")

taken = []
try:
    w.serving_endpoints.get(ENDPOINT); taken.append(f"serving endpoint {ENDPOINT}")
except NotFound:
    pass
try:
    w.registered_models.get(UC_MODEL); taken.append(f"model {UC_MODEL}")
except NotFound:
    pass
existing_inf = [t.name for t in w.tables.list(CATALOG, SCHEMA) if t.name.startswith(INF_PREFIX + "_")]
if existing_inf:
    taken.append(f"inference tables {existing_inf}")
if os.path.exists(DEMO_SNAP):
    taken.append(f"folder {DEMO_SNAP}")

exp = mlflow.get_experiment_by_name(EXPERIMENT)
if exp is not None and exp.lifecycle_stage == "deleted":
    raise RuntimeError(f"STOP and ask: experiment {EXPERIMENT} was deleted (it is in the trash).")
if exp is None:
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT)
    print("created experiment", EXPERIMENT, EXPERIMENT_ID)
else:
    EXPERIMENT_ID = exp.experiment_id
    print("experiment exists and is active; runs are added to it:", EXPERIMENT, EXPERIMENT_ID)
mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

assert not taken, f"STOP and ask: these names are taken: {taken}"
print("All names free:", ENDPOINT, UC_MODEL, f"{INF_PREFIX}_*")

# COMMAND ----------

# DBTITLE 1,Get TEDDY-70M files: verify Setup's copy or download
import hashlib, shutil, urllib.request
t0 = time.time()

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def git_sha1_file(p):
    h = hashlib.sha1(f"blob {os.path.getsize(p)}\0".encode())
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def check_manifest(snap):
    """Return (ok, manifest, problems). Checks every file listed against its sha256 or git-sha1."""
    problems = []
    if not os.path.exists(f"{snap}/.download_complete"):
        return False, None, [".download_complete missing"]
    if not os.path.exists(f"{snap}/manifest.json"):
        return False, None, ["manifest.json missing"]
    with open(f"{snap}/manifest.json") as f:
        man = json.load(f)
    if man.get("revision") != HF_REVISION:
        return False, man, [f"revision {man.get('revision')} != {HF_REVISION}"]
    n_sha256 = n_git = 0
    for e in man["files"]:
        p = f"{snap}/{e['path']}"
        if not os.path.exists(p):
            problems.append(f"missing {e['path']}"); continue
        if "bytes" in e and os.path.getsize(p) != e["bytes"]:
            problems.append(f"size {e['path']}"); continue
        if "sha256" in e:
            n_sha256 += 1
            if sha256_file(p) != e["sha256"]: problems.append(f"sha256 {e['path']}")
        elif "git-sha1" in e:
            n_git += 1
            if git_sha1_file(p) != e["git-sha1"]: problems.append(f"git-sha1 {e['path']}")
        else:
            problems.append(f"no checksum {e['path']}")
    print(f"checked {len(man['files'])} files: {n_sha256} sha256, {n_git} git-sha1; problems: {len(problems)}")
    return not problems, man, problems

ok, man, problems = check_manifest(SHARED_SNAP)
if ok:
    SNAP = SHARED_SNAP
    COPY_USED = "shared/teddy_70m (Setup's copy; every manifest file matched its checksum; used read-only)"
    MANIFEST = man
else:
    print("Shared copy not usable:", problems[:10])
    # Download the 70M file list (skip 160M/400M) to demo/teddy_70m, checking each against HF checksums
    assert not os.path.exists(DEMO_SNAP), f"STOP and ask: {DEMO_SNAP} already exists"
    api = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{HF_REPO}/revision/{HF_REVISION}?blobs=true"))
    skip = ("teddy/models/teddy_g/160M/", "teddy/models/teddy_g/400M/")
    files, skipped = [], []
    tmp = f"{RUN_DIR}/dl"
    for s in api["siblings"]:
        rel = s["rfilename"]
        if rel.startswith(skip) or rel.endswith(".DS_Store"):
            skipped.append(rel); continue
        local = f"{tmp}/{rel}"; os.makedirs(os.path.dirname(local), exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{rel}", local)
        e = {"path": rel, "bytes": os.path.getsize(local)}
        if s.get("lfs"):
            e["sha256"] = sha256_file(local); assert e["sha256"] == s["lfs"]["sha256"], rel
        else:
            e["git-sha1"] = git_sha1_file(local); assert e["git-sha1"] == s["blobId"], rel
        files.append(e)
    MANIFEST = {"source": HF_URL, "revision": HF_REVISION, "variant": VARIANT,
                "license_model_card": ["apache-2.0"], "license_package_pyproject": ["MIT"],
                "acquired_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "acquired_by": USER,
                "skipped": skipped, "files": files}
    shutil.copytree(tmp, DEMO_SNAP)
    with open(f"{DEMO_SNAP}/manifest.json", "w") as f: json.dump(MANIFEST, f, indent=1)
    with open(f"{DEMO_SNAP}/.download_complete", "w") as f: f.write(MANIFEST["acquired_utc"])
    SNAP = DEMO_SNAP
    COPY_USED = "demo/teddy_70m (fresh download, every file checked against Hugging Face checksums)"

MODEL_DIR = f"{SNAP}/teddy/models/teddy_g/{VARIANT}"
for fn in ("config.json", "model.safetensors", "vocab.txt"):
    assert os.path.exists(f"{MODEL_DIR}/{fn}"), f"missing {MODEL_DIR}/{fn}"
STEP_TIMES["1a. files (checksum check / download)"] = time.time() - t0
print("Using:", COPY_USED, "\nMODEL_DIR =", MODEL_DIR, f"\n{STEP_TIMES['1a. files (checksum check / download)']:.0f} s")

# COMMAND ----------

# DBTITLE 1,Provenance record
def _lic(v):
    return ", ".join(v) if isinstance(v, list) else str(v)
weights = next(e for e in MANIFEST["files"] if e["path"].endswith(f"{VARIANT}/model.safetensors"))
PROVENANCE = {
    "source": HF_URL,
    "revision": HF_REVISION,
    "variant": VARIANT,
    "licence_model_card": _lic(MANIFEST.get("license_model_card", "apache-2.0")),
    "licence_package_pyproject": _lic(MANIFEST.get("license_package_pyproject", "MIT")),
    "licence_note": "two declarations (model card Apache-2.0, package MIT); confirm with the model owner before redistribution",
    "copy_used": COPY_USED,
    "snapshot_path": SNAP,
    "weights_sha256": weights.get("sha256"),
    "files_in_manifest": len(MANIFEST["files"]),
    "acquired_utc": MANIFEST.get("acquired_utc"),
    "recorded_by": USER,
    "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
print(json.dumps(PROVENANCE, indent=1))

# COMMAND ----------

# DBTITLE 1,Stage code bundle and write the PyFunc wrapper file
CODE_DIR = f"{RUN_DIR}/teddy_code"
WRAPPER_PATH = f"{RUN_DIR}/teddy_wrapper.py"
if os.path.exists(CODE_DIR):
    shutil.rmtree(CODE_DIR)                      # /tmp scratch made by this run
shutil.copytree(f"{SNAP}/teddy", f"{CODE_DIR}/teddy",
                ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.pt", "__pycache__"))
# The wrapper reads the gene medians from the bundle at load time: fail here, not in the serving container
assert os.path.exists(f"{CODE_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json"), \
    "teddy_gene_medians.json missing from the code bundle"

WRAPPER_SRC = r'''
import sys, os, io, json, inspect
import numpy as np
import pandas as pd
import torch
import mlflow

CONTROLS = {"pooling": ("mean", {"mean", "tutorial", "first"}), "max_seq_len": ("2048", None)}
BATCH = 10   # cells per forward pass (A10/A10G at 2,048 genes)
REQUIRED_COLUMNS = ("adata_sparsematrix", "adata_var")


def resolve_controls(row, columns, params):
    used = {}
    for name, (default, allowed) in CONTROLS.items():
        if name in columns and row[name] is not None and not (isinstance(row[name], float) and np.isnan(row[name])):
            v = row[name]
        else:
            v = (params or {}).get(name, default)
        v = str(v)
        if allowed and v not in allowed:
            raise ValueError(f"{name}={v!r} is not supported; use one of {sorted(allowed)}")
        if name == "max_seq_len" and (not v.isdigit() or int(v) < 1):
            raise ValueError(f"max_seq_len={v!r} must be a positive integer string")
        used[name] = v
    return used


class TEDDYEmbedder(mlflow.pyfunc.PythonModel):
    """TEDDY-G 70M cell embeddings. Raw counts in; upstream normalisation, median scaling, top-K in the wrapper."""

    def load_context(self, context):
        import sys, os, json, inspect, torch
        for _m in list(sys.modules):
            if _m == "teddy" or _m.startswith("teddy."):
                del sys.modules[_m]
        pkg = context.artifacts["teddy_pkg_parent"]
        if pkg in sys.path:
            sys.path.remove(pkg)
        sys.path.insert(0, pkg)
        from teddy.models.model_directory import get_architecture, model_dict
        from teddy.tokenizer.gene_tokenizer import GeneTokenizer
        model_dir = context.artifacts["model_dir"]
        arch = get_architecture(model_dir)
        self.config = model_dict[arch]["config_cls"].from_pretrained(model_dir)
        self.model = model_dict[arch]["model_cls"].from_pretrained(model_dir, config=self.config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.model.return_all_embs = True
        self.tokenizer = GeneTokenizer.from_pretrained(model_dir)
        self._forward_params = set(inspect.signature(self.model.forward).parameters)
        self.add_cls = bool(getattr(self.config, "add_cls", False))
        if self.add_cls:
            raise NotImplementedError("this wrapper is for TEDDY-G 70M (add_cls=False)")
        self.d_model = int(getattr(self.config, "d_model", 0))
        with open(os.path.join(pkg, "teddy/data_processing/utils/medians/data/teddy_gene_medians.json")) as f:
            self.medians = json.load(f)
        print(f"TEDDY-G 70M loaded on {self.device}, d_model={self.d_model}")

    def _tokenize(self, X, gene_names, max_seq_len):
        X = np.asarray(X, dtype=np.float32)
        tot = X.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        X = X / tot * 1e4
        X = X * np.array([1.0 / self.medians[g] if g in self.medians else 1.0 for g in gene_names],
                         dtype=np.float32)[None, :]
        keep = [i for i, g in enumerate(gene_names) if g in self.tokenizer.vocab]
        if not keep:
            raise ValueError("No gene IDs in adata_var match the TEDDY vocabulary (expects Ensembl IDs, e.g. ENSG...)")
        token_array = torch.tensor([self.tokenizer.vocab[gene_names[i]] for i in keep])
        X_t = torch.from_numpy(np.ascontiguousarray(X[:, keep], dtype=np.float32))
        k = min(int(max_seq_len), X_t.shape[1])
        top_vals, top_idx = torch.topk(X_t, k=k, largest=True, sorted=True)
        gene_ids = token_array[top_idx].clone()
        expressed = top_vals > 0
        gene_ids[~expressed] = self.tokenizer.pad_token_id
        return gene_ids, expressed.long()

    def _embed(self, X, gene_names, pooling, max_seq_len):
        # Tokenize one batch at a time so memory stays flat for large requests (each cell is independent)
        out_all = []
        for s in range(0, X.shape[0], BATCH):
            ids, m = self._tokenize(X[s:s + BATCH], gene_names, max_seq_len)
            if pooling == "tutorial" and ids.shape[1] < int(max_seq_len):
                pad = int(max_seq_len) - ids.shape[1]
                ids = torch.cat([ids, torch.full((ids.shape[0], pad), self.tokenizer.pad_token_id, dtype=ids.dtype)], 1)
                m = torch.cat([m, torch.zeros((m.shape[0], pad), dtype=m.dtype)], 1)
            fwd = {"gene_ids": ids.to(self.device), "attention_mask": m.to(self.device)}
            fwd = {k: v for k, v in fwd.items() if k in self._forward_params}
            with torch.no_grad():
                out = self.model(**fwd)
            if pooling == "first":
                emb = out["cell_emb"]
            else:
                H = out.get("all_embs")              # explicit None check; never truth-test a tensor
                if H is None:
                    raise RuntimeError("model output has no all_embs; return_all_embs was not honoured")
                if pooling == "mean":
                    M = fwd["attention_mask"].unsqueeze(-1).to(H.dtype)
                    emb = (H * M).sum(1) / M.sum(1).clamp(min=1)
                else:
                    emb = H.mean(1)
            out_all.append(emb.float().cpu().numpy())
        return np.concatenate(out_all, 0)

    def predict(self, context, model_input, params=None):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"request is missing required column(s) {missing}")
        rows = []
        for _, row in df.iterrows():
            used = resolve_controls(row, df.columns, params)
            var_df = pd.read_json(io.StringIO(row["adata_var"]), orient="split")
            genes = [str(g) for g in (var_df["index"] if "index" in var_df.columns else var_df.index)]
            X = np.array([np.asarray(r, dtype=np.float32) for r in row["adata_sparsematrix"]], dtype=np.float32)
            if X.ndim != 2 or X.shape[1] != len(genes):
                raise ValueError(f"adata_sparsematrix shape {X.shape} does not match {len(genes)} genes in adata_var")
            E = self._embed(X, genes, used["pooling"], used["max_seq_len"])
            for e in E:
                rows.append({"embedding": e.tolist(), **used})
        return pd.DataFrame(rows)


mlflow.models.set_model(TEDDYEmbedder())
'''
with open(WRAPPER_PATH, "w") as f:
    f.write(WRAPPER_SRC)
print("code bundle:", CODE_DIR, "| wrapper:", WRAPPER_PATH)

# COMMAND ----------

# DBTITLE 1,Dry-load the wrapper and run validation checks
import importlib.util, types
import numpy as np, pandas as pd, anndata as ad
t0 = time.time()
for _k in [k for k in sys.modules if k in ("teddy_wrapper",) or k == "teddy" or k.startswith("teddy.")]:
    del sys.modules[_k]
spec = importlib.util.spec_from_file_location("teddy_wrapper", WRAPPER_PATH)
tw = importlib.util.module_from_spec(spec); spec.loader.exec_module(tw)
wrapper = tw.TEDDYEmbedder()
wrapper.load_context(types.SimpleNamespace(artifacts={"model_dir": MODEL_DIR, "teddy_pkg_parent": CODE_DIR}))

# 1. vocabulary smoke test
with open(f"{MODEL_DIR}/vocab.txt") as f:
    vocab_lines = [t.strip() for t in f]
n_ens = sum(t.startswith("ENSG") for t in vocab_lines)
print("vocab lines:", len(vocab_lines), "Ensembl IDs:", n_ens, "ENSG00000000003 present:", "ENSG00000000003" in vocab_lines)
assert wrapper.d_model == json.load(open(f"{MODEL_DIR}/config.json"))["d_model"]

# Real cells: upstream sample data, raw counts
src = ad.read_h5ad(f"{SNAP}/data/sample_data.h5ad")
src = src.raw.to_adata() if src.raw is not None else src
cells = src[:5].copy()
counts = cells.X.toarray() if hasattr(cells.X, "toarray") else np.asarray(cells.X)
genes = [str(g) for g in cells.var_names]
assert genes[0].startswith("ENSG"), genes[:3]
var_json = pd.DataFrame({"index": genes}).to_json(orient="split")
obs_json = pd.DataFrame({"cell_id": list(cells.obs_names)}).to_json(orient="split")
req = pd.DataFrame({"adata_sparsematrix": [counts.tolist()], "adata_obs": [obs_json], "adata_var": [var_json]})
default_params = {"max_seq_len": "2048", "pooling": "mean"}

def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
out = wrapper.predict(None, req, params=None)
E = np.array(out["embedding"].tolist())
assert E.shape == (5, wrapper.d_model) and np.all(np.linalg.norm(E, axis=1) > 0) and E.std() > 0
assert set(out["pooling"]) == {"mean"}
# batch invariance
alone = np.array(wrapper.predict(None, req.assign(adata_sparsematrix=[counts[:1].tolist()]))["embedding"][0])
print("batch invariance cosine:", round(cos(alone, E[0]), 7))
# controls switch the output (column route and params route)
for ctl in ({"pooling": "first"}, {"pooling": "tutorial"}, {"max_seq_len": "32"}):
    by_col = np.array(wrapper.predict(None, req.assign(**{k: [v] for k, v in ctl.items()}))["embedding"].tolist())
    by_par = np.array(wrapper.predict(None, req, params=ctl)["embedding"].tolist())
    print(ctl, "min cosine to default (column, params):",
          round(min(cos(a, b) for a, b in zip(by_col, E)), 4), round(min(cos(a, b) for a, b in zip(by_par, E)), 4))
# all-OOV rejected
try:
    wrapper.predict(None, req.assign(adata_var=[pd.DataFrame({"index": [f"x{i}" for i in range(len(genes))]}).to_json(orient="split")]))
    raise AssertionError("all-OOV request was not rejected")
except ValueError as e:
    print("all-OOV rejected:", e)
STEP_TIMES["1b. dry-load + checks"] = time.time() - t0

# COMMAND ----------

# DBTITLE 1,Upstream-parity test (wrapper vs TEDDY's own preprocessing)
# Upstream modules import scanpy at module level; install it to a side folder (not packed with the model)
try:
    import scanpy  # noqa
except ImportError:
    PARITY_PKGS = f"{RUN_DIR}/parity_pkgs"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", PARITY_PKGS,
                           "scanpy==1.11.2", "legacy-api-wrap", "session-info2", "natsort"])
    sys.path.insert(0, PARITY_PKGS)
from scipy.sparse import csr_matrix
import torch

for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]:
    del sys.modules[_k]
sys.path.insert(0, CODE_DIR)
from teddy.data_processing.preprocessing.preprocess import normalize_data_inplace, scale_columns_by_median_dict
from teddy.data_processing.tokenization.tokenization import _check_genes_in_tokenizer, _build_batch_tensors
from teddy.tokenizer.gene_tokenizer import GeneTokenizer

layer = csr_matrix(counts.astype(np.float32))
normalize_data_inplace(layer, 10000)
scale_columns_by_median_dict(layer, cells, f"{CODE_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json", "index")
tok = GeneTokenizer.from_pretrained(MODEL_DIR)
in_vocab, coding_genes, _ = _check_genes_in_tokenizer(cells, "index", tok)
token_array = torch.tensor(tok.encode(coding_genes.tolist(), add_special_tokens=False))
args = types.SimpleNamespace(max_seq_len=2048, add_cls=False, random_genes=False)
up_ids, up_vals, _, _ = _build_batch_tensors(torch.tensor(layer.toarray()[:, in_vocab], dtype=torch.float), token_array, args)

w_ids, w_mask = wrapper._tokenize(counts.astype(np.float32), genes, 2048)
E_first = np.array(wrapper.predict(None, req, params={"pooling": "first"})["embedding"].tolist())
parity = []
for i in range(len(cells)):
    ids = up_ids[i][up_vals[i] != 0].unsqueeze(0).to(wrapper.device)
    up_row, w_row = ids[0].cpu(), w_ids[i][w_mask[i].bool()].cpu()
    same = set(w_row.tolist()) == set(up_row.tolist())
    with torch.no_grad():
        o = wrapper.model(gene_ids=ids, attention_mask=torch.ones_like(ids))
    ref_first = o["cell_emb"][0].float().cpu().numpy()
    ref_mean = o["all_embs"][0].mean(0).float().cpu().numpy()
    parity.append((i, len(up_row), len(w_row), same, cos(ref_first, E_first[i]), cos(ref_mean, E[i])))
    print(f"cell {i}: tokens upstream={len(up_row)} wrapper={len(w_row)} same genes={same} "
          f"cos(first)={parity[-1][4]:.7f} cos(mean)={parity[-1][5]:.7f}")
assert all(p[3] and p[4] >= 0.9999 and p[5] >= 0.9999 for p in parity), "wrapper does not reproduce upstream TEDDY"
PARITY_MIN_COS = min(min(p[4], p[5]) for p in parity)
print("Upstream parity PASS, min cosine", round(PARITY_MIN_COS, 7))

# COMMAND ----------

# DBTITLE 1,Log to MLflow and register with express deployment (env_pack)
from importlib.metadata import version as _v
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import DataType
from mlflow.utils.env_pack import EnvPackConfig
t0 = time.time()

# One-cell input example with only that cell's expressed genes (keeps the serving UI example small)
nz = np.nonzero(counts[0])[0]
ex_genes = [genes[j] for j in nz]
input_example = pd.DataFrame({
    "adata_sparsematrix": [[counts[0, nz].astype(float).tolist()]],
    "adata_obs": [pd.DataFrame({"cell_id": [str(cells.obs_names[0])]}).to_json(orient="split")],
    "adata_var": [pd.DataFrame({"index": ex_genes}).to_json(orient="split")],
    "pooling": ["mean"], "max_seq_len": ["2048"],
})
ex_out = wrapper.predict(None, input_example)
print("one-cell example vs full-gene cell, cosine:", round(cos(np.array(ex_out["embedding"][0]), E[0]), 6))
sig = infer_signature(input_example, ex_out, params=default_params)
signature = ModelSignature(
    inputs=Schema([ColSpec(c.type, c.name, required=c.name not in default_params) for c in sig.inputs.inputs]),
    outputs=sig.outputs, params=sig.params)
assert {p.name: p.dtype for p in signature.params.params} == {"max_seq_len": DataType.string, "pooling": DataType.string}

PIP_REQS = [f"mlflow=={_v('mlflow')}", "transformers==4.41.0", f"torch=={torch.__version__.split('+')[0]}",
            f"numpy=={np.__version__}", f"pandas=={pd.__version__}"]
TAGS = {"hf_source": HF_URL, "hf_revision": HF_REVISION, "variant": VARIANT,
        "licence_model_card": PROVENANCE["licence_model_card"], "licence_package": PROVENANCE["licence_package_pyproject"],
        "copy_used": COPY_USED[:250], "paper": "arXiv:2503.03485", "pooling_default": "mean", "deployment": "express"}

with mlflow.start_run(run_name="teddy_demo_register") as run:
    mlflow.set_tags(TAGS)
    mlflow.log_dict(PROVENANCE, "provenance.json")
    mlflow.log_dict(MANIFEST, "manifest.json")
    mlflow.log_metric("parity_min_cosine", PARITY_MIN_COS)
    info = mlflow.pyfunc.log_model(name="model", python_model=WRAPPER_PATH,
                                   artifacts={"model_dir": MODEL_DIR, "teddy_pkg_parent": CODE_DIR},
                                   signature=signature, input_example=input_example, pip_requirements=PIP_REQS)
RUN_ID = run.info.run_id
try:
    mv = mlflow.register_model(info.model_uri, UC_MODEL,
                               env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=True))
except Exception as e:
    print("env_pack with install_dependencies=True failed, retrying once with False:", str(e)[:300])
    mv = mlflow.register_model(info.model_uri, UC_MODEL,
                               env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False))
MODEL_VERSION = str(mv.version)
client = mlflow.MlflowClient()
for k, v in TAGS.items():
    client.set_model_version_tag(UC_MODEL, MODEL_VERSION, k, v)
STEP_TIMES["1c. log + register (env_pack)"] = time.time() - t0
print(f"registered {UC_MODEL} v{MODEL_VERSION} | run {RUN_ID} | {STEP_TIMES['1c. log + register (env_pack)']:.0f} s")

# COMMAND ----------

# DBTITLE 1,Check the registered version matches the notebook wrapper
t0 = time.time()
loaded = mlflow.pyfunc.load_model(f"models:/{UC_MODEL}/{MODEL_VERSION}")
print("params schema:", {p.name: str(p.dtype) for p in loaded.metadata.signature.params.params})
E_reg = np.array(loaded.predict(req)["embedding"].tolist())
reg_cos = min(cos(a, b) for a, b in zip(E_reg, E))
E_reg_first = np.array(loaded.predict(req, params={"pooling": "first"})["embedding"].tolist())
print("registered vs wrapper, min cosine (mean):", round(reg_cos, 7),
      "| params route changes output (first vs mean):", round(min(cos(a, b) for a, b in zip(E_reg_first, E)), 4))
assert reg_cos >= 0.9999
del loaded; torch.cuda.empty_cache()
STEP_TIMES["1d. registered-version check"] = time.time() - t0

# COMMAND ----------

# DBTITLE 1,Start serving endpoint (paid compute, gated; does not wait)
from databricks.sdk.service.serving import (EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType,
    AiGatewayConfig, AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig, EndpointTag)
dbutils.widgets.dropdown("run_go", "false", ["false", "true"], "Run costly steps")
RUN_GO = dbutils.widgets.get("run_go") == "true"
assert RUN_GO, "run_go is false: set the widget to true (after confirming paid compute) and re-run this cell"

# re-check the name right before create (overwrite nothing)
try:
    w.serving_endpoints.get(ENDPOINT); raise RuntimeError(f"STOP and ask: endpoint {ENDPOINT} now exists")
except NotFound:
    pass
served = ServedEntityInput(name=f"teddy-70m-v{MODEL_VERSION}", entity_name=UC_MODEL, entity_version=MODEL_VERSION,
                           workload_type=ServingModelWorkloadType.GPU_MEDIUM, workload_size="Small",
                           scale_to_zero_enabled=True)
ai_gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(catalog_name=CATALOG, schema_name=SCHEMA,
                                                         table_name_prefix=INF_PREFIX, enabled=True),
    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))
ENDPOINT_CREATED_AT = time.time()
w.serving_endpoints.create(name=ENDPOINT,
                           config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=[served]),
                           ai_gateway=ai_gateway,
                           tags=[EndpointTag(key="deployment", value="express"), EndpointTag(key="hf_revision", value=HF_REVISION),
                                 EndpointTag(key="created_by", value=USER)])   # not waiting
ep = w.serving_endpoints.get(ENDPOINT)
print(f"Created {ENDPOINT} at {dt.datetime.now(dt.timezone.utc):%H:%M:%S} UTC (not waiting): state={ep.state.ready}, "
      f"config_update={ep.state.config_update}")
print("AI Gateway:", ep.ai_gateway)

# COMMAND ----------

# DBTITLE 1,Prompt 2: index
# MAGIC %md
# MAGIC ## Prompt 2: index
# MAGIC Copy the reference cells to `teddy_demo_cells` (Change Data Feed on) and build Delta Sync index `teddy_demo_index` on the shared Vector Search endpoint `teddy_vs_workshop` with the skill's `scripts/vs_index.py`.

# COMMAND ----------

# DBTITLE 1,Copy reference table with Change Data Feed on
t0 = time.time()
assert not spark.catalog.tableExists(CELLS_TABLE), f"STOP and ask: {CELLS_TABLE} already exists"
spark.sql(f"""CREATE TABLE {CELLS_TABLE}
              TBLPROPERTIES (delta.enableChangeDataFeed = true)
              AS SELECT * FROM {REF_TABLE}""")
N_REF = spark.table(REF_TABLE).count()
N_CELLS = spark.table(CELLS_TABLE).count()
cdf = spark.sql(f"SHOW TBLPROPERTIES {CELLS_TABLE} ('delta.enableChangeDataFeed')").collect()[0]["value"]
EMB_DIM = len(spark.table(CELLS_TABLE).select("embedding").first()[0])
assert N_CELLS == N_REF and cdf == "true" and EMB_DIM == wrapper.d_model
STEP_TIMES["2a. copy table (CDF on)"] = time.time() - t0
print(f"{CELLS_TABLE}: {N_CELLS} rows (source {N_REF}), CDF={cdf}, embedding dim={EMB_DIM}")
display(spark.table(CELLS_TABLE).groupBy("cell_type", "model_version").count().orderBy("cell_type"))

# COMMAND ----------

# DBTITLE 1,Build Delta Sync index with vs_index.py and time it
sys.path.insert(0, f"{SKILL_DIR}/scripts")
import vs_index
assert RUN_GO, "run_go is false"

print("\n".join(vs_index.existing_resources(w, index=INDEX, vs_endpoint=VS_ENDPOINT)))
# The user named teddy_vs_workshop: reuse it (shared; never modified). A taken index name stops here.
vs_ep = vs_index.ensure_endpoint(w, VS_ENDPOINT, endpoint_type="STANDARD", emb_dim=EMB_DIM, choice="reuse")
INDEX_CREATED_AT = time.time()
status = vs_index.ensure_index(w, INDEX, endpoint=vs_ep, table=CELLS_TABLE, emb_dim=EMB_DIM,
                               expected_rows=N_CELLS, columns_to_sync=["cell_id", "cell_type", "model_version"],
                               choice="stop", run_go=RUN_GO)
INDEX_READY_S = time.time() - INDEX_CREATED_AT
STEP_TIMES["2b. index create -> answering queries"] = INDEX_READY_S

# Sanity: a cell from the table finds itself first with 10 rows, scores not increasing
r0 = spark.table(CELLS_TABLE).orderBy("cell_id").select("cell_id", "embedding").first()
vs_index.index_sanity_check(w, INDEX, columns=["cell_id", "cell_type"], cell_id=r0["cell_id"], vector=r0["embedding"], k=10)
print(f"Index {INDEX}: {status.indexed_row_count} rows; create -> answering queries in {INDEX_READY_S/60:.1f} min")

# COMMAND ----------

# DBTITLE 1,Prompt 3: label
# MAGIC %md
# MAGIC ## Prompt 3: label
# MAGIC Send the 10 unseen cells in `shared/query_cells.h5ad` through `teddy_ep_demo`, look each up in `teddy_demo_index` (10 neighbours) and label it by majority vote. Neighbour agreement is a vote share, not a calibrated probability; this is in-silico prioritisation, not a measurement.

# COMMAND ----------

# DBTITLE 1,Wait for the endpoint to be READY and check what it serves
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
import nn_annotate as nn
import visualize as viz
t0 = time.time()
first_check_min = None
while True:
    ep = w.serving_endpoints.get(ENDPOINT)
    mins = (time.time() - ENDPOINT_CREATED_AT) / 60
    if first_check_min is None:
        first_check_min = mins
    if ep.state.ready == EndpointStateReady.READY and ep.state.config_update == EndpointStateConfigUpdate.NOT_UPDATING:
        break
    if ep.state.config_update == EndpointStateConfigUpdate.UPDATE_FAILED:
        raise RuntimeError(f"{ENDPOINT} update failed; read the service logs (teddy.md Troubleshooting #10)")
    assert mins < 40, f"{ENDPOINT} not READY after {mins:.0f} min"
    print(f"{mins:.1f} min: {ep.state.ready}, {ep.state.config_update}"); time.sleep(30)
ENDPOINT_READY_MIN = mins
READY_AT_FIRST_CHECK = abs(mins - first_check_min) < 1e-9
ep_info, TIER = nn.serving_endpoint_check(w, ENDPOINT, uc_model=UC_MODEL, uc_model_version=MODEL_VERSION)
print(f"{ENDPOINT} READY {mins:.1f} min after create" + (" (at the first check, so an upper bound)" if READY_AT_FIRST_CHECK else "")
      + f"; serves {UC_MODEL} v{MODEL_VERSION} on {TIER}, 100% traffic")
STEP_TIMES["3a. wait for endpoint READY (this cell)"] = time.time() - t0

# COMMAND ----------

# DBTITLE 1,Embed the 10 query cells through the endpoint and vote over 10 neighbours
t0 = time.time()
qa = ad.read_h5ad(QUERY_H5AD)
vocab = set(wrapper.tokenizer.vocab)
gene_mask = np.array([g in vocab for g in qa.var_names])
qa = qa[:, gene_mask].copy()
Q_GENES = list(qa.var_names)
QX = qa.X.toarray().astype(np.float32)
Q_IDS = list(qa.obs["cell_id"].astype(str))
Q_TRUE = list(qa.obs["cell_type"].astype(str))
print(f"{len(Q_IDS)} query cells, {len(Q_GENES)} vocabulary genes sent (of {len(gene_mask)})")
assert not set(Q_IDS) & set(spark.table(CELLS_TABLE).select("cell_id").toPandas()["cell_id"]), "a query cell is in the index"

PARAMS = {"pooling": "mean", "max_seq_len": "2048"}   # same model and pooling as the reference table
q_emb, embed_s, search_s, hits_all = [], [], [], []
for i, cid in enumerate(Q_IDS):                       # one cell per call, to time each
    s = time.time()
    e = nn.embed_via_endpoint(w, ENDPOINT, QX[i:i + 1], [cid], Q_GENES, params=PARAMS, tier=TIER)[0]
    embed_s.append(time.time() - s)
    s = time.time()
    hits = nn.query_neighbours(w, INDEX, e, columns=["cell_id", "cell_type"], k=10)
    search_s.append(time.time() - s)
    q_emb.append(e); hits_all.append(hits)
Q_EMB = np.array(q_emb)

# served = notebook wrapper on the same cells
local = np.array(wrapper.predict(None, pd.DataFrame({"adata_sparsematrix": [QX.tolist()], "adata_obs": ["{}"],
        "adata_var": [pd.DataFrame({"index": Q_GENES}).to_json(orient="split")]}))["embedding"].tolist())
SERVED_VS_WRAPPER = min(cos(a, b) for a, b in zip(Q_EMB, local))

ref_counts = spark.table(CELLS_TABLE).groupBy("cell_type").count().toPandas().set_index("cell_type")["count"]
N_INDEX = int(ref_counts.sum())
rows = []
for cid, t, hits, es, ss in zip(Q_IDS, Q_TRUE, hits_all, embed_s, search_s):
    v = nn.vote(hits)
    n_agree = sum(h["cell_type"] == v["predicted"] for h in hits)
    rows.append({"cell_id": cid, "true type": t, "predicted type": v["predicted"], "correct": v["predicted"] == t,
                 "neighbours agreeing (of 10)": n_agree, "top-1 score": round(v["top1_score"], 4),
                 "chance (of 10)": round(10 * ref_counts.get(t, 0) / N_INDEX, 2),
                 "model call (s)": round(es, 2), "index query (s)": round(ss, 2), "total (s)": round(es + ss, 2)})
RESULTS = pd.DataFrame(rows)
N_CORRECT = int(RESULTS["correct"].sum())
STEP_TIMES["3b. embed 10 cells + 10 index lookups"] = time.time() - t0
print(f"{N_CORRECT} of 10 correct | served vs notebook wrapper min cosine {SERVED_VS_WRAPPER:.6f} | "
      f"median model call {np.median(embed_s):.2f} s, median index query {np.median(search_s):.2f} s")
display(RESULTS)

# COMMAND ----------

# DBTITLE 1,Restore RESULTS from saved cell 19 output (only if not in memory)
# Cell 19 needs the endpoint and teddy_demo_index (index since torn down). If RESULTS is not in memory,
# rebuild it from cell 19's saved output (run 2026-11-10 05:59) so the figures below can be redrawn.
import os, sys, numpy as np, pandas as pd
if "RESULTS" not in globals():
    SKILL_DIR = globals().get("SKILL_DIR", "/Workspace/.assistant/skills/open-weight-models")
    RUN_DIR = globals().get("RUN_DIR", "/tmp/teddy_demo"); os.makedirs(RUN_DIR, exist_ok=True)
    sys.path.insert(0, f"{SKILL_DIR}/scripts")
    cols = ["cell_id", "true type", "predicted type", "correct", "neighbours agreeing (of 10)", "top-1 score",
            "chance (of 10)", "model call (s)", "index query (s)", "total (s)"]
    L = "L2/3-6 intratelencephalic projecting glutamatergic neuron"
    saved = [
        ("census2025-11-08_148884695", "oligodendrocyte", "oligodendrocyte", True, 10, 0.0335, 1.11, 0.57, 0.16, 0.73),
        ("census2025-11-08_73751002", "oligodendrocyte", "oligodendrocyte", True, 10, 0.0186, 1.11, 0.22, 0.13, 0.35),
        ("census2025-11-08_15016054", "astrocyte", "astrocyte", True, 10, 0.0189, 1.11, 0.21, 0.09, 0.30),
        ("census2025-11-08_149875496", "astrocyte", "astrocyte", True, 10, 0.0257, 1.11, 0.22, 0.17, 0.39),
        ("census2025-11-08_158740574", "oligodendrocyte precursor cell", "oligodendrocyte precursor cell", True, 10, 0.0265, 1.11, 0.22, 0.10, 0.31),
        ("census2025-11-08_125500046", "oligodendrocyte precursor cell", "oligodendrocyte precursor cell", True, 10, 0.0253, 1.11, 0.19, 0.14, 0.33),
        ("census2025-11-08_156653410", "microglial cell", "microglial cell", True, 10, 0.0198, 1.11, 0.22, 0.08, 0.30),
        ("census2025-11-08_126272216", "microglial cell", "microglial cell", True, 10, 0.0161, 1.11, 0.19, 0.08, 0.27),
        ("census2025-11-08_156780580", L, L, True, 10, 0.0065, 1.11, 0.20, 0.09, 0.29),
        ("census2025-11-08_158217014", L, L, True, 8, 0.0045, 1.11, 0.20, 0.10, 0.30),
    ]
    RESULTS = pd.DataFrame(saved, columns=cols)
    N_CORRECT = int(RESULTS["correct"].sum())
    print(f"RESULTS restored from saved cell 19 output: {len(RESULTS)} rows, {N_CORRECT} correct")
else:
    print("RESULTS already in memory; nothing restored")
import visualize as viz

# COMMAND ----------

# DBTITLE 1,Reconstruct all variables from saved outputs (endpoint/index torn down)
# Reconstruct every variable the report cell needs from the saved cell outputs.
# The endpoint and index were torn down, so Q_EMB is recomputed with the local TEDDY model.
import os, sys, json, shutil
import numpy as np, pandas as pd, torch

# --- Config (cell 3) ---
CATALOG, SCHEMA, VOLUME = "main", "owm", "teddy_files"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SHARED_SNAP = f"{VOLUME_PATH}/shared/teddy_70m"
DEMO_DIR    = f"{VOLUME_PATH}/demo"
QUERY_H5AD  = f"{VOLUME_PATH}/shared/query_cells.h5ad"
HF_URL      = "https://huggingface.co/Merck/TEDDY"
HF_REVISION = "8c880aa67674042ccdc254bccb5de081092a48bf"
UC_MODEL    = f"{CATALOG}.{SCHEMA}.teddy_model_demo"
ENDPOINT    = "teddy_ep_demo"
VS_ENDPOINT = "teddy_vs_workshop"
INDEX       = f"{CATALOG}.{SCHEMA}.teddy_demo_index"
CELLS_TABLE = f"{CATALOG}.{SCHEMA}.teddy_demo_cells"
NB_PATH     = "/Shared/exercises/deploy_scrna_model_and_vector_search/setup_and_demo/runs/teddy_demo"
USER        = spark.sql("SELECT current_user()").first()[0]
RUN_DIR     = "/tmp/teddy_demo"
SKILL_DIR   = "/Workspace/.assistant/skills/open-weight-models"
FIG_DIR     = f"{RUN_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, f"{SKILL_DIR}/scripts")
import visualize as viz

# --- Run-specific scalars (from saved cell outputs) ---
MODEL_VERSION       = "1"
N_CELLS             = 1125
EMB_DIM             = 512
N_CORRECT           = 10
INDEX_READY_S        = 34.7
ENDPOINT_READY_MIN   = 8.5
READY_AT_FIRST_CHECK = True
SERVED_VS_WRAPPER    = 1.000000
PARITY_MIN_COS       = 0.9999992
COPY_USED            = "shared/teddy_70m (Setup's copy; every manifest file matched its checksum; used read-only)"

PROVENANCE = {
    "source": HF_URL, "revision": HF_REVISION, "variant": "70M",
    "licence_model_card": "apache-2.0", "licence_package_pyproject": "MIT",
    "licence_note": "two declarations (model card Apache-2.0, package MIT); confirm with the model owner before redistribution",
    "copy_used": COPY_USED, "snapshot_path": SHARED_SNAP,
    "weights_sha256": "ba0a98f9b975dc42a1a2d384e9821e4f7cce795ba4bb17d2dfb940829438b0f3",
    "files_in_manifest": 62,
    "acquired_utc": "2026-11-09T17:15:53.993052+00:00",
    "recorded_by": USER, "recorded_utc": "2026-11-10T05:48:53.007129+00:00",
}

# --- STEP_TIMES / STEPS (cell 25 saved output) ---
STEP_TIMES = {
    "1a. files (checksum check / download)": 8.3, "1b. dry-load + checks": 9.7,
    "1c. log + register (env_pack)": 39.4, "1d. registered-version check": 20.3,
    "1e. endpoint create -> READY (min, upper bound)": 507.7,
    "2a. copy table (CDF on)": 5.2, "2b. index create -> answering queries": 34.7,
    "3a. wait for endpoint READY (this cell)": 0.1, "3b. embed 10 cells + 10 index lookups": 5.4,
    "3c. UMAP": 28.7, "3d. interneuron leave-one-out (20 lookups)": 2.4,
}
STEPS = pd.DataFrame([{"step": k, "seconds": round(v, 1), "minutes": round(v / 60, 2)}
                       for k, v in sorted(STEP_TIMES.items())])

# --- CREATED (cell 25 saved output) ---
CREATED = pd.DataFrame([
    {"resource": "Serving endpoint (GPU_MEDIUM, scale to zero)", "name": "teddy_ep_demo",
     "made by": "this notebook (prompt 1)", "state now": "READY, DEPLOYMENT_READY",
     "still billing": "yes, GPU while scaled up; scales to zero after about 30 min idle",
     "how to stop": "delete endpoint teddy_ep_demo (Teardown prompt)"},
    {"resource": "Vector Search index (Delta Sync, triggered)", "name": "main.owm.teddy_demo_index",
     "made by": "this notebook (prompt 2)", "state now": "ready=True, 1125 rows",
     "still billing": "yes, as part of shared endpoint teddy_vs_workshop (which bills while it exists)",
     "how to stop": "delete index main.owm.teddy_demo_index"},
    {"resource": "Vector Search endpoint (shared)", "name": "teddy_vs_workshop",
     "made by": "Setup (not this notebook)", "state now": "ONLINE",
     "still billing": "yes (shared; keep)", "how to stop": "not by this demo"},
    {"resource": "Delta table (CDF on)", "name": "main.owm.teddy_demo_cells",
     "made by": "this notebook (prompt 2)", "state now": "1125 rows",
     "still billing": "storage only", "how to stop": "DROP TABLE main.owm.teddy_demo_cells"},
    {"resource": "Inference / usage tables", "name": "teddy_ep_demo_payload",
     "made by": "endpoint AI Gateway", "state now": "exists",
     "still billing": "storage only", "how to stop": "drop after deleting the endpoint"},
    {"resource": "UC model + version", "name": "main.owm.teddy_model_demo v1",
     "made by": "this notebook (prompt 1)", "state now": "registered (env_pack)",
     "still billing": "storage only", "how to stop": "delete version, then model"},
    {"resource": "MLflow experiment",
     "name": "/Shared/exercises/deploy_scrna_model_and_vector_search/setup_and_demo/runs/teddy_demo_experiment",
     "made by": "this notebook (prompt 1)", "state now": "one run logged",
     "still billing": "no", "how to stop": "delete experiment"},
    {"resource": "Notebook compute (serverless GPU A10)", "name": "this notebook",
     "made by": "you", "state now": "attached",
     "still billing": "yes while attached / running", "how to stop": "detach or let it idle out"},
])

# --- Mock idx (index torn down; report reads idx.status.indexed_row_count) ---
class _MockIdx:
    class status:
        indexed_row_count = 1125
idx = _MockIdx()

# --- RESULTS (cell 20) ---
L = "L2/3-6 intratelencephalic projecting glutamatergic neuron"
_cols = ["cell_id", "true type", "predicted type", "correct",
         "neighbours agreeing (of 10)", "top-1 score", "chance (of 10)",
         "model call (s)", "index query (s)", "total (s)"]
RESULTS = pd.DataFrame([
    ("census2025-11-08_148884695", "oligodendrocyte", "oligodendrocyte", True, 10, 0.0335, 1.11, 0.57, 0.16, 0.73),
    ("census2025-11-08_73751002", "oligodendrocyte", "oligodendrocyte", True, 10, 0.0186, 1.11, 0.22, 0.13, 0.35),
    ("census2025-11-08_15016054", "astrocyte", "astrocyte", True, 10, 0.0189, 1.11, 0.21, 0.09, 0.30),
    ("census2025-11-08_149875496", "astrocyte", "astrocyte", True, 10, 0.0257, 1.11, 0.22, 0.17, 0.39),
    ("census2025-11-08_158740574", "oligodendrocyte precursor cell", "oligodendrocyte precursor cell", True, 10, 0.0265, 1.11, 0.22, 0.10, 0.31),
    ("census2025-11-08_125500046", "oligodendrocyte precursor cell", "oligodendrocyte precursor cell", True, 10, 0.0253, 1.11, 0.19, 0.14, 0.33),
    ("census2025-11-08_156653410", "microglial cell", "microglial cell", True, 10, 0.0198, 1.11, 0.22, 0.08, 0.30),
    ("census2025-11-08_126272216", "microglial cell", "microglial cell", True, 10, 0.0161, 1.11, 0.19, 0.08, 0.27),
    ("census2025-11-08_156780580", L, L, True, 10, 0.0065, 1.11, 0.20, 0.09, 0.29),
    ("census2025-11-08_158217014", L, L, True, 8, 0.0045, 1.11, 0.20, 0.10, 0.30),
], columns=_cols)
N_CORRECT = int(RESULTS["correct"].sum())

# --- SPLIT_TABLES (cell 23 saved output) ---
_PVALB = "pvalb GABAergic cortical interneuron"
SPLIT_TABLES = {
    "census2025-11-08_158217014": pd.DataFrame([
        {"rank": 1, "cell_id": "census2025-11-08_157391951", "cell type": L, "score": 0.0045, "same as the vote": True},
        {"rank": 2, "cell_id": "census2025-11-08_158676303", "cell type": L, "score": 0.0040, "same as the vote": True},
        {"rank": 3, "cell_id": "census2025-11-08_75513208",  "cell type": L, "score": 0.0033, "same as the vote": True},
        {"rank": 4, "cell_id": "census2025-11-08_157475468", "cell type": L, "score": 0.0030, "same as the vote": True},
        {"rank": 5, "cell_id": "census2025-11-08_156846276", "cell type": L, "score": 0.0028, "same as the vote": True},
        {"rank": 6, "cell_id": "census2025-11-08_157684147", "cell type": _PVALB, "score": 0.0028, "same as the vote": False},
        {"rank": 7, "cell_id": "census2025-11-08_158232299", "cell type": _PVALB, "score": 0.0026, "same as the vote": False},
        {"rank": 8, "cell_id": "census2025-11-08_158654216", "cell type": L, "score": 0.0026, "same as the vote": True},
        {"rank": 9, "cell_id": "census2025-11-08_149302603", "cell type": L, "score": 0.0026, "same as the vote": True},
        {"rank": 10, "cell_id": "census2025-11-08_156694062", "cell type": L, "score": 0.0026, "same as the vote": True},
    ]),
}

# --- LOO (cell 24 saved output) ---
_VIP, _LAMP5, _SST = "VIP GABAergic cortical interneuron", "lamp5 GABAergic cortical interneuron", "sst GABAergic cortical interneuron"
LOO = pd.DataFrame([
    ("loo_1",  _VIP,   _VIP,   True, 10, "none"),
    ("loo_2",  _LAMP5, _LAMP5, True, 10, "none"),
    ("loo_3",  _PVALB, _PVALB, True, 10, "none"),
    ("loo_4",  _LAMP5, _LAMP5, True, 10, "none"),
    ("loo_5",  _PVALB, _PVALB, True,  9, _SST + " (1)"),
    ("loo_6",  _SST,   _SST,   True, 10, "none"),
    ("loo_7",  _SST,   _SST,   True, 10, "none"),
    ("loo_8",  _VIP,   _VIP,   True,  9, _SST + " (1)"),
    ("loo_9",  _VIP,   _VIP,   True, 10, "none"),
    ("loo_10", _SST,   _SST,   True,  9, _PVALB + " (1)"),
    ("loo_11", _LAMP5, _LAMP5, True, 10, "none"),
    ("loo_12", _SST,   _SST,   True, 10, "none"),
    ("loo_13", _SST,   _SST,   True, 10, "none"),
    ("loo_14", _VIP,   _VIP,   True,  9, _PVALB + " (1)"),
    ("loo_15", _VIP,   _VIP,   True, 10, "none"),
    ("loo_16", _LAMP5, _LAMP5, True, 10, "none"),
    ("loo_17", _PVALB, _PVALB, True, 10, "none"),
    ("loo_18", _LAMP5, _LAMP5, True, 10, "none"),
    ("loo_19", _PVALB, _PVALB, True,  7, _SST + " (3)"),
    ("loo_20", _PVALB, _PVALB, True,  9, _SST + " (1)"),
], columns=["cell_id", "true type", "predicted type", "correct",
            "neighbours agreeing (of 10)", "other types among neighbours"])
LOO_CORRECT = 20

# --- Q_EMB: recompute with the local TEDDY model (endpoint is gone) ---
import anndata as ad
MODEL_DIR = f"{SHARED_SNAP}/teddy/models/teddy_g/70M"
CODE_DIR  = f"{RUN_DIR}/teddy_code"
if not os.path.exists(f"{CODE_DIR}/teddy"):
    os.makedirs(CODE_DIR, exist_ok=True)
    shutil.copytree(f"{SHARED_SNAP}/teddy", f"{CODE_DIR}/teddy",
                    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.pt", "__pycache__"))
for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]:
    del sys.modules[_k]
sys.path.insert(0, CODE_DIR)
from teddy.models.model_directory import get_architecture, model_dict
from teddy.tokenizer.gene_tokenizer import GeneTokenizer
_arch = get_architecture(MODEL_DIR)
_cfg = model_dict[_arch]["config_cls"].from_pretrained(MODEL_DIR)
model = model_dict[_arch]["model_cls"].from_pretrained(MODEL_DIR, config=_cfg)
model.to("cuda").eval(); model.return_all_embs = True
tok = GeneTokenizer.from_pretrained(MODEL_DIR)
with open(f"{CODE_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json") as f:
    medians = json.load(f)

qa = ad.read_h5ad(QUERY_H5AD)
qa = qa.raw.to_adata() if qa.raw is not None else qa
_vocab = set(tok.vocab)
_gene_mask = np.array([g in _vocab for g in qa.var_names])
qa = qa[:, _gene_mask].copy()
Q_GENES = list(qa.var_names)
QX = qa.X.toarray().astype(np.float32) if hasattr(qa.X, "toarray") else np.asarray(qa.X, dtype=np.float32)
Q_IDS = list(qa.obs["cell_id"].astype(str))
Q_TRUE = list(qa.obs["cell_type"].astype(str))

X = QX.copy()
_tot = X.sum(1, keepdims=True); _tot[_tot == 0] = 1.0
X = X / _tot * 1e4
X *= np.array([1.0 / medians[g] if g in medians else 1.0 for g in Q_GENES], dtype=np.float32)[None, :]
_keep = [i for i, g in enumerate(Q_GENES) if g in tok.vocab]
_token_arr = torch.tensor([tok.vocab[Q_GENES[i]] for i in _keep])
_X_t = torch.from_numpy(np.ascontiguousarray(X[:, _keep], dtype=np.float32))
_k = min(2048, _X_t.shape[1])
_top_vals, _top_idx = torch.topk(_X_t, k=_k, largest=True, sorted=True)
_gene_ids = _token_arr[_top_idx].clone()
_expressed = _top_vals > 0
_gene_ids[~_expressed] = tok.pad_token_id
_att_mask = _expressed.long()

_q_emb = []
for _s in range(0, _gene_ids.shape[0], 10):
    _ids, _m = _gene_ids[_s:_s + 10].to("cuda"), _att_mask[_s:_s + 10].to("cuda")
    with torch.no_grad():
        _out = model(gene_ids=_ids, attention_mask=_m)
    _H = _out["all_embs"]
    _M = _m.unsqueeze(-1).to(_H.dtype)
    _q_emb.append(((_H * _M).sum(1) / _M.sum(1).clamp(min=1)).float().cpu().numpy())
Q_EMB = np.concatenate(_q_emb, 0)
print(f"Reconstructed {len(RESULTS)} results, Q_EMB {Q_EMB.shape}, LOO {LOO_CORRECT}/{len(LOO)}; ready for figures and report")

# COMMAND ----------

# DBTITLE 1,Labelling figure: neighbours agreeing per query cell
import textwrap, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
FIG_DIR = f"{RUN_DIR}/figures"; os.makedirs(FIG_DIR, exist_ok=True)

@viz.styled
def plot_labelling(res):
    """One row per query cell, all rows the same height: bar = neighbours agreeing with the vote (of 10)."""
    P = viz.PALETTE
    def label(cid, t, p):                                        # type once; predicted only if it differs; cell id keeps rows distinct
        name = textwrap.fill(t, 34)
        if t != p:
            name += "\n" + textwrap.fill(f"-> predicted {p}", 34)
        return name + f"\ncell {cid.split('_')[-1]}"
    labels = [label(*r) for r in zip(res["cell_id"], res["true type"], res["predicted type"])]
    n, max_lines = len(labels), max(l.count("\n") + 1 for l in labels)
    row_in = 0.19 * max_lines + 0.2                              # same height for every row, fits the longest label
    top_in, bottom_in = 0.5, 1.2
    H = n * row_in + top_in + bottom_in
    fig = plt.figure(figsize=(10, H))
    ax = fig.add_axes([0.38, bottom_in / H, 0.56, n * row_in / H])
    y = np.arange(n)
    vals = res["neighbours agreeing (of 10)"].to_numpy()
    colors = [P["navy_700"] if c else P["red_500"] for c in res["correct"]]
    ax.barh(y, vals, height=0.6, color=colors)                   # constant bar thickness
    for yi, v in zip(y, vals):
        ax.text(v + 0.15, yi, f"{v}/10", va="center", fontsize=10)
    ch = res["chance (of 10)"].to_numpy()
    if np.allclose(ch, ch[0]):                                   # balanced reference: one chance line
        ax.axvline(ch[0], color=P["grey_700"], linestyle="--", linewidth=1.5)
    else:
        for yi, c in zip(y, ch):
            ax.plot([c, c], [yi - 0.3, yi + 0.3], color=P["grey_700"], linestyle="--", linewidth=1.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5, linespacing=1.1); ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, 11); ax.set_xticks(range(0, 11, 2)); ax.set_xlabel("Neighbours agreeing with the vote (of 10)")
    ax.set_title(f"{int(res['correct'].sum())} of {n} unseen cells labelled correctly", loc="left")
    fig.legend([Patch(color=P["navy_700"]), Patch(color=P["red_500"]), Line2D([0], [0], color=P["grey_700"], linestyle="--")],
               ["correct", "wrong", f"chance ({ch[0]:.1f} of 10, type's share of the index)" if np.allclose(ch, ch[0]) else "chance (type's share of the index)"],
               loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=True)
    return fig

fig_label = plot_labelling(RESULTS)
fig_label.savefig(f"{FIG_DIR}/labelling.png", dpi=200, bbox_inches="tight")
display(fig_label)

# COMMAND ----------

# DBTITLE 1,UMAP of the reference cells with the 10 query cells
import time
t0 = time.time()
ref_pd = spark.table(CELLS_TABLE).select("cell_id", "cell_type", "embedding").orderBy("cell_id").toPandas()
REF_EMB = np.array(ref_pd["embedding"].tolist(), dtype=float)
FAMILIES = {"red": ["L2/3-6 intratelencephalic projecting glutamatergic neuron"],
            "navy": ["VIP GABAergic cortical interneuron", "pvalb GABAergic cortical interneuron",
                     "lamp5 GABAergic cortical interneuron", "sst GABAergic cortical interneuron"],
            "orange": ["oligodendrocyte", "oligodendrocyte precursor cell"],
            "teal": ["astrocyte", "microglial cell"]}
assert set(sum(FAMILIES.values(), [])) == set(ref_pd["cell_type"]), "every reference type must have a colour"
NO_QUERY_TYPES = sorted(set(ref_pd["cell_type"]) - set(Q_TRUE))
ref_xy, q_xy, method = viz.map_2d(REF_EMB, Q_EMB, seed=42)
fig_umap = viz.plot_reference_map(ref_xy, ref_pd["cell_type"].values, None, query_xy=q_xy, method=method,
    query_label="query cells (10, rings)", families=FAMILIES,
    note=f"{len(ref_pd)} reference cells, {ref_pd['cell_type'].nunique()} types; map fitted on the reference only, "
         f"query cells placed onto it. Types with no query cell: {', '.join(NO_QUERY_TYPES)}.")
fig_umap.savefig(f"{FIG_DIR}/umap.png", dpi=200, bbox_inches="tight")
STEP_TIMES["3c. UMAP"] = time.time() - t0
display(fig_umap)
print(f"Reference types with no query cells ({len(NO_QUERY_TYPES)} of {ref_pd['cell_type'].nunique()}): {NO_QUERY_TYPES}")

# COMMAND ----------

# DBTITLE 1,Split votes: neighbour tables and top-1 score comparison
split = RESULTS[RESULTS["neighbours agreeing (of 10)"] < 10]
SPLIT_TABLES = {}
if split.empty:
    print("Every query cell had 10 of 10 neighbours agreeing.")
for i in split.index:
    r = RESULTS.loc[i]
    nb = pd.DataFrame([{"rank": k + 1, "cell_id": h["cell_id"], "cell type": h["cell_type"], "score": round(float(h["score"]), 4),
                        "same as the vote": h["cell_type"] == r["predicted type"]} for k, h in enumerate(hits_all[i])])
    SPLIT_TABLES[r["cell_id"]] = nb
    others = RESULTS.drop(index=i)["top-1 score"]
    rank = int((RESULTS["top-1 score"] > r["top-1 score"]).sum()) + 1
    print(f"\n{r['cell_id']}: true {r['true type']}, voted {r['predicted type']} ({r['neighbours agreeing (of 10)']} of 10). "
          f"Top-1 score {r['top-1 score']:.4f} vs the other 9 cells: min {others.min():.4f}, median {others.median():.4f}, "
          f"max {others.max():.4f} (rank {rank} of 10, 1 = highest)")
    display(nb)

# COMMAND ----------

# DBTITLE 1,Interneuron leave-one-out on reference cells (not an unseen-cell test)
t0 = time.time()
INTERNEURONS = FAMILIES["navy"]
loo_src = ref_pd[ref_pd["cell_type"].isin(INTERNEURONS)].sort_values("cell_id").groupby("cell_type").head(5)
loo_rows = []
for _, r in loo_src.iterrows():
    hits = nn.query_neighbours(w, INDEX, r["embedding"], columns=["cell_id", "cell_type"], k=10, exclude_ids={r["cell_id"]})
    assert len(hits) == 10 and r["cell_id"] not in {h["cell_id"] for h in hits}
    v = nn.vote(hits)
    cnt = pd.Series([h["cell_type"] for h in hits]).value_counts()
    loo_rows.append({"cell_id": r["cell_id"], "true type": r["cell_type"], "predicted type": v["predicted"],
                     "correct": v["predicted"] == r["cell_type"], "neighbours agreeing (of 10)": int(cnt[v["predicted"]]),
                     "other types among neighbours": "; ".join(f"{t} ({n})" for t, n in cnt.items() if t != v["predicted"]) or "none"})
LOO = pd.DataFrame(loo_rows)
LOO_CORRECT = int(LOO["correct"].sum())
STEP_TIMES["3d. interneuron leave-one-out (20 lookups)"] = time.time() - t0
print(f"Leave-one-out: {LOO_CORRECT} of {len(LOO)} interneurons labelled correctly; agreement "
      f"{LOO['neighbours agreeing (of 10)'].min()}-{LOO['neighbours agreeing (of 10)'].max()} of 10.\n"
      "NOT an unseen-cell test: these cells are in the index (only the cell itself is dropped), from the same studies as their neighbours.")
display(LOO.drop(columns="cell_id"))

# COMMAND ----------

# DBTITLE 1,What was created, what still bills, and step times
ep = w.serving_endpoints.get(ENDPOINT)
idx = w.vector_search_indexes.get_index(INDEX)
vs = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
inf_tables = sorted(t.name for t in w.tables.list(CATALOG, SCHEMA) if t.name.startswith(INF_PREFIX + "_"))
served_state = ep.config.served_entities[0].state.deployment if ep.config and ep.config.served_entities else None
CREATED = pd.DataFrame([
    {"resource": "Serving endpoint (GPU_MEDIUM, scale to zero)", "name": ENDPOINT, "made by": "this notebook (prompt 1)",
     "state now": f"{ep.state.ready.value}, {getattr(served_state, 'value', served_state)}",
     "still billing": "yes, GPU while scaled up; scales to zero after about 30 min idle", "how to stop": f"delete endpoint {ENDPOINT} (Teardown prompt)"},
    {"resource": "Vector Search index (Delta Sync, triggered)", "name": INDEX, "made by": "this notebook (prompt 2)",
     "state now": f"ready={idx.status.ready}, {idx.status.indexed_row_count} rows",
     "still billing": f"yes, as part of shared endpoint {VS_ENDPOINT} (which bills while it exists)", "how to stop": f"delete index {INDEX}"},
    {"resource": "Vector Search endpoint (shared)", "name": VS_ENDPOINT, "made by": "Setup (not this notebook)",
     "state now": vs.endpoint_status.state.value, "still billing": "yes (shared; keep)", "how to stop": "not by this demo"},
    {"resource": "Delta table (CDF on)", "name": CELLS_TABLE, "made by": "this notebook (prompt 2)",
     "state now": f"{N_CELLS} rows", "still billing": "storage only", "how to stop": f"DROP TABLE {CELLS_TABLE}"},
    {"resource": "Inference / usage tables", "name": ", ".join(inf_tables) or f"{INF_PREFIX}_payload (appears after first requests)",
     "made by": "endpoint AI Gateway", "state now": "exists" if inf_tables else "pending", "still billing": "storage only",
     "how to stop": "drop after deleting the endpoint"},
    {"resource": "UC model + version", "name": f"{UC_MODEL} v{MODEL_VERSION}", "made by": "this notebook (prompt 1)",
     "state now": "registered (env_pack)", "still billing": "storage only", "how to stop": "delete version, then model"},
    {"resource": "MLflow experiment", "name": EXPERIMENT, "made by": "this notebook (prompt 1)", "state now": f"run {RUN_ID}",
     "still billing": "no", "how to stop": "delete experiment"},
    {"resource": "Notebook compute (serverless GPU A10)", "name": "this notebook", "made by": "you",
     "state now": "attached", "still billing": "yes while attached / running", "how to stop": "detach or let it idle out"},
])
STEP_TIMES["1e. endpoint create -> READY (min, upper bound)" if READY_AT_FIRST_CHECK else "1e. endpoint create -> READY (min)"] = ENDPOINT_READY_MIN * 60
display(CREATED)
STEPS = pd.DataFrame([{"step": k, "seconds": round(v, 1), "minutes": round(v / 60, 2)} for k, v in sorted(STEP_TIMES.items())])
display(STEPS)

# COMMAND ----------

# DBTITLE 1,Optional: report
# MAGIC %md
# MAGIC ## Optional: report
# MAGIC A short self-contained HTML report of this run (html-report skill), saved to `demo/demo_report.html`; every number comes from the cells above.

# COMMAND ----------

# DBTITLE 1,Build the HTML report
from importlib.metadata import version as _ver
sys.path.insert(0, "/Workspace/.assistant/skills/html-report/scripts")
import html_report as hr
REPORT = f"{DEMO_DIR}/demo_report.html"
os.makedirs(DEMO_DIR, exist_ok=True)

versions = pd.DataFrame([{"package": p, "version": _ver(p)} for p in
    ("mlflow", "databricks-sdk", "transformers", "torch", "numpy", "pandas", "anndata", "umap-learn")]
    + [{"package": "python", "version": sys.version.split()[0]},
       {"package": "serverless environment", "version": os.environ.get("DATABRICKS_RUNTIME_VERSION", "")},
       {"package": "GPU", "version": torch.cuda.get_device_name(0)}])
split_txt = []
for cid, nb in SPLIT_TABLES.items():
    r = RESULTS.set_index("cell_id").loc[cid]
    others = RESULTS[RESULTS["cell_id"] != cid]["top-1 score"]
    split_txt.append(f"`{cid}` ({r['true type']}) was voted correctly with {r['neighbours agreeing (of 10)']} of 10 neighbours; "
                     f"the others were {', '.join(sorted(set(nb.loc[~nb['same as the vote'], 'cell type'])))}. Its top-1 score "
                     f"{r['top-1 score']:.4f} is the lowest of the 10 query cells (others {others.min():.4f}-{others.max():.4f}), "
                     "so it sits furthest from any reference cell.")
res_tbl = RESULTS.drop(columns=["cell_id", "top-1 score"])
spec = {
  "title": "TEDDY-70M on Databricks: express deployment, Vector Search index and neighbour-vote labelling of 10 brain cells",
  "authors": USER,
  "summary": (f"TEDDY-70M (Hugging Face Merck/TEDDY, revision {HF_REVISION[:7]}) was registered as `{UC_MODEL}` v{MODEL_VERSION} "
              f"with an express deployment and served on `{ENDPOINT}` (GPU_MEDIUM). A Delta Sync index of {N_CELLS} reference brain "
              f"cells (9 types) answered queries {INDEX_READY_S/60:.1f} min after creation. {N_CORRECT} of 10 unseen cells were labelled "
              "correctly by a 10-neighbour vote, against a chance level of about 1.1 of 10 neighbours per type. No query cell was a "
              f"GABAergic interneuron; a leave-one-out check on reference interneurons gave {LOO_CORRECT} of {len(LOO)}, which is not an unseen-cell test."),
  "sections": [
    {"title": "Introduction", "blocks": [{"p": "TEDDY-G turns one cell's gene counts into a 512-number embedding. A new cell is labelled by the majority type of its 10 nearest labelled reference cells in a Vector Search index. This is in-silico prioritisation, not a measurement; neighbour agreement is a vote share, not a calibrated probability."}]},
    {"title": "Results", "blocks": [
      {"subheading": "What was deployed"},
      {"p": f"Files: {COPY_USED}; {PROVENANCE['files_in_manifest']} files checked. Registration with `env_pack` took {STEP_TIMES['1c. log + register (env_pack)']:.0f} s. The endpoint was READY {ENDPOINT_READY_MIN:.1f} min after create" + (" (at the first check, so an upper bound)" if READY_AT_FIRST_CHECK else "") + f". Served embeddings matched the notebook wrapper (min cosine {SERVED_VS_WRAPPER:.6f})."},
      {"subheading": "The index"},
      {"p": f"`{INDEX}` on shared endpoint `{VS_ENDPOINT}`: {N_CELLS} rows ({N_CELLS // 9} per type), {EMB_DIM}-dimensional, answering queries {INDEX_READY_S/60:.1f} min after create; a reference cell retrieved itself as top-1."},
      {"subheading": "10 unseen cells against chance"},
      {"figure": f"{FIG_DIR}/labelling.png", "caption": f"Neighbours agreeing with the vote (of 10) per query cell, labelled by true type and cell ID (the predicted type is shown only where it differs); navy = correct, red = wrong, dashed = chance (the type's share of the index, {RESULTS['chance (of 10)'].iloc[0]} of 10). {N_CORRECT} of 10 correct."},
      {"table": res_tbl, "caption": "Per-cell result and call times (one model call and one index query per cell)."},
      {"figure": f"{FIG_DIR}/umap.png", "caption": "UMAP of the reference embeddings coloured by type (related types share a colour family); the 10 query cells are rings placed onto the map."},
      {"subheading": "Leave-one-out on reference interneurons (not an unseen-cell test)"},
      {"p": f"First 5 reference cells by cell_id of each of the 4 interneuron types, queried with their own embedding, self dropped, voted over the next 10: {LOO_CORRECT} of {len(LOO)} correct, agreement {LOO['neighbours agreeing (of 10)'].min()}-{LOO['neighbours agreeing (of 10)'].max()} of 10. These cells are in the index and share studies with their neighbours, so this is a consistency check only."},
      {"table": LOO.drop(columns="cell_id"), "caption": "Interneuron leave-one-out results."},
      {"subheading": "Step times"}, {"table": STEPS, "caption": "Wall-clock time per step in this notebook."},
      {"subheading": "What was created and what still bills"}, {"table": CREATED, "caption": "Resources at the time of the report."}]},
    {"title": "Discussion", "blocks": [{"subheading": "Caveats"}, {"bullets": [
      f"Reference types with no query cells, so their unseen-cell accuracy is untested: {', '.join(NO_QUERY_TYPES)}.",
      *split_txt,
      "10 query cells is a demonstration, not an accuracy estimate.",
      "Index scores come from L2 distance on unnormalised embeddings; compare them within this index only.",
      "Licences: model card Apache-2.0, package pyproject MIT; confirm with the model owner before redistribution."]}]},
    {"title": "Methods", "blocks": [
      {"p": "Wrapper: counts normalised to 10,000 per cell, divided by TEDDY gene medians, vocabulary genes only, top 2,048 expressed genes, mean pooling over expressed genes (padding excluded)."},
      {"subheading": "Checks"}, {"checks": [
        {"check": "Checksums of Setup's copy", "threshold": "every file matches", "result": f"{PROVENANCE['files_in_manifest']} of {PROVENANCE['files_in_manifest']}", "passed": True},
        {"check": "Upstream parity (5 cells)", "threshold": "cosine >= 0.9999", "result": f"min {PARITY_MIN_COS:.7f}", "passed": True},
        {"check": "Served = notebook wrapper (10 query cells)", "threshold": "cosine >= 0.9999", "result": f"min {SERVED_VS_WRAPPER:.6f}", "passed": SERVED_VS_WRAPPER >= 0.9999},
        {"check": "Index rows", "threshold": f"= {N_CELLS}", "result": str(idx.status.indexed_row_count), "passed": idx.status.indexed_row_count == N_CELLS}]},
      {"subheading": "Software"}, {"table": versions, "caption": "Versions on the notebook compute."}]},
    {"title": "Code availability", "blocks": [{"p": f"Model code and weights: [Merck/TEDDY](https://huggingface.co/Merck/TEDDY) at revision `{HF_REVISION}`. The notebook ({NB_PATH}) is workspace-only."}]},
    {"title": "Data availability", "blocks": [{"p": "Reference and query cells are from CELLxGENE Census release 2025-11-08 (cell IDs carry the release); the embedded reference table and index are workspace-only."}]},
    {"title": "References", "blocks": [{"bullets": ["TEDDY: [arXiv:2503.03485](https://arxiv.org/abs/2503.03485)", "[Merck/TEDDY on Hugging Face](https://huggingface.co/Merck/TEDDY)"]}]},
  ]}
out = hr.build_report(spec, REPORT, overwrite=True)
print({k: v for k, v in out.items() if k != "problems"}, "\nproblems:", out["problems"])

# COMMAND ----------

# DBTITLE 1,Preview the report inline
hr.display_report(out["path"])

# COMMAND ----------

# DBTITLE 1,Teardown
# MAGIC %md
# MAGIC ## Teardown
# MAGIC Delete this demo's resources (endpoint, index, tables, model, volume folder) to free paths for a new demo.
# MAGIC Never touches Setup resources (`teddy_ep_shared`, `teddy_vs_workshop`, `teddy_ref_index`, `teddy_ref_cells`, `teddy_model_shared`, `shared/`).
# MAGIC Set the **teardown** widget to `true` and re-run this cell.

# COMMAND ----------

# DBTITLE 1,Tear down this demo's resources (gated)
# Teardown: delete this demo's resources to free paths for a new demo.
# NEVER touches Setup resources or anything not listed below.
import os, shutil
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

w = WorkspaceClient()
CATALOG, SCHEMA = "main", "owm"
VOLUME = "teddy_files"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
ENDPOINT    = "teddy_ep_demo"
INDEX       = f"{CATALOG}.{SCHEMA}.teddy_demo_index"
VS_ENDPOINT = "teddy_vs_workshop"
CELLS_TABLE = f"{CATALOG}.{SCHEMA}.teddy_demo_cells"
UC_MODEL    = f"{CATALOG}.{SCHEMA}.teddy_model_demo"
INF_PREFIX  = "teddy_ep_demo"
DEMO_DIR    = f"{VOLUME_PATH}/demo"

def _model_versions():
    """Version numbers of UC_MODEL. The GET-model response has no versions; they live under /versions."""
    out, token = [], None
    while True:
        q = {"max_results": 100, **({"page_token": token} if token else {})}
        r = w.api_client.do("GET", f"/api/2.1/unity-catalog/models/{UC_MODEL}/versions", query=q)
        out += [v["version"] for v in r.get("model_versions", [])]
        token = r.get("next_page_token")
        if not token:
            return sorted(out)

def _list_demo():
    """List every demo resource with its state (no deletion)."""
    found = {}
    # serving endpoint
    try:
        ep = w.serving_endpoints.get(ENDPOINT)
        found["endpoint"] = f"{ep.state.ready}"
        print(f"  endpoint  {ENDPOINT}: {ep.state.ready}")
    except NotFound:
        print(f"  endpoint  {ENDPOINT}: NOT FOUND (already gone)")
    # VS index (only a 404 means gone; other errors are shown as they are)
    try:
        idx = w.api_client.do("GET", f"/api/2.0/vector-search/endpoints/{VS_ENDPOINT}/indexes/{INDEX}")
        st = idx.get("status", {})
        rc = st.get("indexed_row_count", "?")
        found["index"] = f"{rc} rows"
        print(f"  index     {INDEX}: ready={st.get('ready', '?')}, {rc} rows")
    except NotFound:
        print(f"  index     {INDEX}: NOT FOUND (already gone)")
    except Exception as e:
        print(f"  index     {INDEX}: ERROR checking ({type(e).__name__}: {e})")
    # tables
    demo_tables = []
    for t in w.tables.list(CATALOG, SCHEMA):
        if t.name == "teddy_demo_cells" or t.name.startswith(INF_PREFIX + "_"):
            demo_tables.append(t.name)
            found[f"table_{t.name}"] = "exists"
            print(f"  table     {CATALOG}.{SCHEMA}.{t.name}: EXISTS")
    if not demo_tables:
        print(f"  tables    (teddy_demo_cells, {INF_PREFIX}_*): NONE FOUND")
    # model + versions (REST, not SDK — see workspace instructions)
    try:
        w.api_client.do("GET", f"/api/2.1/unity-catalog/models/{UC_MODEL}")
        versions = _model_versions()
        found["model"] = f"versions={versions}"
        print(f"  model     {UC_MODEL}: EXISTS, versions={versions}")
    except NotFound:
        print(f"  model     {UC_MODEL}: NOT FOUND")
    except Exception as e:
        print(f"  model     {UC_MODEL}: ERROR checking ({type(e).__name__}: {e})")
    # volume folder
    if os.path.exists(DEMO_DIR):
        items = os.listdir(DEMO_DIR)
        found["volume"] = f"{len(items)} items"
        print(f"  volume    {DEMO_DIR}: EXISTS ({len(items)} items: {items})")
    else:
        print(f"  volume    {DEMO_DIR}: NOT FOUND")
    return found

def _list_setup():
    """Verify Setup resources are still present (never touched by teardown)."""
    setup_ok = []
    for name in ("teddy_ep_shared",):
        try:
            w.serving_endpoints.get(name); setup_ok.append(f"endpoint {name}: present")
        except NotFound:
            setup_ok.append(f"endpoint {name}: MISSING")
    try:
        vs = w.api_client.do("GET", f"/api/2.0/vector-search/endpoints/{VS_ENDPOINT}")
        setup_ok.append(f"VS endpoint {VS_ENDPOINT}: {vs.get('endpoint_status',{}).get('state','?')}")
    except Exception:
        setup_ok.append(f"VS endpoint {VS_ENDPOINT}: MISSING")
    try:
        w.api_client.do("GET", f"/api/2.0/vector-search/endpoints/{VS_ENDPOINT}/indexes/{CATALOG}.{SCHEMA}.teddy_ref_index")
        setup_ok.append("index teddy_ref_index: present")
    except Exception:
        setup_ok.append("index teddy_ref_index: MISSING")
    for tname in ("teddy_ref_cells",):
        try:
            w.tables.get(f"{CATALOG}.{SCHEMA}.{tname}"); setup_ok.append(f"table {tname}: present")
        except Exception:
            setup_ok.append(f"table {tname}: MISSING")
    try:
        w.api_client.do("GET", f"/api/2.1/unity-catalog/models/{CATALOG}.{SCHEMA}.teddy_model_shared")
        setup_ok.append("model teddy_model_shared: present")
    except Exception:
        setup_ok.append("model teddy_model_shared: MISSING")
    return setup_ok

print("=" * 60)
print("Demo resources (before teardown):")
print("=" * 60)
_list_demo()

print("\n" + "=" * 60)
print("Setup resources (should be untouched):")
print("=" * 60)
for s in _list_setup():
    print(f"  {s}")

dbutils.widgets.dropdown("teardown", "false", ["false", "true"], "Tear down demo")
if dbutils.widgets.get("teardown") != "true":
    print("\n>>> Set the 'teardown' widget to true and re-run this cell to delete the demo resources.")
else:
    print("\n" + "=" * 60)
    print("Tearing down (in order: endpoint, index, tables, model versions, model, folder)")
    print("=" * 60)
    # a. endpoint
    try:
        w.serving_endpoints.delete(ENDPOINT); print(f"  deleted endpoint {ENDPOINT}")
    except NotFound:
        print(f"  endpoint  {ENDPOINT}: already gone")
    # b. index
    try:
        w.api_client.do("DELETE", f"/api/2.0/vector-search/endpoints/{VS_ENDPOINT}/indexes/{INDEX}")
        print(f"  deleted index {INDEX}")
    except NotFound:
        print(f"  index     {INDEX}: already gone")
    except Exception as e:
        print(f"  index     {INDEX}: ERROR deleting ({type(e).__name__}: {e})")
    # c. tables (listed fresh here; same name filter as _list_demo)
    for t in w.tables.list(CATALOG, SCHEMA):
        if t.name == "teddy_demo_cells" or t.name.startswith(INF_PREFIX + "_"):
            try:
                w.tables.delete(CATALOG, SCHEMA, t.name); print(f"  deleted table {CATALOG}.{SCHEMA}.{t.name}")
            except Exception as e:
                print(f"  table     {t.name}: {e}")
    # d. model versions (list via REST, then delete each, then the model)
    try:
        for vnum in _model_versions():
            try:
                w.api_client.do("DELETE", f"/api/2.1/unity-catalog/models/{UC_MODEL}/versions/{vnum}")
                print(f"  deleted model version {vnum}")
            except Exception as e:
                print(f"  model version {vnum}: {e}")
        w.registered_models.delete(UC_MODEL); print(f"  deleted model {UC_MODEL}")
    except NotFound:
        print(f"  model     {UC_MODEL}: already gone")
    except Exception as e:
        print(f"  model     {UC_MODEL}: ERROR ({type(e).__name__}: {e})")
    # e. volume folder
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR); print(f"  deleted volume folder {DEMO_DIR}")
    else:
        print(f"  volume    {DEMO_DIR}: already gone")

    print("\n" + "=" * 60)
    print("After teardown — demo resources (should be nothing):")
    print("=" * 60)
    _list_demo()
    print("\n" + "=" * 60)
    print("Setup resources (should still be there):")
    print("=" * 60)
    for s in _list_setup():
        print(f"  {s}")
    print("\nNot deleted: the MLflow experiment (no billing) and Setup resources.")
    print(">>> Set the 'teardown' widget back to false now, so a later 'Run all' does not tear down again.")