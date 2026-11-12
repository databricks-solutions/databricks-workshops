# Databricks notebook source
# DBTITLE 1,VHH Nanobody Design — RFD4-Proteina
# MAGIC %md
# MAGIC # VHH Nanobody Design against Hen Egg-White Lysozyme via RFD4-Proteina
# MAGIC
# MAGIC **Paper:** Corley, Didi, et al., *Biomolecular design through latent reasoning with RFD4-Proteina*, 2026 (CC BY 4.0).  
# MAGIC **Task:** `antibody_vhh` — epitope-conditioned single-domain antibody (VHH / nanobody) design.
# MAGIC
# MAGIC **Goal:** Design VHH nanobodies against **hen egg-white lysozyme** (PDB `1LYZ`, chain A), epitope residues `19,21,23,73,75,100,101`, and triage through a **binding + developability + immunogenicity** funnel.
# MAGIC
# MAGIC **Funnel:** RFD4-Proteina (design) → ESMFold (monomer fold + pLDDT gate) → Boltz-2 (VHH–antigen co-fold + ipTM) → developability (NetSolP, DeepSTABp, PLTNUM) → immunogenicity (MHCflurry MHC-I **and** HLAIIPred MHC-II / anti-drug-antibody) → composite reward → ranked shortlist.
# MAGIC
# MAGIC **Deviations from the paper:**
# MAGIC - GWB AlphaFold2 is monomer-only → **Boltz-2 ipTM** is the complex/interface filter.
# MAGIC - No Rosetta ΔΔG → rank on ipTM + pLDDT.
# MAGIC - RFD4-Proteina designs a VHH-length binder conditioned on the epitope (C_HOT) — a first-draft VHH, not yet a full immunoglobulin-framework scaffold with fixed framework + designed CDR loops.
# MAGIC - Developability endpoints are proxies that prioritize what to order; PLTNUM is a 0–1 ranker, not hours.
# MAGIC
# MAGIC **Honest scope:** This in-silico funnel does **enrichment & prioritization** (which designs to order), not measurement. Wet-lab assays still decide.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q "databricks-sdk>=0.50.0" requests
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,GWB Protein-Design Toolkit (endpoints only)
import numpy as np, pandas as pd, requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.serving import DataframeSplitInput

w = WorkspaceClient(config=Config(http_timeout_seconds=900))

_EP_CACHE, _ALL_ENDPOINTS = {}, None
def get_ep(slug, override="", exclude=()):
    """Resolve a GWB serving-endpoint name by model slug; fail loud if missing/asleep."""
    if slug in _EP_CACHE:
        return _EP_CACHE[slug]
    if override.strip():
        _EP_CACHE[slug] = override.strip(); return _EP_CACHE[slug]
    global _ALL_ENDPOINTS
    if _ALL_ENDPOINTS is None:
        _ALL_ENDPOINTS = [e.name for e in w.serving_endpoints.list()]
    cands = [n for n in _ALL_ENDPOINTS if slug in n and not any(x in n for x in exclude)]
    if not cands:
        raise RuntimeError(f"No serving endpoint matches '{slug}'. Available: {sorted(_ALL_ENDPOINTS)}. "
                           f"Is the model deployed and awake?")
    def stem(n): return n[:-len("_endpoint")] if n.endswith("_endpoint") else n
    exact = [n for n in cands if stem(n).endswith(slug)]
    _EP_CACHE[slug] = (exact or cands)[0]
    return _EP_CACHE[slug]

def _unwrap(preds):
    if isinstance(preds, dict) and "predictions" in preds:
        preds = preds["predictions"]
    return [preds] if isinstance(preds, dict) else list(preds or [])

def to_float(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

_PC_COLS = ["target_pdb", "binder_length_min", "binder_length_max",
            "num_samples", "hotspot_residues", "target_chain"]

def _proteina(slug, slot_pdb, length_min, length_max, num_samples, hotspots, chain, exclude=()):
    resp = w.serving_endpoints.query(name=get_ep(slug, exclude=exclude),
        dataframe_split=DataframeSplitInput(columns=_PC_COLS,
            data=[[slot_pdb, int(length_min), int(length_max), int(num_samples), str(hotspots), str(chain)]]))
    return [r for r in _unwrap(resp.predictions) if isinstance(r, dict)]

def gen_protein_binders(target_pdb, target_chain="A", hotspots="", length_min=50, length_max=80, num_samples=4):
    return _proteina("proteina_complexa", target_pdb, length_min, length_max, num_samples,
                     hotspots, target_chain, exclude=("ligand", "ame"))

def gen_ligand_binders(ligand_pdb, length_min=50, length_max=80, num_samples=4):
    return _proteina("proteina_complexa_ligand", ligand_pdb, length_min, length_max, num_samples, "", "A")

def scaffold_motif(motif_pdb, motif_chain="B", length_min=50, length_max=80, num_samples=4):
    return _proteina("proteina_complexa_ame", motif_pdb, length_min, length_max, num_samples, "", motif_chain)

_RFD4_COLS = ["task", "target_pdb", "binder_length_min", "binder_length_max", "num_samples",
              "hotspot_residues", "target_chain", "condition_spec_json", "structure_cif", "exclude_chain"]

def gen_vhh_antibodies(target_pdb, target_cif="", target_chain="A", hotspots="", length_min=110, length_max=130, num_samples=4):
    """Epitope-conditioned single-domain antibody (VHH / nanobody) design via RFD4-Proteina.
    VHH default length window 110-130. exclude_chain = the antigen chain so designed_sequence
    is the VHH ALONE and complex_pdb is the full co-complex.
    RFD4-Proteina requires CIF format — pass target_cif through structure_cif column.
    If target_cif is provided, leave target_pdb empty to avoid PDB-format rejection."""
    structure_input = str(target_cif) if target_cif else str(target_pdb)
    resp = w.serving_endpoints.query(name=get_ep("rfd4_proteina"),
        dataframe_split=DataframeSplitInput(columns=_RFD4_COLS,
            data=[["binder", "", int(length_min), int(length_max), int(num_samples),
                   str(hotspots), str(target_chain), "", str(structure_input), str(target_chain)]]))
    out = []
    for r in _unwrap(resp.predictions):
        if not isinstance(r, dict):
            continue
        out.append({"sample_id": r.get("sample_id"), "sequence": str(r.get("designed_sequence", "")),
                    "rewards": 0.0, "pdb_output": r.get("designed_pdb", ""),
                    "complex_pdb": r.get("complex_pdb", "")})
    return out

def esmfold(sequence):
    return w.serving_endpoints.query(name=get_ep("esmfold"), inputs=[sequence]).predictions[0]

def boltz_complex(spec, msa="no_msa", use_msa_server="True"):
    resp = w.serving_endpoints.query(name=get_ep("boltz"),
        inputs=[{"input": spec, "msa": msa, "use_msa_server": use_msa_server}])
    out = _unwrap(resp.predictions)
    return out[0] if out else {}

_DEV_SPEC = {
    "netsolp":   (lambda s: {"sequence": s}, "predicted_solubility", "predicted_solubility"),
    "deepstabp": (lambda s: {"sequence": s, "growth_temp": 37.0, "mt_mode": "Cell"}, "predicted_tm_celsius", "predicted_tm_celsius"),
    "pltnum":    (lambda s: {"sequence": s}, "predicted_stability", "predicted_stability"),
    "mhcflurry": (lambda s: {"sequence": s,
                  "alleles": "HLA-A*02:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02,HLA-C*07:01,HLA-C*04:01"},
                  "predicted_immuno_burden", "predicted_immuno_burden"),
    "hlaiipred": (lambda s: {"sequence": s,
                  "alleles": "DRB1*01:01,DRB1*03:01,DRB1*04:01,DRB1*07:01,DRB1*08:01,DRB1*11:01,DRB1*13:01,DRB1*15:01"},
                  "predicted_immuno_burden", "predicted_immuno_mhc2"),
}

def developability(sequence, which=("netsolp", "deepstabp", "pltnum", "mhcflurry")):
    out = {}
    for slug in which:
        build, col, alias = _DEV_SPEC[slug]
        resp = w.serving_endpoints.query(name=get_ep(slug), dataframe_records=[build(sequence)])
        rows = _unwrap(resp.predictions)
        out[alias] = to_float(rows[0].get(col)) if rows else float("nan")
    return out

_AA3TO1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H",
           "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
           "TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O"}

def fetch_pdb(pdb_id):
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
    r.raise_for_status()
    return r.text

def fetch_cif(pdb_id):
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.cif", timeout=30)
    r.raise_for_status()
    return r.text

def chain_sequence(pdb_text, chain="A"):
    seq, seen = [], set()
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA" and line[21] == chain:
            key = line[22:27]
            if key in seen:
                continue
            seen.add(key)
            seq.append(_AA3TO1.get(line[17:20].strip(), "X"))
    return "".join(seq)

def mean_plddt(pdb_text):
    vals = [float(l[60:66]) for l in pdb_text.splitlines()
            if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    # ESMFold returns B-factors on 0-1 scale; scale to 0-100 pLDDT
    return float(np.mean(vals)) * 100.0 if vals else float("nan")

def _norm(vals, invert=False):
    arr = np.array([v if v is not None and not np.isnan(v) else np.nan for v in vals], dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or hi == lo:
        base = np.full(arr.shape, 0.5)
    else:
        base = (arr - lo) / (hi - lo)
        base = np.where(np.isnan(base), 0.0, base)
    return (1.0 - base) if invert else base

DEFAULT_WEIGHTS = {"plddt": 1.3, "iptm": 0.5, "solubility": 1.0, "stability": 2.6, "tm": 1.0, "immuno": 1.5, "immuno_mhc2": 1.5}

def composite_reward(df, axis_cols, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    keys = [k for k in axis_cols if k in weights]
    normed = {k: _norm(df[axis_cols[k][0]], invert=not axis_cols[k][1]) for k in keys}
    total = sum(weights[k] for k in keys) or 1.0
    return pd.Series(sum(weights[k] * normed[k] for k in keys) / total, index=df.index)

def show_molstar(structure, fmt="pdb", binder_chain="A", target_chain="B", highlight_resids=None,
                 binder_label="Design", target_label="Target", height=620):
    """PDBe Mol* viewer (multi-CDN fallback). Binder=teal, target=gold, epitope=red ball-and-stick."""
    import base64, json
    if not structure:
        print("show_molstar: no structure to display"); return
    b64 = base64.b64encode(structure.encode()).decode()
    sel = [{"auth_asym_id": binder_chain, "color": {"r": 51, "g": 181, "b": 229}},
           {"auth_asym_id": target_chain, "color": {"r": 230, "g": 168, "b": 23}}]
    for r in (highlight_resids or []):
        sel.append({"auth_asym_id": target_chain, "auth_residue_number": int(r),
                    "color": {"r": 231, "g": 76, "b": 60}, "sideChain": True,
                    "representation": "ball-and-stick", "representationColor": {"r": 231, "g": 76, "b": 60}})
    hl = ", ".join(str(int(r)) for r in (highlight_resids or []))
    epi_legend = ('<span><span class="sw" style="background:#e74c3c;"></span>Epitope/hotspots (' + hl
                  + ')</span>') if highlight_resids else ""
    html = ("""
<!DOCTYPE html><html><head><style>
  body{margin:0;padding:8px;background:#1a1a2e;font-family:'Segoe UI',Arial,sans-serif;}
  #molviewer{position:relative;width:100%;height:__H__px;}
  .legend{display:flex;flex-wrap:wrap;gap:16px;padding:8px 2px;color:#aaa;font-size:13px;line-height:1.6;}
  .sw{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:5px;vertical-align:middle;}
  .hint{color:#666;font-style:italic;}
</style></head><body>
  <div id="molviewer"></div>
  <div class="legend">
    <span><span class="sw" style="background:#33b5e5;"></span>__BINDER__ (chain __BC__)</span>
    <span><span class="sw" style="background:#e6a817;"></span>__TARGET__ (chain __TC__)</span>
    __EPI_LEGEND__
    <span class="hint">drag = rotate &middot; scroll = zoom &middot; click = residue info</span>
  </div>
  <script>
  (function(){
    var CDNS=[
      {js:"https://cdn.jsdelivr.net/npm/pdbe-molstar@3.3.0/build/pdbe-molstar-plugin.js",
       css:"https://cdn.jsdelivr.net/npm/pdbe-molstar@3.3.0/build/pdbe-molstar.css"},
      {js:"https://unpkg.com/pdbe-molstar@3.3.0/build/pdbe-molstar-plugin.js",
       css:"https://unpkg.com/pdbe-molstar@3.3.0/build/pdbe-molstar.css"},
      {js:"https://www.ebi.ac.uk/pdbe/pdb-component-library/js/pdbe-molstar-plugin-3.1.3.js",
       css:"https://www.ebi.ac.uk/pdbe/pdb-component-library/css/pdbe-molstar-3.1.3.css"}];
    var el=document.getElementById("molviewer");
    function render(){
      try{
        var v=new PDBeMolstarPlugin();
        v.render(el,{customData:{url:"data:text/plain;base64,__B64__",format:"__FMT__"},
                     bgColor:{r:26,g:26,b:46},hideControls:false,sequencePanel:true,landscape:true});
        v.events.loadComplete.subscribe(function(){ try{ v.visual.select({data:__SEL__}); }catch(e){} });
      }catch(e){ el.innerText="PDBe Mol* render failed: "+(e&&e.message); }
    }
    function tryCdn(i){
      if(typeof PDBeMolstarPlugin!=="undefined"){render();return;}
      if(i>=CDNS.length){el.innerText="PDBe Mol* failed: all CDNs unreachable";return;}
      var link=document.createElement("link");link.rel="stylesheet";link.href=CDNS[i].css;document.head.appendChild(link);
      var s=document.createElement("script");s.src=CDNS[i].js;
      s.onload=function(){ (typeof PDBeMolstarPlugin!=="undefined")?render():tryCdn(i+1); };
      s.onerror=function(){tryCdn(i+1);};
      document.head.appendChild(s);
    }
    tryCdn(0);
  })();
  </script>
</body></html>
""".replace("__H__", str(int(height))).replace("__B64__", b64).replace("__FMT__", fmt)
   .replace("__BC__", binder_chain).replace("__TC__", target_chain)
   .replace("__BINDER__", binder_label).replace("__TARGET__", target_label)
   .replace("__EPI_LEGEND__", epi_legend).replace("__SEL__", json.dumps(sel)))
    displayHTML(html)

print("\u2705 GWB protein-design toolkit loaded (endpoints resolve lazily via get_ep; Mol* = default viewer).")

# COMMAND ----------

# DBTITLE 1,Problem definition — lysozyme target + epitope
# --- Target: hen egg-white lysozyme (PDB 1LYZ, chain A) ---
target_pdb_text = fetch_pdb("1LYZ")
target_cif_text = fetch_cif("1LYZ")
TARGET_CHAIN = "A"
target_seq = chain_sequence(target_pdb_text, TARGET_CHAIN)
print(f"Lysozyme chain {TARGET_CHAIN}: {len(target_seq)} residues")
print(target_seq[:80], "...")
print(f"CIF fetched: {len(target_cif_text)} chars")

# Epitope residues (canonical nanobody benchmark site)
HOTSPOTS = "19,21,23,73,75,100,101"

# VHH length window (single-domain antibody: ~110–130 residues)
LENGTH_MIN, LENGTH_MAX = 110, 130

# Number of designs per batch (keep small for a class run; raise for production)
NUM_DESIGNS = 4

# COMMAND ----------

# DBTITLE 1,Generate VHH candidates (RFD4-Proteina)
# --- Step 1: Epitope-conditioned VHH design via RFD4-Proteina ---
# RFD4-Proteina generates VHH-length binders conditioned on the epitope hotspots (C_HOT).
# Cold endpoints can take ~10 min to spin up; get_ep fails loud if a model is missing/asleep.
print(f"Generating {NUM_DESIGNS} VHH candidates against lysozyme epitope [{HOTSPOTS}] ...")
raw = gen_vhh_antibodies(
    target_pdb=target_pdb_text,
    target_cif=target_cif_text,
    target_chain=TARGET_CHAIN,
    hotspots=HOTSPOTS,
    length_min=LENGTH_MIN,
    length_max=LENGTH_MAX,
    num_samples=NUM_DESIGNS,
)

designs = [
    {"sample_id": str(r.get("sample_id", i)),
     "sequence": str(r.get("sequence", "")),
     "rewards": to_float(r.get("rewards", 0.0), 0.0),
     "backbone_pdb": r.get("pdb_output", ""),
     "complex_pdb": r.get("complex_pdb", "")}
    for i, r in enumerate(raw) if str(r.get("sequence", "")).strip()
]
print(f"\u2705 {len(designs)} VHH candidates returned.")
for d in designs:
    print(f"  {d['sample_id']}: {len(d['sequence'])} aa")

# COMMAND ----------

# DBTITLE 1,Monomer fold gate (ESMFold — pLDDT ≥ 70)
# --- Step 2: Monomer fold gate (ESMFold) ---
# Fold each VHH monomer and gate on mean pLDDT >= 70.
PLDDT_CUTOFF = 70.0
print(f"Folding {len(designs)} VHH monomers with ESMFold (gate: pLDDT \u2265 {PLDDT_CUTOFF}) ...")
for d in designs:
    fold_pdb = esmfold(d["sequence"])
    d["mean_plddt"] = mean_plddt(fold_pdb)
    d["fold_pdb"] = fold_pdb
    print(f"  {d['sample_id']}: pLDDT = {d['mean_plddt']:.1f} {'\u2705' if d['mean_plddt'] >= PLDDT_CUTOFF else '\u274c'}")

folded = sorted(
    [d for d in designs if d["mean_plddt"] >= PLDDT_CUTOFF],
    key=lambda d: d["mean_plddt"], reverse=True,
)
print(f"\n{len(folded)}/{len(designs)} pass the pLDDT gate.")

# COMMAND ----------

# DBTITLE 1,Complex confidence (Boltz-2 — ipTM)
# --- Step 3: VHH–lysozyme co-fold with Boltz-2 (interface confidence: ipTM) ---
TOP_N = min(4, len(folded))
IPTM_CUTOFF = 0.50
shortlist = folded[:TOP_N]

print(f"Co-folding top {len(shortlist)} VHH–lysozyme complexes with Boltz-2 ...")
for d in shortlist:
    spec = f"protein_A:{d['sequence']};protein_B:{target_seq}"
    res = boltz_complex(spec)
    d["iptm"] = to_float(res.get("iptm"))
    d["ptm"] = to_float(res.get("ptm"))
    d["complex_pdb"] = res.get("pdb", d.get("complex_pdb", ""))
    print(f"  {d['sample_id']}: ipTM = {d['iptm']:.3f}, pTM = {d['ptm']:.3f}")

ranked = sorted(shortlist, key=lambda d: (d["iptm"] if not np.isnan(d["iptm"]) else -1), reverse=True)
passers = [d for d in ranked if d["iptm"] >= IPTM_CUTOFF] or ranked[:TOP_N]
print(f"\n{len(passers)} designs with ipTM \u2265 {IPTM_CUTOFF} (or top-{TOP_N} fallback).")

# COMMAND ----------

# DBTITLE 1,Developability + immunogenicity triage
# --- Step 4: Developability + immunogenicity (including MHC-II / ADA for antibodies) ---
# For antibodies the MHC-II / anti-drug-antibody (ADA) axis (HLAIIPred) is the key
# immunogenicity signal for a biologic; MHC-I (MHCflurry) is the CD8 axis.
DEV_WHICH = ("netsolp", "deepstabp", "pltnum", "mhcflurry", "hlaiipred")

print(f"Running developability + immunogenicity on {len(passers)} designs ...")
print(f"  Endpoints: {DEV_WHICH}")
for d in passers:
    d.update(developability(d["sequence"], which=DEV_WHICH))
    print(f"  {d['sample_id']}: sol={d.get('predicted_solubility','?'):.3f}, "
          f"Tm={d.get('predicted_tm_celsius','?'):.1f}\u00b0C, "
          f"stab={d.get('predicted_stability','?'):.3f}, "
          f"MHC-I={d.get('predicted_immuno_burden','?'):.4f}, "
          f"MHC-II/ADA={d.get('predicted_immuno_mhc2','?'):.4f}")
print("\u2705 Developability complete.")

# COMMAND ----------

# DBTITLE 1,Composite reward + ranked selection
# --- Step 5: Composite reward + ranked selection ---
df = pd.DataFrame(passers)

axis_cols = {
    "plddt":      ("mean_plddt",              True),   # higher is better
    "iptm":       ("iptm",                    True),   # higher is better
    "solubility": ("predicted_solubility",     True),   # higher is better
    "stability":  ("predicted_stability",      True),   # higher is better (0–1 ranker)
    "tm":         ("predicted_tm_celsius",     True),   # higher is better
    "immuno":     ("predicted_immuno_burden",  False),  # LOWER is better (MHC-I)
    "immuno_mhc2":("predicted_immuno_mhc2",   False),  # LOWER is better (MHC-II / ADA)
}
# Keep only axes that were actually computed
axis_cols = {k: v for k, v in axis_cols.items() if v[0] in df.columns}

df["composite_reward"] = composite_reward(df, axis_cols)
df = df.sort_values("composite_reward", ascending=False).reset_index(drop=True)

show_cols = ["sample_id", "composite_reward"] + [v[0] for v in axis_cols.values()] + ["sequence"]
display(df[show_cols])
print(f"\n\u2705 Top candidate: {df.iloc[0]['sample_id']} (composite reward = {df.iloc[0]['composite_reward']:.4f})")

# COMMAND ----------

# DBTITLE 1,Visualize top VHH–lysozyme complex (PDBe Mol*)
# --- Step 6: Render the top design in PDBe Mol* ---
# VHH = teal (chain A), lysozyme = gold (chain B), epitope = red ball-and-stick.
top = df.iloc[0]
_hot = [int(x) for x in str(HOTSPOTS).replace(" ", "").split(",") if x.strip().isdigit()]

structure = top.get("complex_pdb") or top.get("backbone_pdb") or ""
show_molstar(
    structure,
    binder_chain="A", target_chain="B",
    highlight_resids=_hot,
    binder_label="VHH (designed)",
    target_label="Lysozyme",
)

# COMMAND ----------

# DBTITLE 1,Log candidates + metrics to MLflow
# --- Step 7: Log ranked candidates and per-axis metrics to MLflow ---
import mlflow

EXPERIMENT_PATH = f"/Users/{w.current_user.me().user_name}/mlflow_experiments/antibody_design_exercise"
mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="VHH_lysozyme_RFD4") as run:
    # Log parameters
    mlflow.log_params({
        "target": "hen egg-white lysozyme (1LYZ)",
        "target_chain": TARGET_CHAIN,
        "epitope_residues": HOTSPOTS,
        "length_min": LENGTH_MIN,
        "length_max": LENGTH_MAX,
        "num_designs": NUM_DESIGNS,
        "plddt_cutoff": PLDDT_CUTOFF,
        "iptm_cutoff": IPTM_CUTOFF,
        "design_model": "RFD4-Proteina",
        "fold_model": "ESMFold",
        "complex_model": "Boltz-2",
        "dev_endpoints": ",".join(DEV_WHICH),
    })
    # Log top-candidate metrics
    top_row = df.iloc[0]
    for col in [v[0] for v in axis_cols.values()] + ["composite_reward"]:
        if col in top_row and pd.notna(top_row[col]):
            mlflow.log_metric(f"top_{col}", float(top_row[col]))
    # Log the full ranked table as an artifact
    artifact_df = df[[c for c in show_cols if c != "sequence"]].copy()
    artifact_df.to_csv("/tmp/ranked_candidates.csv", index=False)
    mlflow.log_artifact("/tmp/ranked_candidates.csv")
    # Also log the sequences
    seq_df = df[["sample_id", "sequence"]].copy()
    seq_df.to_csv("/tmp/candidate_sequences.csv", index=False)
    mlflow.log_artifact("/tmp/candidate_sequences.csv")
    print(f"\u2705 Logged to MLflow experiment: {EXPERIMENT_PATH}")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Artifacts: ranked_candidates.csv, candidate_sequences.csv")

# COMMAND ----------

# DBTITLE 1,Reflection & honest scope
# MAGIC %md
# MAGIC ## Reflection
# MAGIC
# MAGIC - **Epitope drives everything.** The hotspot residues (`19,21,23,73,75,100,101`) condition the VHH design; changing them changes the entire shortlist.
# MAGIC - **Binding vs developability.** A strong binder (high ipTM) is not automatically a good drug — solubility, thermal stability, and (critically for antibodies) **MHC-II / anti-drug-antibody (ADA) burden** matter just as much.
# MAGIC - **MHC-II vs MHC-I for biologics.** MHC-II (HLAIIPred) drives the CD4 T-cell / ADA response and is the dominant immunogenicity axis for a therapeutic antibody. MHC-I (MHCflurry) is the CD8 / cytotoxic axis — important but secondary for a biologic.
# MAGIC - **In-silico = enrichment, not measurement.** ipTM is a docking *confidence*, not an affinity; developability outputs are proxies that decide which designs to order. Wet-lab assays (SPR, DSF, ADA titering) still decide.
# MAGIC - **Known limitation.** This generates a VHH-length binder conditioned on the epitope — a first-draft VHH, not yet a full immunoglobulin-framework scaffold with fixed framework + designed CDR loops.
# MAGIC
# MAGIC ## Next steps
# MAGIC - **Compare to a real nanobody:** `fetch_pdb("1MEL")` (cAb-Lys3 bound to lysozyme) and eyeball your top design against it.
# MAGIC - **Retarget:** re-run for **RSV F site II** (therapeutic palivizumab epitope) — same prompt, new target.
# MAGIC - **Optimize a hit:** re-sample around your top design (more samples, tighter length window) and re-run the funnel.