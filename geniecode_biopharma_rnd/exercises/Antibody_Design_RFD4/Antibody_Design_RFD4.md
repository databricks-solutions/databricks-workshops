# Antibody Design using RFD4-Proteina

---
**Estimated time:** ~45 mins

**Level:** Intermediate

## Title
**From a paper to a notebook: designing single-domain antibodies (VHH / nanobodies) with RFD4-Proteina — one Genie Code prompt → a reusable, endpoints-only design notebook.**

## Objective
Starting from the RFD4-Proteina paper, use **Genie Code** to generate a complete notebook **in your own workspace** that designs nanobodies (VHH) against a chosen antigen epitope and triages them through an in-silico **binding + developability + immunogenicity** funnel. The goal is to show how a published design workflow becomes a **one-prompt, reusable asset** on Databricks — the notebook calls Genesis Workbench **only through serving endpoints** (no library import), so any attendee can regenerate it for their own paper/target.

Your target for this run: **hen egg-white lysozyme** (the canonical nanobody benchmark). The same prompt retargets to any antigen.

## Key Takeaways and Core Concepts
- **Genie Code skills (paper → notebook):** the workspace skill `protein-design-from-paper` codifies an entire design workflow once; a single prompt regenerates it for your target. Its `antibody_vhh` task drives RFD4-Proteina.
- **Serving endpoints + MCP + Genie Code:** the same funnel is reachable three ways — a notebook (this exercise), the GWB web app, and MCP tools.
- **Multi-model orchestration (the funnel):** RFD4-Proteina (epitope-conditioned VHH design) → ESMFold (monomer fold + pLDDT) → Boltz-2 (VHH–antigen co-fold + ipTM) → developability (NetSolP, DeepSTABp, PLTNUM) → immunogenicity (MHCflurry MHC-I **and HLAIIPred MHC-II / anti-drug-antibody**).
- **Composite scoring & selection:** z-score→min-max per axis, weighted sum → a ranked "orderable" shortlist.
- **Honest scope:** the in-silico funnel does **enrichment & prioritization** (which designs to order), not measurement. Developability scores are proxies; wet-lab assays still decide.

## Reference
Corley, Didi, et al., *Biomolecular design through latent reasoning with RFD4-Proteina*, 2026 (CC BY 4.0). RFD4-Proteina is the NVIDIA × Baker-lab flow-matching design model; its protein-binding benchmark (pass = ipTM ≥ 0.80, min-PAE ≤ 1.5 Å, binder RMSD ≤ 2.5 Å) is exactly the binder/antibody task you run here.

---

## Step-by-Step Guide

### Step 1 — Pick a target from the catalog
**Action:** In a notebook or Genie Code, look at the staged antigen catalog for this exercise:
```sql
SELECT target_name, therapeutic_context, antigen_source, antigen_chain, suggested_epitope_csv, pdb_path
FROM main.antibody_design.targets;
```
(The antigen PDBs are also staged in the volume `/Volumes/main/antibody_design/antigens/`.)

**Result:** Two targets — **hen egg-white lysozyme** (`RCSB 1LYZ`, chain A, epitope `19,21,23,73,75,100,101`) and **RSV F site II** (the therapeutic palivizumab epitope, chain A, `262,265,268,272,275`). Use lysozyme for your first run.


### Step 2 — Generate the design notebook from the paper (Genie Code skill)
**Action:** Open **Genie Code** in a blank notebook and prompt (force the skill with `@protein-design-from-paper` if it doesn't load automatically):

> **"Use the RFD4-Proteina paper to design a VHH nanobody against hen egg-white lysozyme (PDB 1LYZ, chain A), epitope residues 19,21,23,73,75,100,101. Generate the whole endpoints-only notebook, and log the ranked candidates + per-axis metrics to the MLflow experiment `/Users/<you>/mlflow_experiments/antibody_design_exercise`."**


Genie Code loads the workspace skill `protein-design-from-paper` and classifies this as the **`antibody_vhh`** task. *(While it generates: the same funnel is also available via the GWB web app's **Antibody Design** tab and via **MCP tools** — `endpoint_rfd4_proteina`, `endpoint_esmfold`, `endpoint_boltz`, `endpoint_hlaiipred`, … — but here we build the notebook ourselves.)*

**Result:** A complete, **endpoints-only** notebook scaffolded into your current one — no `import genesis_workbench`, just `WorkspaceClient().serving_endpoints.query(...)`. You should see these cells:
1. `%pip install -q "databricks-sdk>=0.50.0" requests` → `dbutils.library.restartPython()`
2. **Toolkit** (`get_ep`, `gen_vhh_antibodies`, `esmfold`, `boltz_complex`, `developability`, `composite_reward`, `fetch_pdb`, …)
3. **Problem definition** (`target_pdb_text = fetch_pdb("1LYZ")`, `TARGET_CHAIN="A"`, `HOTSPOTS="19,21,23,73,75,100,101"`, VHH length 110–130, `NUM_DESIGNS`)
4. **Generation** → **Funnel** → **ranked table**, with an **MLflow logging** cell (standard `mlflow.set_experiment(...)` + `start_run` to your folder from Step 1) → reflection.


### Step 3 — Read & run the generation + fold cells (RFD4 → ESMFold → Boltz-2)
**Action:** You don't have to write any code here — **Step 2 already generated these cells for you.** Open the generated **generation** and **funnel** cells, read them top to bottom, and make sure you understand what each call does; change a value only if you want to (for a class run keep **`NUM_DESIGNS = 4`**, or adjust the pLDDT / ipTM cutoffs). The generation cell looks like:
```python
raw = gen_vhh_antibodies(target_pdb=target_pdb_text, target_chain=TARGET_CHAIN,
                         hotspots=HOTSPOTS, length_min=110, length_max=130, num_samples=NUM_DESIGNS)
```
The funnel cells then run: **ESMFold** folds each VHH monomer and gates on `mean_plddt ≥ 70`; **Boltz-2** co-folds each surviving VHH + lysozyme complex and scores interface confidence (**ipTM**). Once you've read them, run the install + toolkit + problem cells, then these.

**Result:** A set of candidate VHHs ranked by interface confidence (ipTM); the top VHH–lysozyme `complex_pdb` is rendered inline in the **PDBe Mol\* viewer** (`show_molstar`, the default) — VHH teal, lysozyme gold, the epitope residues as red ball-and-stick, with a legend + sequence panel.

#### Answer key
- Expected [Notebook]($./assets/answer key)

> ⏱️ **Warm the endpoints first.** RFD4-Proteina runs on an H100 and other models autoscale to zero — a cold endpoint can take ~10 min to spin up, and `get_ep` deliberately **fails loud** if a model is missing/asleep. A VHH batch of K=4 is a few minutes once warm. (If `get_ep` errors, open the model's serving page to wake it, then re-run.)

### Step 4 — Read & run the developability + immunogenicity triage
**Action:** Nothing to write here either — the developability, immunogenicity, and scoring cells were **already generated in Step 2.** Read the generated developability cell and confirm it includes **HLAIIPred (MHC-II / anti-drug-antibody)** — the key immunogenicity axis for a biologic:
```python
DEV_WHICH = ("netsolp", "deepstabp", "pltnum", "mhcflurry", "hlaiipred")   # hlaiipred = MHC-II / ADA
```
See how the composite reward weights each axis (the MHC-II/ADA axis `immuno_mhc2` is weighted into the score); change a weight only if you want to, then run the cells to compute the ranked shortlist.

**Result:** A short, ranked list of "orderable" VHH candidates with their per-axis scores — binding (ipTM), fold (pLDDT), solubility, Tm, stability, MHC-I burden, and **MHC-II / ADA burden** — and a composite reward. These are logged to your MLflow experiment so you can reopen the shortlist later (MLflow UI or the app's Search Past Runs).

### Step 5 — Compare to a real nanobody and interpret
**Action:** Fetch the canonical camelid anti-lysozyme nanobody complex **`fetch_pdb("1MEL")`** (cAb-Lys3 bound to lysozyme) and eyeball your top designed VHH–lysozyme complex against it. Then read your shortlist: is #1 ranked there because it **binds** best (ipTM), or because it's the most **developable / least immunogenic**? Compare it to a design that binds well but scores poorly on **HLAIIPred (MHC-II / ADA)**.

**Result:** You can explain the central antibody-design trade-off — **a strong binder is not automatically a good drug** — and sanity-check your designed interface against a real nanobody–lysozyme structure.

### Step 6 (Bonus) — Optimize a hit, or retarget in one prompt
**Action:** Prompt Genie Code to **re-sample around your top design** (more samples, tighter length window) and re-run the funnel — an endpoints-only optimization loop. 

**Or** regenerate the entire notebook for a different antigen with one prompt: *"…design a VHH against RSV F site II (`/Volumes/main/antibody_design/antigens/rsv_f_site_ii.pdb`, chain A), epitope 262,265,268,272,275."*

**Result:** An improved candidate, or the **same workflow instantly retargeted** — the point of the skill: codify the research workflow once, reuse it for any paper/target.

---

## Reflection
- **Epitope drives everything.** How would the shortlist change for the RSV F site II target vs lysozyme? Why is epitope choice the most consequential input?
- **Binding vs developability.** Which reward axis would you up-weight for a therapeutic nanobody, and why is **MHC-II / anti-drug-antibody** screening especially important for a biologic (vs MHC-I)?
- **Reasoning → results.** The paper's *extrinsic reasoning* discards weak trajectories early; how is that idea mirrored by the funnel's ESMFold/Boltz gates that drop designs before the expensive steps?
- **In-silico = enrichment, not measurement.** What does a high ipTM or a low MHC-II burden actually tell you, and what still needs a wet-lab assay?
- **Known limitation.** The `antibody_vhh` task generates an epitope-conditioned VHH-length binder — a first-draft VHH, not yet a full immunoglobulin-framework scaffold with fixed framework + designed CDR loops. How would that change the problem?

## Next Steps & Resources
- **Retarget:** re-run the whole flow against **RSV F site II** (therapeutic palivizumab epitope) — same prompt, new target from the catalog.
- **The skill:** `protein-design-from-paper` (workspace skill, `@protein-design-from-paper`) — also covers protein binders, ligand binders, motif scaffolding, de novo monomers, and optimization. Broaden your prompt to explore those.
- **Endpoints used** (all on Model Serving, autoscale-to-zero, callable from notebook / app / MCP): `rfd4_proteina`, `esmfold`, `boltz`, `netsolp`, `deepstabp`, `pltnum`, `mhcflurry`, `hlaiipred`.
- **Alternatives to the notebook:** the GWB app **Large Molecule → Antibody Design** tab (guided UI) and the **MCP** server (agentic / Genie Code tool calls).
- **Paper:** *Biomolecular design through latent reasoning with RFD4-Proteina* (2026, CC BY 4.0).

## Key lessons
- A **Genie Code skill** turns a published design workflow into a one-prompt, reusable notebook — regenerate it for any paper/target.
- Generative **antibody design** runs end-to-end on **serving endpoints only**: RFD4-Proteina designs, and a family of fold / complex / developability / immunogenicity endpoints triage — orchestrated from a plain notebook, no library import.
- Designing a binder is **multi-objective**: binding, fold, solubility, stability, and (critically for antibodies) **MHC-II / ADA immunogenicity** are optimized together via an interpretable composite reward.
- The funnel does **enrichment**, deciding what to order — not measurement.
