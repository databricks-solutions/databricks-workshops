# De Novo Protein Binder Design & Optimization

---
**Estimated time:** ~45 mins

**Level:** Intermediate

## Title
**From a paper to a notebook: de novo protein binder design against a snake-venom toxin with Proteina-Complexa — one Genie Code prompt → a reusable, endpoints-only design notebook.**

## Objective
Starting from a published de novo protein-design paper, use **Genie Code** to generate a complete notebook **in your own workspace** that designs novel protein binders against a chosen target and triages them through an in-silico **developability funnel** — demonstrating how a published research workflow becomes a **one-prompt, reusable asset** on Databricks. The notebook calls Genesis Workbench **only through its serving endpoints** (no library import), so any attendee can regenerate it for their own paper/target.

Your target for this run: **α-cobratoxin** (PDB `1CTX`), the lethal neurotoxin the paper neutralizes with a de novo binder.

## Key Takeaways and Core Concepts
- **Genie Code skills (paper → notebook):** the workspace skill `protein-design-from-paper` codifies an entire research workflow once; a single prompt regenerates it for your target.
- **Model serving endpoints + MCP + Genie Code:** the same funnel is reachable three ways — a notebook (this exercise), the GWB web app, and MCP tools.
- **Multi-model orchestration (the funnel):** design → fold → complex-confidence → developability.
- **Composite scoring & selection:** weighted per-axis scores → a ranked "orderable" shortlist.
- **Apps and analysis in notebooks:** the same endpoints power the GWB app, the MCP server, and your own notebook.
- **Honest scope:** the in-silico funnel does **enrichment & prioritization** (which designs to order), not measurement; developability scores are proxies, wet-lab assays still decide.

## Reference
Vázquez Torres et al., *De novo designed proteins neutralize lethal snake venom toxins*, **Nature** 2025 (DOI 10.1038/s41586-024-08393-x). The paper designs de novo binders against three-finger snake-venom toxins; Proteina-Complexa stands in for the paper's RFdiffusion+ProteinMPNN generation, and Boltz-2 ipTM for its AlphaFold2-multimer interface filter.

---

## Step-by-Step Guide

### Step 1 — Set up MLflow logging and your GWB profile (one-time)
**Action:** Open the Genesis Workbench app → **Profile** and set your **MLflow experiment folder** (e.g. `mlflow_experiments`), then save. Pick an experiment name for this exercise — `protein_binder_exercise` — so your design funnel logs to `/Users/<you>/mlflow_experiments/protein_binder_exercise`.

**Result:** Your design runs are tracked in **MLflow** and appear under the GWB app's **Search Past Runs** (and the MLflow UI), so the ranked candidates + per-axis scores are saved and shareable — not just printed once.

### Step 2 — Generate a design notebook from the paper (Genie Code skill)
**Action:** Open **Genie Code** in a blank notebook and prompt (force the skill with `@protein-design-from-paper` if it doesn't load automatically):

> **"Use this paper to design a protein binder against α-cobratoxin (PDB 1CTX), hotspots 30,33,36. Generate the whole endpoints-only notebook, and log the ranked candidates + per-axis metrics to the MLflow experiment `/Users/<you>/mlflow_experiments/protein_binder_exercise`."**

Genie Code loads the workspace skill `protein-design-from-paper` and classifies this as the **`binder_vs_protein`** task. *(While it generates: the same funnel is also available via the GWB web app's **Protein Binder Design** tab and via **MCP tools** — `endpoint_proteina_complexa`, `endpoint_boltz`, `endpoint_esmfold`, … — but here we build the notebook ourselves.)*

**Result:** A complete, **endpoints-only** notebook scaffolded into your current one — no `import genesis_workbench`, just `WorkspaceClient().serving_endpoints.query(...)`.

> 💡 **Takeaway:** A Genie Code skill codifies an entire research workflow once; any attendee can regenerate it for their own paper/target with a single prompt.

### Step 3 — Generate binders and validate the fold (design → ESMFold → Boltz-2)
**Action:** Run the generated cells. **Proteina-Complexa Binder** generates candidate binders (backbone + sequence); **ESMFold** folds each and gates on pLDDT; **Boltz-2** co-folds each binder–toxin complex and scores interface confidence (**ipTM**). For a class run, keep `NUM_DESIGNS` small (6–8).

**Result:** A set of candidate binders ranked by interface confidence (ipTM), with the top binder–toxin complex visualized in the notebook in the **PDBe Mol\* viewer** (`show_molstar`, the default) — binder teal, α-cobratoxin gold, the hotspot residues as red ball-and-stick, with a legend + sequence panel.

> ⏱️ **Warm the endpoints first.** Proteina-Complexa runs on a GPU and the models autoscale to zero — a cold endpoint can take ~10 min to spin up, and the toolkit's `get_ep` deliberately **fails loud** if a model is missing/asleep.

> 💡 **Takeaway:** GWB serves a whole family of protein models and the notebook orchestrates them as a funnel. Proteina-Complexa replaces the paper's RFdiffusion+ProteinMPNN, and Boltz-2 ipTM stands in for the AlphaFold2-multimer interface filter.

### Step 4 — Developability triage, selection, and comparison to the paper
**Action:** Run the surviving designs through the developability endpoints — **NetSolP** (solubility), **DeepSTABp** (Tm), **PLTNUM** (stability), **MHCflurry** (immunogenicity). Compute a composite reward and pick finalists. Then fetch the authors' deposited binder–toxin complex **`fetch_pdb("9BK5")`** and compare it to your top design.

**Result:** A short, ranked list of "orderable" binders with their per-axis scores, and a side-by-side with the real, experimentally validated design — logged to your MLflow experiment from Step 1.

> 💡 **Takeaway:** The in-silico funnel does **enrichment and prioritization** — deciding which designs to order — not measurement. Developability scores are proxies (PLTNUM is a relative ranker, not hours); wet-lab assays still decide.

### Step 5 (Optional) — Optimize a promising hit
**Action:** Prompt Genie Code to **re-sample binders around your top design** (an endpoints-only partial-diffusion analog), or dispatch GWB's reward-weighted **enzyme-optimization job** for a production refinement loop.

**Result:** An improved candidate, or a dispatched optimization run to review later.

> 💡 **Takeaway:** The same endpoints support iterative optimization; Databricks Jobs turn a long-running refinement loop into a scheduled, reviewable asset.

---

## Reflection
- **Binding vs developability.** Is your #1 ranked there because it *binds* best (ipTM), or because it's the most *developable*? Which axis would you up-weight for a therapeutic, and why?
- **Model substitutions.** The paper uses RFdiffusion+ProteinMPNN and AlphaFold2-multimer; you used Proteina-Complexa and Boltz-2. What does each substitution preserve, and what does it change?
- **What ipTM means.** A high ipTM is a *confidence* that the complex is modeled well — not a measured affinity. What does it actually tell you, and what still needs a wet-lab assay?
- **In-silico = enrichment, not measurement.** How does the funnel's ESMFold/Boltz gating mirror the way the paper discards weak designs before expensive validation?
- **Retargeting.** How would the shortlist change for a different three-finger toxin? Why is the hotspot choice so consequential?

## Next Steps & Resources
- **Retarget:** re-run the whole flow against another snake-venom toxin or any protein target — same prompt, new PDB + hotspots.
- **The skill:** `protein-design-from-paper` (workspace skill, `@protein-design-from-paper`) — also covers single-domain antibodies (VHH), ligand binders, motif scaffolding, de novo monomers, and optimization.
- **Models used** (all on Databricks Model Serving, autoscale-to-zero, callable from notebook / app / MCP): **Proteina-Complexa** (de novo binder generation; the RFdiffusion+ProteinMPNN analog), **ESMFold** (monomer fold + pLDDT), **Boltz-2** (complex co-fold + ipTM/pTM), and developability models **NetSolP, DeepSTABp, PLTNUM, MHCflurry**.
- **Target:** α-cobratoxin (PDB `1CTX`); the paper's deposited design complex is PDB `9BK5`. Both are fetched from RCSB at runtime — no data staging needed.
- **Alternatives to the notebook:** the GWB app **Large Molecule → Protein Binder Design** tab (guided UI) and the **MCP** server (agentic / Genie Code tool calls).
- **Paper:** *De novo designed proteins neutralize lethal snake venom toxins* (Nature 2025).

## Key lessons
- A **Genie Code skill** turns a published design workflow into a one-prompt, reusable notebook — regenerate it for any paper/target.
- De novo **binder design** runs end-to-end on **serving endpoints only**: Proteina-Complexa designs, and a family of fold / complex / developability endpoints triage — orchestrated from a plain notebook, no library import.
- Designing a binder is **multi-objective**: binding (ipTM), fold (pLDDT), solubility, stability, and immunogenicity are optimized together via an interpretable composite reward.
- The funnel does **enrichment**, deciding what to order — not measurement.
