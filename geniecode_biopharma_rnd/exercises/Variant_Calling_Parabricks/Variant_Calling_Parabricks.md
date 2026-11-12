# GPU Variant Calling with Parabricks — From Raw Reads to a Genetic Diagnosis

---
**Estimated time:** ~45 mins

**Level:** Intermediate

## Title
**From a paper to a notebook: GPU-accelerated germline variant calling (NVIDIA Parabricks + DeepVariant) that finds a real hereditary-hemochromatosis genotype — one Genie Code prompt → a reusable, Glow-free genomics notebook.**

## Objective
Starting from the **DeepVariant** paper, use **Genie Code** to generate a complete notebook **in your own workspace** that calls germline variants from real sequencing reads on **GPU** (NVIDIA Parabricks `fq2bam` + DeepVariant), then annotates the calls against **ClinVar** — and lands on a genuine clinical finding. The notebook drives Genesis Workbench's **GPU variant-calling job** and does all the post-calling analysis in **pure Python / pandas (no Glow, no Spark)**, so any attendee can regenerate it for their own sample, gene, or paper.

Your sample for this run: **`HG00264`** — a real 1000 Genomes individual (British ancestry) who is a **homozygous carrier of the HFE C282Y variant**, the classic cause of **hereditary hemochromatosis (iron overload)**.

## Key Takeaways and Core Concepts
- **Genie Code skills (paper → notebook):** the workspace skill `variant-calling-from-paper` codifies the whole *reads → clinically-annotated variants* workflow once; a single prompt regenerates it for your sample.
- **GPU secondary analysis:** Parabricks runs the GATK-Best-Practices alignment (`fq2bam`) and the **DeepVariant** deep-learning caller on a GPU — the same germline pipeline, in **minutes instead of hours**.
- **Genesis Workbench as a service:** the notebook dispatches GWB's deployed **`run_parabricks` GPU job** (Jobs API) — you need no container and no GPU of your own.
- **Glow-free clinical annotation:** the VCF is parsed with plain Python into a DataFrame and joined to a **ClinVar** table — small data, no distributed engine.
- **Genotype ≠ variant:** a ClinVar "Pathogenic" label describes the *variant*; **zygosity and inheritance** decide what it means for the person (HFE hemochromatosis is recessive — two C282Y copies).
- **Honest scope:** germline SNV/indel calling on region-subset reads; this enriches/flags findings to confirm, it is not a clinical diagnosis.

## Reference
Poplin, Chang, Alexander, et al., *A universal SNP and small-indel variant caller using deep neural networks*, **Nature Biotechnology** 2018 (DOI 10.1038/nbt.4235). DeepVariant reframes variant calling as image classification over pileups; NVIDIA Parabricks provides the GPU-accelerated implementation (`pbrun deepvariant`) used here. Clinical interpretation uses **ClinVar** (Landrum et al.).

---

## Step-by-Step Guide

### Step 1 — Set up your workspace profile and MLflow logging (one-time)
**Action:** Open the Genesis Workbench app → **Profile** and set your **MLflow experiment folder** (e.g. `mlflow_experiments`), then save. Pick an experiment name for this exercise — `variant_calling_exercise` — so your annotated findings log to `/Users/<you>/mlflow_experiments/variant_calling_exercise`.

**Result:** Your run's pathogenic-variant table is tracked in **MLflow** and is reopenable/shareable later, not just printed once.

### Step 2 — Look at the staged sample and the ClinVar reference
**Action:** In a notebook or Genie Code, inspect the exercise data (staged to a dedicated schema and granted to all workshop users):
```sql
SELECT sample_id, population, target_gene, variant_name, variant_locus,
       expected_genotype, expected_clinvar, condition
FROM main.genomics_hemochromatosis.sample_catalog;

-- the ClinVar annotation reference for the HFE locus
SELECT pos, ref, alt, rs, gene, clnsig
FROM main.genomics_hemochromatosis.clinvar_hfe
WHERE clnsig LIKE '%athogenic%'
ORDER BY pos;
```
The reads, reference, known-sites, and a pre-computed reference VCF live in the volume `/Volumes/main/genomics_hemochromatosis/data/`.

**Result:** One sample — **HG00264 (GBR)** — with the target gene **HFE** and the variant of interest **C282Y (chr6:26,092,913 G>A, rs1800562)**, expected genotype **1/1 (homozygous)**. The ClinVar table lists C282Y as **"Pathogenic / low-penetrance | risk factor"** for Hemochromatosis type 1.

### Step 3 — Generate the genomics notebook from the paper (Genie Code skill)
**Action:** Open **Genie Code** in a blank notebook and prompt (force the skill with `@variant-calling-from-paper` if it doesn't load automatically):

> **"Use the DeepVariant paper to call germline variants from the HG00264 HFE-locus reads and find any pathogenic HFE variant, then log the pathogenic findings to the MLflow experiment `/Users/<you>/mlflow_experiments/variant_calling_exercise`. Paired FASTQ: `/Volumes/main/genomics_hemochromatosis/data/reads/HG00264_HFE_R1.fastq.gz` and `..._R2.fastq.gz`. GRCh38 reference (with BWA index): `/Volumes/main/genesis_workbench/gwas_reference/genomes/GRCh38_full_analysis_set_plus_decoy_hla.fa`. Known-sites: `/Volumes/main/genomics_hemochromatosis/data/reads/known_sites.vcf.gz`. Annotate against the ClinVar table `main.genomics_hemochromatosis.clinvar_hfe`. Keep it Glow-free."**

**Result:** A complete, **Glow-free** notebook scaffolded into your current one — no `import genesis_workbench`, no Glow. You should see these cells:
1. `%pip install -q "databricks-sdk>=0.50.0" pandas` → `dbutils.library.restartPython()`
2. **Toolkit** (`resolve_job`, `launch_variant_calling`, `wait_for_run`, `read_vcf`, `load_clinvar_table`, `annotate`, `zygosity`)
3. **Problem definition** (the FASTQ / reference / known-sites / output dir / ClinVar table / target gene)
4. **Launch variant calling** → **Annotate + interpret**, with an **MLflow logging** cell → reflection.

### Step 4 — Launch GPU variant calling (Parabricks fq2bam → DeepVariant)
**Action:** Run the install + toolkit + problem cells, then the launch cell. It writes the job's input config and dispatches the deployed **`run_parabricks`** GPU job:
```python
run_id = launch_variant_calling(FQ1, FQ2, REF, OUT_DIR, RUN_TAG, known_sites=KNOWN_SITES)
wait_for_run(run_id)      # fq2bam + DeepVariant on GPU
vcf_path = f"{OUT_DIR}/deepvariant.vcf"
```
Because the reads are subset to the HFE locus, the GPU run finishes in a **few minutes** and writes `deepvariant.vcf` (+ `fq2bam.bam`, `recal.txt`).

**Result:** A VCF of the germline variants in the HFE region. (The call `chr6:26,092,913 G>A` should appear with genotype `1/1`, `PASS`.)

> ⏱️ **Shared-GPU note.** The GPU job takes its inputs from one shared config file, so if several people launch at the *exact same moment* the runs can collide. In a class, either launch one at a time, **or** set `USE_PRECOMPUTED_VCF = True` in the notebook to read the pre-computed VCF at `/Volumes/main/genomics_hemochromatosis/data/reference_vcf/HG00264_HFE_deepvariant.vcf` and go straight to annotation.

### Step 5 — Annotate the variants with ClinVar (pure Python)
**Action:** Run the annotation cells. The VCF is parsed into a pandas DataFrame and inner-joined to the ClinVar table on `(chrom, pos, ref, alt)`; pathogenic calls are filtered out with their zygosity:
```python
calls   = read_vcf(vcf_path)
clinvar = load_clinvar_table("main.genomics_hemochromatosis.clinvar_hfe")
ann     = annotate(calls, clinvar)
pathogenic = ann[ann["clnsig"].str.contains("athogenic", na=False)]
```
**Result:** A short table of clinically significant findings. The headline row is **chr6:26,092,913 G>A — HFE C282Y — homozygous (GT 1/1) — "Pathogenic / low-penetrance | risk factor" — Hemochromatosis type 1.**

### Step 6 — Interpret the finding
**Action:** Read the result as a clinician would. C282Y is autosomal recessive: one copy = unaffected carrier, **two copies (homozygous) = the at-risk genotype** for iron overload. Note the review status ("multiple submitters, no conflicts") and the "low penetrance" qualifier.

**Result:** You can explain the finding — **HG00264 is a C282Y homozygote, consistent with hereditary hemochromatosis risk** — and articulate why *zygosity and inheritance*, not the ClinVar label alone, drive the interpretation. Log the findings table to your MLflow experiment from Step 1.

### Step 7 (Bonus) — Retarget in one prompt
**Action:** Ask Genie Code to re-run the annotation against a **different gene panel** (e.g. the broader **ACMG SF** secondary-findings list) or a different sample — the same workflow, a new target. Or point the launch step at your own paired FASTQ.

**Result:** The **same workflow instantly retargeted** — the point of the skill: codify the research pipeline once, reuse it for any sample/gene/paper.

---

## Reflection
- **Zygosity matters.** How would your interpretation change if HG00264 were *heterozygous* for C282Y? Why does that distinction matter more for a recessive condition than a dominant one?
- **Caller choice.** DeepVariant is a learned model; GATK HaplotypeCaller is a statistical one. What does framing variant calling as *image classification over read pileups* buy you (the paper's central idea)?
- **GPU value.** The scientific result is identical to a CPU run — so what exactly is Parabricks selling, and when does it matter (cohort size, turnaround, cost)?
- **In-silico = enrichment, not diagnosis.** A ClinVar "Pathogenic" on a called variant tells you what to confirm. What would you still do before acting clinically (orthogonal assay, phasing, clinical correlation)?
- **Region subset.** We called only the HFE locus for speed. What do you gain, and what do you give up, versus a whole-genome run?

## Next Steps & Resources
- **Retarget:** re-run against the **ACMG SF v3.2** panel, or annotate a whole-gene-panel VCF.
- **The skill:** `variant-calling-from-paper` (workspace skill, `@variant-calling-from-paper`) — reusable for any sample, gene, or paper.
- **Resources used:** the GPU job `run_parabricks` (Parabricks `fq2bam` + DeepVariant), the staged schema `main.genomics_hemochromatosis` (sample catalog + ClinVar table + reads + reference VCF), and the shared GRCh38 reference in `main.genesis_workbench.gwas_reference`.
- **Alternatives to the notebook:** the GWB app **Genomics → Variant Calling** tab (guided UI) launches the same GPU job.
- **Papers:** *DeepVariant* (Poplin et al., 2018); **ClinVar** (Landrum et al.); HFE hemochromatosis (OMIM 235200).

## Key lessons
- A **Genie Code skill** turns a published genomics pipeline into a one-prompt, reusable notebook — regenerate it for any sample/gene/paper.
- GPU **variant calling** (Parabricks `fq2bam` + DeepVariant) runs as a Genesis Workbench **job** you dispatch from a notebook — no container, no GPU of your own.
- The clinical half is **Glow-free**: parse the VCF with Python, join ClinVar with pandas — small data needs no distributed engine.
- A real sample gives a real finding: **HG00264 is a homozygous HFE C282Y carrier** — and *zygosity + inheritance*, not the ClinVar label alone, make it meaningful.
