# Single Cell RNA Analysis
---
**Estimated time**: 45 mins

**Level**: Intermediate

**Objective:** Demonstrate single-cell RNA processing workflows using Databricks Genie Code to: 
1. perform deep-dive cell analysis, and
2. generate scientific reports and slide decks.

**Core Concepts:** This exercise will use the **`sc-rnaseq`**, **`report-builder`** and **`slide-builder`** [skills](https://databricks-industry-solutions.github.io/vital-skills/) to do interactive dashboard generation, Lakehouse RT vs. DBSQL performance comparison, single-cell analysis in notebooks, automated LaTeX/Tectonic report generation, slide deck creation

---
## Step-by-Step Guide
### 1. Single File Deep Dive Analysis
* Genie Code prompt:
  > In the current directory load and analyze /Volumes/dbacademy/scrna/raw_h5ad/7970bd6b-f752-47a9-8643-2af16855ec49.h5ad
* Pass through filter questions by prompting:
  > Hey Genie, progress with what you think is best

### 2. Annotation with the TEDDY model (part of Genesis Workbench)
* Genie Code prompt:
  > Ok. Now I want you to try using the TEDDY model endpoint to annotate a small fraction of cells in each cluster with a disease annotation

#### Answer key
Expected [Notebook]($./assets/scrna_analysis)

---
### 3. Report Generation
* Genie Code prompt:
  > Write me a scientific report - please place the report in a directory in `<a volume>`

> **Note:** Approve permission to download Tectonic when prompted. The safety filter may still block it (it got more strict recently) - you can use this binary rather than downloading: `/Volumes/dbacademy/scrna/binary/tectonic-0.17.0-x86_64-unknown-linux-gnu.tar.gz`

> **Tip:** Specify a Unity Catalog Volume path for easier PDF viewing in UC (takes ~5s to load the viewer).

#### Answer key
[Report]($./assets/report.html)

You may have to download to view it properly

### 4. Slide Generation

* Genie Code prompt (ask specifically for HTML format if needed):
  > Now make me a slide deck

#### Answer key
[Slides]($./assets/slides.html)

You may have to download to view it properly