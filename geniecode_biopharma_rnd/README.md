# Genie Code for biopharma R&D

Hands-on exercises from the Databricks and NVIDIA workshop on AI-assisted R&D workflows in biopharma. Each exercise
gives the Genie Code prompts to paste and an answer key (the expected notebook, figures and results) for one R&D
task: small-molecule screening, antibody and protein-binder design, single-cell RNA analysis, cell and protein
embedding search, and variant calling.

This workshop deploys nothing: there is no bundle and no **Open in Databricks** button. You open an exercise in a
workspace that has the resources below, and prompt Genie Code.

## Exercises

Start with [exercises/SETUP.md](exercises/SETUP.md) (about 10 minutes), then pick an exercise (30-45 minutes each).
The list, with levels, is in [exercises/README.md](exercises/README.md).

| Exercise | What you do |
|---|---|
| [ADMET virtual screening](exercises/KERMT_ADMET_virtual_screening/KERMT.md) | Score small molecules with the KERMT model, relate predicted toxicity to the other ADMET properties, schedule the scoring as a job |
| [Antibody design](exercises/Antibody_Design_RFD4/Antibody_Design_RFD4.md) | Design nanobodies against an antigen with RFD4-Proteina and rank them through a structure and developability funnel |
| [Transcriptomics vector search](exercises/deploy_scrna_model_and_vector_search/README.md) | Embed single cells with TEDDY and label them by their nearest cells in an AI Search index |
| [Single-cell RNA ingestion](exercises/scRNA_ingestion/scRNA_ingestion.md) | Load h5ad files into Delta tables and build a dashboard |
| [Single-cell RNA analysis](exercises/scRNA_analysis/scRNA_analysis.md) | Quality control, clustering, marker genes, a report and slides for one sample |
| [Variant calling](exercises/Variant_Calling_Parabricks/Variant_Calling_Parabricks.md) | Call variants with Parabricks on a GPU and annotate them with ClinVar |
| [Protein embedding and search](exercises/protein_vector_search/protein_vector_search.md) (bonus) | Embed antibody sequences with ESMC and search them in Lakebase |
| [Protein binder design](exercises/De_Novo_Protein_Binder/De_Novo_Protein_Binder.md) (bonus) | Design and optimise a protein binder with Proteina-Complexa |

## What you need

These exercises were built for a prepared lab workspace and need more than a Databricks Free Edition workspace gives
(see [USER_GUIDE.md](../USER_GUIDE.md)):

- **Genie Code with the [Vital Skills](https://github.com/databricks-industry-solutions/vital-skills)** added under
  Customizations ([exercises/SETUP.md](exercises/SETUP.md)).
- **[Genesis Workbench](https://github.com/databricks-industry-solutions/genesis-workbench)** deployed in the
  workspace, with its model serving endpoints (for example KERMT, RFD4-Proteina, Proteina-Complexa, ESMFold, Boltz-2,
  TEDDY), its app and its `mcp-genesis-workbench` MCP server, and its Parabricks job for variant calling.
- **GPU compute:** serverless GPU (A10) for the notebooks that run a model in the notebook, and GPU model serving.
- **The lab's data and shared resources**, which the prompts name: schemas in catalog `main` (`owm`, `kermt`,
  `antibody_design`, `genomics_hemochromatosis`, `genesis_workbench`), the raw h5ad files in `dbacademy.scrna`, and a
  schema of your own to write to (`dbacademy.<you>` in the lab). In another workspace, change these names in the
  prompts.
- **For protein embedding and search:** an ESMC model serving endpoint and a Lakebase project with Lakebase Search
  turned on. Optionally, the ChEMBL MCP from Marketplace for ADMET screening.

## What the exercises create, and what keeps billing

- **In your own schema:** tables, volumes and files (figures, reports). Storage only.
- **ADMET screening, task 5** schedules a job to run every night: it runs until you pause or delete it.
- **Transcriptomics vector search:** participants write only a table and files. The facilitator's setup creates a
  shared serving endpoint, an AI Search endpoint and an index in `main.owm`, which bill until deleted; see that
  exercise's README.
- **Variant calling** runs Genesis Workbench's GPU job, which bills while it runs.
- **Protein embedding and search** writes tables and an index in your Lakebase project.
- Genie Code also saves the notebooks it writes in your home folder, and some exercises log MLflow experiments there.

## Tear down

Nothing here is deployed by a bundle, so there's no **Destroy**. When you're done, delete by hand:

- your tables and volumes (or your whole schema);
- the nightly job from the ADMET exercise (**Jobs & Pipelines**);
- your tables and index in Lakebase;
- the notebooks and MLflow experiments Genie Code made in your home folder.

Facilitators: after the demo, run the Teardown prompt in
[setup_and_demo/PROMPT_demo.md](exercises/deploy_scrna_model_and_vector_search/setup_and_demo/PROMPT_demo.md); after
the workshop, delete the setup resources it lists (`teddy_ep_shared`, `teddy_vs_workshop`, `teddy_ref_index`).

## How to get help

Databricks support does not cover this content. For questions or bugs, open a GitHub issue. The team will help on a
best effort basis.
