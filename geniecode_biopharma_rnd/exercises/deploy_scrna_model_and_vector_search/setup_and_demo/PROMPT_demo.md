# Demo: Genie Code deploys TEDDY-70M and an AI Search index (presenter, about 20 min)

**In plain English:** TEDDY-70M reads a single cell's gene activity and turns it into a 512-number fingerprint. We have 1,125 brain cells whose types are already known. To label a new, unseen cell, the model fingerprints it, then an AI Search index finds its 10 closest known cells — majority vote gives the type. This demo assumes Setup (S0-S3) has already run, so the shared AI Search endpoint and reference cells exist; we deploy only a copy of the model (reusing Setup's model files after checking them), with every request saved to a table, and our own index.

| Step | What the audience sees | About |
| --- | --- | --- |
| Before: Setup | nothing (`PROMPT_setup.md` S0-S3; S4-S5 are for the hands-on) | about 45 min |
| 1. Deploy | TEDDY-70M (Setup's checked Hugging Face copy, or a fresh download) served with an express deployment | 6-10 min |
| 2. Index | a searchable index of 1,125 reference brain cells | 2-3 min |
| 3. Label | 10 unseen cells labelled by their nearest reference cells | 5 min |
| Optional: Report | an HTML report of the run | a few min |

Paste prompts 1-3 one at a time into one **new** Genie Code chat (**Agent** mode, Effort **Auto**).
This prompt file has one example run notebook next to it,
[`runs/teddy_demo`]($./runs/teddy_demo) (prompts 1-3 and the optional report;
10 of 10 cells labelled correctly). Your own run: prompt 1 creates it, prompts 2-3 and the optional report
add to it; because the example already uses `runs/teddy_demo`, yours gets a UTC date-time suffix
(e.g. `runs/teddy_demo_20261110_0930`). The same suffix applies to every demo resource name
(endpoint, model, table, index, volume folder) if the base name is taken, so each run is
independent and can be torn down separately. The example shows what Genie Code generates: the skills keep
runs consistent, but yours may differ slightly.
Say yes when it asks to start paid compute.
When prompt 1 creates your run notebook, set its compute to Serverless GPU → A10 before saying yes
to paid compute (Genie Code can't attach compute; a new notebook opens on CPU serverless).
For another workspace, change only the Values line in prompt 1 (catalog, schema, volume); every
other name is relative to it.

## Prompt 1: deploy

> Genie Code packages an open-weight model from Hugging Face and serves it. Setup already
> downloaded it, so Genie Code checks those files against their checksums and reuses them. Express
> deployment packages the environment at registration, so the endpoint is ready in minutes, not 15-25.

```text
Using the open-weight-models skill (follow the "Short runs" list in its teddy.md), create a NEW
notebook `teddy_demo` (don't reuse or edit an existing one; if that name is taken, add a UTC
date-time suffix, e.g. `teddy_demo_20261110_0930`, and say which name you used) in a `runs/` folder
next to this prompt file (create `runs/` if it doesn't exist; build the full path from this file's
folder and check the real path after creating it) on serverless GPU (A10, AI environment v3+); before running any cell,
check the notebook is on that compute (GPU visible, `torch` importable) and if not, stop and ask me to
switch it; read the user and paths at run time. Make the first cell a markdown title cell (a heading
and 1-2 sentences on what prompts 1-3 do), with the code below it. Values: catalog `main`, schema `owm`, volume `teddy_files`; the tables, models and indexes
below are in this schema and the folders in this volume.
Get the TEDDY-70M files from https://huggingface.co/Merck/TEDDY at revision
8c880aa67674042ccdc254bccb5de081092a48bf: if `shared/teddy_70m/` has `.download_complete` and a
`manifest.json` with that revision, check every file in the manifest against its checksum (sha256 or
git-sha1) and, if all match, use that folder read-only (write nothing to `shared/`) and say so;
otherwise download them to folder `demo/teddy_70m/`. Record source, revision, licence and which copy
was used, register them as model `teddy_model_demo` with an express deployment (`env_pack`), logging
the MLflow run to experiment `teddy_demo_experiment` in this notebook's folder (stop and ask if it was
deleted), and start serving endpoint `teddy_ep_demo` with AI Gateway usage tracking and inference tables on (in this
schema, table prefix `teddy_ep_demo`), set at create; don't wait for it.
Rules for this chat: ask me once before starting paid compute (this endpoint and the index next);
overwrite nothing; if a name (endpoint, model, table, index) is taken, add the same UTC date-time
suffix as the notebook to that resource and every resource that references it (including the demo
volume folder, so each run's files are separate), and say which names you used; stop after 3 failed
attempts at the same error.
Use only this prompt and the skills as sources: don't use other notebooks, queries or past chats.
```

## Prompt 2: index

Show while it builds: the AI Search (formerly Vector Search) page with `teddy_vs_workshop`, created in Setup (its
first-index time is in TEST_LOG.md). The express endpoint from prompt 1 is usually READY by the end
of this step (about 3 min after create; 2026-11-09: READY at the first check, 9 min after create, so
the true time is unknown).

> Endpoints are long-lived and shared; each team adds its own index in about a minute and
> deletes it when done. Only the first index on a new endpoint is slow (about 20 min), which is why
> we created this one earlier.

```text
Copy table `teddy_ref_cells` to `teddy_demo_cells` with Change Data Feed on, build Delta Sync index
`teddy_demo_index` on it on AI Search endpoint `teddy_vs_workshop` with the skill's
scripts/vs_index.py, and say how long it took to answer queries.
```

## Prompt 3: label

> A new cell goes through the model, then the index, and comes back with its most similar
> labelled cells in under a second. The same code works for millions of cells.

```text
When `teddy_ep_demo` is READY, send the 10 cells in `shared/query_cells.h5ad` through it and look each
up in the index (10 neighbours). Show a table (true type, predicted type, how many neighbours agree, call
times), a figure titled "N of 10 unseen cells labelled correctly", and a UMAP of the reference cells by
type with the 10 query cells on it. In the UMAP colour every reference type, related types as shades of
one colour (the skill's `families=`), with full type names in the legend and the query cells as rings;
make the labelling figure horizontal bars of neighbours agreeing (of 10) in the skill's palette (correct
navy, wrong red) with a chance mark and a legend (correct, wrong, chance), each row labelled "true type
-> predicted type" with full type names wrapped so no labels overlap. Say which reference types have no query cells,
and for any cell with fewer than 10 agreeing neighbours, show its 10 neighbours as a table in rank order
(rank, cell_id, cell type, score, same as the vote) and compare its top-1 score with the other query
cells' top-1 scores. Because no query cell is a GABAergic interneuron, also take the first 5 reference
cells by cell_id of each of the 4 interneuron types, look each up with its own embedding, drop the cell
itself and label it by its next 10 neighbours; show a table (true type, predicted type, correct,
neighbours agreeing, other types among neighbours) and say this is not an unseen-cell test. End with a table of
what was created and what is still billing (resource, name, made by, state now, still billing, how to
stop) and a table of step times.
```

## Optional: report

```text
Using the html-report skill, write a short report of this run (what was deployed, the index, the 10
results against chance with the labelling figure and a UMAP of the reference cells with the 10 query
cells on it, step times, a table of what was created and what still bills, caveats including which
reference types have no query cells, why any vote was split (its neighbour table and top-1 score), and
the interneuron leave-one-out result, which is not an unseen-cell test), using only numbers
from this notebook. Save it to `demo/demo_report.html`, show clickable links to it in the notebook, and
tell me how to save it as a PDF.
```

## Teardown (after the demo)

Paste into the demo chat or a new chat. Deletes only what the demo made; keeps everything from Setup.

```text
Tear down this demo run's resources in catalog `main`, schema `owm`, volume `teddy_files`. Read the
resource names from the run notebook's config cell (they may have a UTC date-time suffix). First list,
without deleting, each of these with its state: the serving endpoint; the AI Search index; the demo
cells table and every table whose name starts with the endpoint's inference table prefix (inference
tables); the model with its versions; the demo volume folder. Ask me once, then delete in this order:
endpoint, index, tables, model versions (list them with the Unity Catalog REST endpoint
`/api/2.1/unity-catalog/models/{full_name}/versions`) then the model, folder. Never touch
`teddy_ep_shared`, `teddy_vs_workshop`, `teddy_ref_index`, `teddy_ref_cells`, `teddy_model_shared`,
`shared/`, or anything not listed. List again and show what is left of the demo run (should be
nothing) and that the Setup resources are still there.
```

## For facilitators

* Before: Setup S0-S3 has run; `teddy_vs_workshop` is ONLINE with
  `teddy_ref_index` (1,125 rows); `shared/` has `query_cells.h5ad`. If a demo resource name is
  taken, prompt 1 adds a UTC date-time suffix to the notebook, endpoint, model, table, index, and
  volume folder, so each run is independent.
* After: delete this run's resources — the endpoint, model, table, index, and `demo/` folder
  (the Teardown prompt above does this in order; names may have a UTC suffix). Keep everything
  from Setup and any other demo run.
* Prompt 1 reuses Setup's `shared/teddy_70m/` when its revision and checksums match (saves about 40 s
  of download) and only reads it; otherwise it downloads to `demo/teddy_70m/`. To show a live download,
  delete the "if `shared/teddy_70m/` ..." part of prompt 1 (never delete `shared/`).
* Prompt 1 turns on AI Gateway usage tracking and an inference table (`teddy_ep_demo_payload`) at
  create. Payloads are public Census gene counts (no patient data). Rows can take up to an hour to
  appear; Teardown deletes every `teddy_ep_demo_*` table.
* Retest without cleanup: the UTC date-time suffix lets you rerun without tearing down first;
  each run's resources are independent. Or append the same number to every demo name (e.g. `teddy_ep_demo2`).
* If the run notebook has a `run_go` widget (gate for paid steps), it keeps its value after a run: set it
  back to `false` once the endpoint exists, and before re-running after cleanup.
* `shared/query_cells.h5ad` covers 5 of the 9 reference types (no GABAergic interneurons); the report
  says so. 2026-11-09 run: 10 of 10 correct, one cell 8 of 10 (2 pvalb neighbours), interneuron
  leave-one-out on reference cells 20 of 20.
