# TEDDY-70M hands-on session: deploy a single-cell model and an AI Search index

In this hands-on session Genie Code deploys TEDDY-70M, an open-weight single-cell foundation model
([Merck/TEDDY](https://huggingface.co/Merck/TEDDY), paper
[arXiv:2503.03485](https://arxiv.org/abs/2503.03485)), on Databricks Model Serving, builds an AI Search
(formerly Vector Search) index of CELLxGENE Census brain cells, and labels new cells by their nearest reference cells.

**In plain English:** TEDDY-70M reads a single cell's gene activity and turns it into a 512-number fingerprint. Setup fingerprints 1,125 brain cells whose types are already known and puts them in an AI Search index (`teddy_ref_index`). To label a new cell, the model (served as `teddy_ep_shared`) fingerprints it, the index finds its 10 closest known cells, and majority vote gives the type. In the hands-on you do exactly that with the shared endpoint and index; you deploy nothing.

All prompts are here so you can see how the whole workflow was built.

**Participants run only [`[Attendees] PROMPT_handson.md`]($./[Attendees] PROMPT_handson.md).** You can go [there]($./[Attendees] PROMPT_handson.md) to begin your exercise now or read on for more context.

The facilitator's prompts are in `setup_and_demo/`. The facilitator has already run Setup and the demo in this workspace; the
hands-on uses what Setup built. The hands-on creates only your own notebook, table and
folder, deploys nothing and needs no GPU. Plan on about 30 min (times vary).

One caveat when you read the results: the hands-on labels 7 closely related interneuron subtypes
from one tissue, and the expected Census type for In7 (VIP) is approximate. The % matching says
nothing about cross-tissue or rare-type use.

## Files here

| File | What it is | Who | You |
| --- | --- | --- | --- |
| `README.md` | This page | everyone | read |
| `setup_and_demo/PROMPT_setup.md` | S0-S5: clean the schema, download TEDDY-70M (or reuse a checksum-checked earlier download), sample Census cells, build the shared index and serving endpoint, check the endpoint against TEDDY's tutorial (about 70 min, GPU) | facilitator, once, before the session | read; don't paste (already run) |
| `setup_and_demo/PROMPT_demo.md` | Demo: deploy TEDDY-70M with an express deployment, add an index on the Setup AI Search endpoint, label 10 unseen cells; includes a Teardown box (about 20 min) | presenter, before or during the session | read; don't paste (already run) |
| **`[Attendees] PROMPT_handson.md`** | **H1: embed TEDDY's 3,538 tutorial cells through the shared endpoint; H2: label 200 of them with the shared Census index; optional HTML report** | **participants** | **open it here and paste its boxes into Genie Code (no copy needed; don't edit it)** |
| `setup_and_demo/runs/` | Reference output of the facilitator's prompts, with their results (`teddy_setup` from `PROMPT_setup.md`, `teddy_demo` from `PROMPT_demo.md`) | everyone | read; don't edit |
| `runs/` | Reference output of the hands-on: `teddy_handson`, a test run of `[Attendees] PROMPT_handson.md` with its results | everyone | **open to compare with your run; don't edit** |

Why not run Setup or the demo yourself: they create the session's shared resources under fixed names.
Those names already exist here, so the prompts would stop and ask, and with new names they would start
new GPU serving endpoints for every participant.

## Before you start

1. From any page, open Genie Code in a **new chat** and choose **Agent** mode.
2. Set Effort to **Auto**. It can switch back to Low, so check it again before you paste.
3. You don't need to open a notebook first or pick a GPU. Genie Code creates the notebook and runs it
   on CPU serverless; if it opens on GPU compute, it switches to CPU serverless.
4. Nothing to copy. Open `[Attendees] PROMPT_handson.md` in this folder (don't edit it). H1 creates
   your notebook in your own home folder, `/Users/<user>/teddy_handson/`, and writes only to
   `dbacademy.<you>`, the schema the lab created for you. You never write to `main.owm` or `/Shared`.

## Run the hands-on

1. Open `[Attendees] PROMPT_handson.md` in this folder and copy the grey box under **H1: embed the
   tutorial cells**.
2. Paste it into the Genie Code chat and send it. Follow along in the notebook it opens. It stops
   and asks if your table or folder already exists, or after 3 failed attempts at the same error.
3. When H1 is done (your table and a UMAP), paste the **H2** box into the same chat.
4. Optional, when H2 is done: paste the **Optional: report** box for an HTML write-up of your run.
5. Compare your notebook with the reference run `runs/teddy_handson` in this shared folder (open it;
   don't edit).

## What it creates (yours only)

| What | Made by | Unity Catalog name | Path |
| --- | --- | --- | --- |
| Notebook | H1 | (workspace, not UC) | `/Users/<user>/teddy_handson/teddy_handson_<you>` (suffix such as `_2` if the name is taken) |
| Schema (already exists) | the lab | `dbacademy.<you>` | Catalog Explorer: `dbacademy` > `<you>` |
| Table of your embeddings (3,538 rows) | H1 | `dbacademy.<you>.teddy_tutorial_cells` | Catalog Explorer: `dbacademy` > `<you>` > Tables |
| Volume for figures and report | H1 | `dbacademy.<you>.teddy_handson` | `/Volumes/dbacademy/<you>/teddy_handson/` |
| UMAP figure | H1 | (file in the volume) | `/Volumes/dbacademy/<you>/teddy_handson/umap_cell_types.png` |
| Confusion matrix | H2 | (file in the volume) | `/Volumes/dbacademy/<you>/teddy_handson/confusion_matrix.png` |
| Report (optional) | report box | (file in the volume) | `/Volumes/dbacademy/<you>/teddy_handson/handson_report.html` |

`<user>` is your full user name (your email, as in your home folder `/Users/<user>`); `<you>` is the
part before the @, with dots and dashes as underscores. Example: user
`jane.doe@example.com` gets table `dbacademy.jane_doe.teddy_tutorial_cells` and volume
`/Volumes/dbacademy/jane_doe/teddy_handson/`. Figure file names come from Genie
Code and can differ slightly in your run; the table and volume names are fixed by the prompt. Only
you can write to `dbacademy.<you>`.

## What it uses (shared, read only)

| What | Name |
| --- | --- |
| Serving endpoint (TEDDY-70M) | `teddy_ep_shared` (model `main.owm.teddy_model_shared`) |
| AI Search index (1,125 Census brain cells, 9 types) | `main.owm.teddy_ref_index`, on `teddy_vs_workshop` |
| TEDDY files and tutorial sample data | `/Volumes/main/owm/teddy_files/shared/teddy_70m/` |
| Tutorial check from Setup S5 | `/Volumes/main/owm/teddy_files/shared/tutorial_validation.json` |
| Genie Code skills | `open-weight-models`, `html-report` (installed for the workspace) |

The serving endpoint scales to zero when idle, so the first request after a quiet spell can take a
few minutes while it wakes up. That is expected.

## If something goes wrong

* "Endpoint, index or shared files not found": tell the facilitator; don't create your own.
* "Your table or folder already exists" (for example a second attempt): Genie Code stops and asks. Say
  whether to stop or use a new name.
* "Spark and SDK users differ" or "`dbacademy.<you>` not found": sign out, sign in as your lab user
  only, start a new Genie Code chat and paste H1 again.
* A permissions error on the endpoint, index, table or volume: tell the facilitator (it is a grant on
  the shared resources, not something to fix in your notebook).

## Demo (presenter, about 20 min)

Genie Code deploys TEDDY-70M and an AI Search index. In plain English: TEDDY-70M reads a single
cell's gene activity and turns it into a 512-number fingerprint. We have 1,125 brain cells whose types
are already known. To label a new, unseen cell, the model fingerprints it, then an AI Search index
finds its 10 closest known cells — majority vote gives the type.

| Step | What the audience sees | About |
| --- | --- | --- |
| Before: Setup | nothing (`PROMPT_setup.md` S0-S3; S4-S5 are for the hands-on) | about 45 min |
| 1. Deploy | TEDDY-70M (Setup's checked Hugging Face copy, or a fresh download) served with an express deployment | 6-10 min |
| 2. Index | a searchable index of 1,125 reference brain cells | 2-3 min |
| 3. Label | 10 unseen cells labelled by their nearest reference cells | 5 min |
| Optional: Report | an HTML report of the run | a few min |

Paste prompts 1-3 one at a time into one **new** Genie Code chat (**Agent** mode, Effort **Auto**).
This prompt file has one example run notebook, `runs/teddy_demo` next to it: prompt 1 creates it,
prompts 2-3 and the optional report add to it. It shows what Genie Code generates: the skills keep
runs consistent, but yours may differ slightly. Say yes when it asks to start paid compute. When
prompt 1 creates `runs/teddy_demo`, set its compute to Serverless GPU → A10 before saying yes to
paid compute (Genie Code can't attach compute; a new notebook opens on CPU serverless). For another
workspace, change only the Values line in prompt 1 (catalog, schema, volume); every other name is
relative to it.

The full prompts are in `setup_and_demo/PROMPT_demo.md`; the steps below summarize what each does.

* **Prompt 1 (deploy):** Genie Code packages an open-weight model from Hugging Face and serves it.
  Express deployment packages the environment at registration, so the endpoint is ready in minutes,
  not 15-25. Reuses Setup's TEDDY-70M files read-only if their revision and checksums match (else
  downloads them from Hugging Face), registers model `teddy_model_demo` with an express deployment
  (`env_pack`), and starts serving endpoint `teddy_ep_demo` with AI Gateway usage tracking and an
  inference table on.
* **Prompt 2 (index):** Endpoints are long-lived and shared; each team adds its own index in about a
  minute and deletes it when done. Only the first index on a new endpoint is slow (about 20 min),
  which is why we created this one earlier. Copies table `teddy_ref_cells` to `teddy_demo_cells` with
  Change Data Feed on, builds Delta Sync index `teddy_demo_index` on AI Search endpoint
  `teddy_vs_workshop`.
* **Prompt 3 (label):** A new cell goes through the model, then the index, and comes back with its
  most similar labelled cells in under a second. The same code works for millions of cells. Sends 10
  cells from `shared/query_cells.h5ad` through the endpoint and looks each up in the index (10
  neighbours). Shows a results table, a labelling figure, and a UMAP of reference cells with query
  cells overlaid. Also runs an interneuron leave-one-out on reference cells (not an unseen-cell test).
* **Optional (report):** Using the html-report skill, writes a short HTML report of the run and saves
  it to `demo/demo_report.html`.
* **Teardown (after the demo):** Deletes only what the demo made; keeps everything from Setup. Deletes
  in order: endpoint, index, tables, model versions then model, folder.

### What the demo creates (presenter only)

| What | Name |
| --- | --- |
| Serving endpoint | `teddy_ep_demo` |
| Model | `main.owm.teddy_model_demo` |
| Table | `main.owm.teddy_demo_cells` |
| AI Search index | `main.owm.teddy_demo_index` (on `teddy_vs_workshop`) |
| Inference table | `main.owm.teddy_ep_demo_payload` (AI Gateway; rows can take up to an hour) |
| TEDDY files (only if Setup's copy fails its check) | `/Volumes/main/owm/teddy_files/demo/teddy_70m/` |
| Report (optional) | `/Volumes/main/owm/teddy_files/demo/demo_report.html` |
| Notebook | `setup_and_demo/runs/teddy_demo` |

### What the demo uses (shared, from Setup)

| What | Name |
| --- | --- |
| AI Search endpoint | `teddy_vs_workshop` |
| Reference table | `main.owm.teddy_ref_cells` |
| Query cells | `/Volumes/main/owm/teddy_files/shared/query_cells.h5ad` |
| TEDDY files (read only) | `/Volumes/main/owm/teddy_files/shared/teddy_70m/` |

### Demo results (2026-11-09 run)

`shared/query_cells.h5ad` covers 5 of the 9 reference types (no GABAergic interneurons); the report
says so. 10 of 10 correct, one cell 8 of 10 (2 pvalb neighbours); interneuron leave-one-out on
reference cells 20 of 20.

## For facilitators

* Order: `setup_and_demo/PROMPT_setup.md` S0-S5 before the session (S0-S3 are enough for the demo; S4-S5 are for the
  hands-on), then the demo, then the hands-on.
* For another workspace, change only the Values line (catalog, schema, volume) in the first prompt of
  each file; every other name is relative to it.
* Setup and demo notebooks go to `setup_and_demo/runs/` (next to their prompts); the hands-on
  reference run is `runs/teddy_handson`.
* After the demo, run the Teardown box in `setup_and_demo/PROMPT_demo.md`; it keeps everything from
  Setup.
* `[Attendees] PROMPT_handson.md` has no teardown box and writes nothing to `main.owm` or `/Shared`:
  each participant's table and volume are in their own lab schema `dbacademy.<you>`, and their notebook
  is in `/Users/<user>/teddy_handson/`. Attendee read grants are set and checked in
  `setup_and_demo/Attendee Access Grants and Checks`. Setup resources (`teddy_ep_shared`,
  `teddy_vs_workshop`, `teddy_ref_index`) bill until deleted.

## After the session

You don't need to delete anything; the facilitator cleans up the shared resources. Your notebook,
table and volume stay in your own home folder and lab schema.
