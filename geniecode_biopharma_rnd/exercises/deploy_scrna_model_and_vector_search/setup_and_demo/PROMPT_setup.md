# Setup: shared resources for the demo and hands-on (facilitator, once, about 70 min)

**In plain English:** You do this once before the session to prepare everything the demo and hands-on will use: get the TEDDY-70M model files (download them, or reuse an earlier download after checking nothing changed), pull 1,125 known brain cells from Census, embed them, build an AI Search index over them, and start a shared serving endpoint that saves every request to a table. After this, the demo and every participant just call what you already made — nothing to download or deploy at session time.

| Step | What it makes | Why before the session | About |
| --- | --- | --- | --- |
| S0. Clean | removes earlier runs' resources from the workshop schema (keeps a TEDDY-70M download at the pinned revision) | v3 starts from an empty schema, so every result ties to one run | 5 min |
| S1. Files | pinned TEDDY-70M files and tutorial data (reused if every checksum matches) | one download, reused by everything | 5 min |
| S2. Cells | 9 types x 125 Census reference cells, 10 demo query cells | Census queries are slow and can fail | 10 min |
| S3. Index | reference table, AI Search (formerly Vector Search) endpoint, base index | the first index on a new endpoint takes about 20 min | 25-30 min |
| S4. Serve | shared serving endpoint for the hands-on | participants deploy nothing | 10 min |
| S5. Tutorial check (optional, recommended) | the endpoint vs TEDDY's own tutorial | needs a GPU; the hands-on just shows the result | 10 min |

Paste S0-S5 one at a time into one **new** Genie Code chat (**Agent** mode, Effort **Auto**). This
prompt file has one example run notebook next to it,
[`runs/teddy_setup`]($./runs/teddy_setup) (a full S0-S5 run, all steps passed).
Your own run goes in `runs/teddy_setup`: S0 creates it, S1-S5 add to it. The example shows what
Genie Code generates: the skills keep runs consistent, but yours may differ slightly.
For another workspace, change only the Values line in S0 (catalog, schema, volume); every other
name is relative to it.

Or paste the S0-S5 boxes together as one message. Genie Code does them in order in the same
notebook, still asks before deleting and before paid compute, and stops at the first failed check.

## S0: clean the workshop schema

```text
Using the open-weight-models skill (follow the "Short runs" list in its teddy.md), create a NEW
notebook `teddy_setup` (don't reuse or edit an existing one) in a `runs/` folder next to this prompt
file (create `runs/` if it doesn't exist; build the full path from this file's folder and check the
real path after creating it) on serverless GPU (A10, AI environment v3+).
Values: catalog `main`, schema `owm`, volume `teddy_files`; the tables, models and indexes below are
in this schema and the folders in this volume. Keep cells few and short, print summaries only; read
the user and paths at run time, nothing workspace-specific hard-coded. Rules for this chat: ask me
once before deleting (here) and once before starting paid compute (AI Search endpoint, serving
endpoint); overwrite nothing; stop and ask if a name is taken; stop after 3 failed attempts at the
same error; finish and check each step before starting the next, and stop at the first failed check
and tell me; note each step's UTC start and end. Use only this prompt and the skills as sources:
don't use other notebooks, queries or past chats.
S0: list, without deleting, every table, view, AI Search index, model (with its versions) and
volume folder in the schema, every serving endpoint that serves a model from the schema, and every
AI Search endpoint that holds an index from the schema, as one table (type, name, created,
owner). After I confirm, delete them all (endpoints first, then indexes, then tables, models and
folders), keeping the schema and the volume; delete an AI Search endpoint only if it then holds
no indexes. List model versions with the Unity Catalog REST endpoint
`/api/2.1/unity-catalog/models/{full_name}/versions` and delete them before their model. Keep
folder `shared/teddy_70m/` (list it as kept) if it has `.download_complete` and a `manifest.json`
with revision 8c880aa67674042ccdc254bccb5de081092a48bf; S1 checks it. Otherwise delete it with the
rest. Make the delete cell refuse to run if `shared/teddy_70m/.download_complete` is newer than this
step's start, so a re-run after S1 can't wipe it.
List again and check only the schema, `teddy_files` and (if kept) `shared/teddy_70m/` are left.
```

## S1: files

```text
In the same notebook: create folder `shared/` if it's missing. Get the TEDDY-70M files at
revision 8c880aa67674042ccdc254bccb5de081092a48bf, plus `data/sample_data.h5ad` and
`tutorials/model_tutorial.ipynb`, in `shared/teddy_70m/`: if S0 kept that folder, check every file
in its manifest against its checksum (sha256 or git-sha1); if all match and those files are listed,
reuse it and say so; if any check fails, stop and ask me before replacing it. If there is no folder,
download them there and record source, revision, licence. Check 20
tutorial sample cells two ways and compare per cell (cosine >= 0.999): (a) the tutorial's own
preprocess, tokenize, model-load and pooling cells, unchanged except paths; (b) the embedding code
this notebook will use in S3-S4 (raw counts in, TEDDY's package functions, tutorial pooling). Don't
compare a run with a repeat of itself. Write h5ad files under `/tmp/teddy_s1/`, not the volume.
```

## S2: reference and query cells

```text
In the same notebook: from Census 2025-11-08 (human brain, primary data, 10x 3' v3, TEDDY's QC;
leave out every dataset in CELLxGENE collection b953c942-f5d8-434f-9da7-e726ba7c1481, the
tutorial's study), sample 125 cells of each: oligodendrocyte; astrocyte; oligodendrocyte precursor
cell; microglial cell; L2/3-6 intratelencephalic projecting glutamatergic neuron; VIP, pvalb, sst
and lamp5 GABAergic cortical interneuron. Also 2 more each of the first 5 types as demo query
cells. Save both as h5ad (raw counts, cell_id, cell_type, dataset_id) with provenance:
`shared/ref_cells.h5ad`, `shared/query_cells.h5ad`. Write each h5ad under `/tmp/teddy_s2/` first,
then copy the finished file to the volume.
```

## S3: reference table and index

```text
In the same notebook (GPU): embed the 1,125 reference cells (pooling="mean", float32) into table
`teddy_ref_cells` (cell_id, embedding, cell_type, model_version) with Change Data Feed on. Create
AI Search endpoint `teddy_vs_workshop`, wait until ONLINE, build Delta Sync index
`teddy_ref_index` with the skill's scripts/vs_index.py, check 20 cells find themselves first, and
report how long the index took from create to answering queries. Grant CAN_USE on
the AI Search endpoint to all workspace users (group `users`).
```

## S4: shared serving endpoint

```text
In the same notebook: register the files in `shared/teddy_70m/` as `teddy_model_shared` with an
express deployment (`env_pack`) and serve it as `teddy_ep_shared` with AI Gateway usage tracking and inference tables on
(in this schema, table prefix `teddy_ep_shared`), set at create. When READY: send 20 reference
cells through it and check they match the table's embeddings (cosine >= 0.999), check a non-default
setting changes the output, and report create-to-READY time. Grant CAN_QUERY on the serving endpoint
to all workspace users (group `users`).
```

## S5: tutorial check (optional, recommended)

```text
In the same notebook (GPU): run the tutorial's steps on `shared/teddy_70m/` sample data (3,538
cells, `.raw.X` counts), then embed the same cells through `teddy_ep_shared` with tutorial pooling
and compare per cell (cosine >= 0.999). One figure: per-cell cosine against the 0.999 line. Save the
result to `shared/tutorial_validation.json` (endpoint, revision, cells, min and median cosine,
PASS/FAIL, UTC time). End with a summary: names created, each step's time, and what is still
billing.
```
