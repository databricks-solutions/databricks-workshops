# Bean & Brew Coffee Co.: Databricks SQL workshop in a box

A Declarative Automation Bundle for a Databricks SQL workshop built around a fictional 12-store coffee chain. Deploying
sets up the infrastructure and the raw data; a notebook then walks through building everything else. One destroy
removes all of it.

| Set up by the deploy | What it is |
|---|---|
| SQL warehouse | Serverless, 2X-Small, one cluster, shortest auto-stop |
| `dbsql_workshop_<you>_bronze` / `_silver` / `_gold` schemas | Your three schemas. Bronze holds the bundle's settings for the notebook as schema properties |
| Raw data load | 6 bronze tables with deliberate defects (about 1.5M register tickets, 2025-01-01 to 2026-10-19), loaded in SQL on the bundle's warehouse during the deploy |

| Built by `Bean & Brew DBSQL Workshop.ipynb` | What it is |
|---|---|
| Silver and gold | 7 cleaned silver tables and 5 gold tables, plus the promo target list |
| Semantic layer and governance | 3 metric views, primary and foreign keys, column comments and governed tags |
| Validation | Checks every demo scenario and a fixed data fingerprint |
| SQL editor script and saved queries 01-09 | One modern SQL feature each (metric views, QUALIFY, `ai_query()`, …) |
| Visual data prep flow | No-code loyalty promo target list |
| AI/BI dashboard | 4-page store operations dashboard, published, with a daily refresh schedule (paused) |
| Genie agent | Instructions, worked examples, joins, starter questions and benchmarks |
| 3 SQL alerts | Inventory below reorder point, store traffic drop, labor cost above target (weekly, paused) |

The data is deterministic (every random value is a hash of the row key), so every deployment produces the same rows
and the same numbers.

## Deploy with Open in Databricks

- Git repository: `https://github.com/databricks-solutions/databricks-workshops`
- Subdirectory: `dbsql_workshop`

Open in Databricks clones the repository into your workspace, opens this bundle and deploys the default `dev` target.
No input is needed.

## Deploy from a terminal

Requires Databricks CLI 1.7.0 or later.

```bash
cd dbsql_workshop
databricks bundle deploy --profile <PROFILE>               # default target: dev
databricks bundle deploy -t fevm                           # the workshop team's workspace
databricks bundle deploy --var catalog=<catalog> --profile <PROFILE>
```

Re-running `deploy` is safe. The raw tables are created with `IF NOT EXISTS`, and the load re-runs only when its
notebook changes.

## Run the workshop

Open `Bean & Brew DBSQL Workshop.ipynb` (in the cloned Git folder, or under the bundle's `files/` folder after a
terminal deploy), attach serverless compute and run the cells in order. The Connect cell finds your schemas and reads
the bundle's settings, so there is nothing to fill in. Re-running any cell is safe: tables are created only if
missing and the workspace assets are updated in place.

### Variables

| Variable | Default | Purpose |
|---|---|---|
| `catalog` | `workspace` (target `fevm`: `for_startups_demos_catalog`) | Existing catalog for the three schemas. The catalog itself is never changed |
| `warehouse_id` | the warehouse this bundle creates | Warehouse for the raw data load and the notebook's dashboard, Genie agent, alerts and saved queries |
| `llm_endpoint` | `databricks-meta-llama-3-3-70b-instruct` | Pay-per-token endpoint for `ai_query()` in saved query 08 |
| `alert_email` | the deploying user | Alert subscriber |

## Tear down

- **In the workspace:** open `databricks.yml`, click the **Deployments** icon, then **Destroy**.
- **From a terminal:** `databricks bundle destroy -t <target>`

This removes the warehouse, the schemas with every table and view in them, the raw data job, and the bundle's folder,
which holds every workspace asset the notebook created (saved queries, data prep flow, dashboard, Genie agent and
alerts). The catalog stays, and so do the bundle's source files (for example the Git folder that Open in Databricks
cloned).

## Names and safety

Every name carries `dbsql_workshop` and the deploying user, so several people can deploy into one workspace:

- schemas `dbsql_workshop_<user>_bronze|silver|gold`
- workspace objects `[dbsql_workshop <user>] …`

Nothing that already exists is reused or overwritten. If a schema or warehouse with one of these names already
exists, the deploy fails instead, and the notebook only ever updates assets it created itself. Deploy one target per
workspace, because the `dev` and `fevm` targets would share the same schema names.

## Requirements

- Unity Catalog, with `USE CATALOG` and `CREATE SCHEMA` on the target catalog
- Serverless compute for notebooks and SQL warehouses, and permission to create a SQL warehouse
- A pay-per-token Foundation Model endpoint for `ai_query()` (only saved query 08 uses it)

## Layout

```
databricks.yml                         bundle, variables, targets (dev, fevm)
Bean & Brew DBSQL Workshop.ipynb       the workshop notebook: silver, gold, semantic layer, validation, assets
resources/                             warehouse, schemas (with the notebook's settings), raw data job + its
                                       deploy-time run
src/load_raw_data.ipynb                the bronze load (SQL, runs on the bundle's warehouse during deploy)
src/bean_and_brew.py                   notebook helpers: connect, and create the workspace assets
src/assets/                            saved query SQL, the data prep flow, dashboard, Genie agent and alert
                                       definitions, and their manifest
```
