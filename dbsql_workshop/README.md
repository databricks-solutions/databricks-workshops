# Bean & Brew Coffee Co.: Databricks SQL workshop in a box

A Declarative Automation Bundle that deploys a complete Databricks SQL demo for a fictional 12-store coffee chain,
then removes it again with one command. Deploying builds everything; destroying removes everything.

| Built on deploy | What it is |
|---|---|
| SQL warehouse | Serverless, 2X-Small, one cluster, shortest auto-stop |
| `dbsql_workshop_<you>_bronze` / `_silver` / `_gold` schemas | 18 generated tables (about 1.5M register tickets, 2025-01-01 to 2026-10-19) with keys, comments and tags, plus 3 metric views |
| Build job | Generates the data, checks every demo scenario, creates the saved queries and the flow. It runs once during the deploy |
| AI/BI dashboard | 4-page store operations dashboard, published, plus a daily refresh job (paused) |
| Genie agent | Instructions, worked examples, joins, starter questions and benchmarks |
| 3 SQL alerts | Inventory below reorder point, store traffic drop, labor cost above target (weekly) |
| SQL editor script and saved queries 01-09 | One modern SQL feature each (metric views, QUALIFY, `ai_query()`, …) |
| Visual data prep flow | No-code loyalty promo target list |

The data is deterministic (every random value is a hash of the row key), so every deployment produces the same rows
and the same numbers. The build's `validate` task checks this with a fixed data fingerprint.

## Deploy with Open in Databricks

- Git repository: `https://github.com/databricks-solutions/databricks-workshops`
- Subdirectory: `dbsql_workshop`

Open in Databricks clones the repository into your workspace, opens this bundle and deploys the default `dev` target.
No input is needed. The deployment takes a few minutes because it waits for the build job.

## Deploy from a terminal

Requires Databricks CLI 1.7.0 or later.

```bash
cd dbsql_workshop
databricks bundle deploy --profile <PROFILE>               # default target: dev
databricks bundle deploy -t fevm                           # the workshop team's workspace
databricks bundle deploy --var catalog=<catalog> --profile <PROFILE>
```

Re-running `deploy` is safe. Tables are created with `IF NOT EXISTS`, the build re-runs only when its code changes,
and the saved queries are updated in place.

### Variables

| Variable | Default | Purpose |
|---|---|---|
| `catalog` | `workspace` (target `fevm`: `for_startups_demos_catalog`) | Existing catalog for the three schemas. The catalog itself is never changed |
| `warehouse_id` | the warehouse this bundle creates | Warehouse for the dashboard, Genie agent, alerts and saved queries |
| `llm_endpoint` | `databricks-meta-llama-3-3-70b-instruct` | Pay-per-token endpoint for `ai_query()` in saved query 08 |
| `alert_email` | the deploying user | Alert subscriber |

## Tear down

- **In the workspace:** open `databricks.yml`, click the **Deployments** icon, then **Destroy**.
- **From a terminal:** `databricks bundle destroy -t <target>`

This removes the warehouse, the schemas with every table and view in them, the dashboard, the Genie agent, the
alerts, both jobs, and the bundle's folder holding the saved queries and the data prep flow. The catalog stays, and
so do the bundle's source files (for example the Git folder that Open in Databricks cloned).

## Names and safety

Every name carries `dbsql_workshop` and the deploying user, so several people can deploy into one workspace:

- schemas `dbsql_workshop_<user>_bronze|silver|gold`
- workspace objects `[dbsql_workshop <user>] …`

Nothing that already exists is reused or overwritten. If a schema or warehouse with one of these names already
exists, the deploy fails instead. Deploy one target per workspace, because the `dev` and `fevm` targets would share
the same schema names.

## Requirements

- Unity Catalog, with `USE CATALOG` and `CREATE SCHEMA` on the target catalog
- Serverless compute for jobs and SQL warehouses, and permission to create a SQL warehouse
- A pay-per-token Foundation Model endpoint for `ai_query()` (only saved query 08 uses it)

## Layout

```
databricks.yml                         bundle, variables, targets (dev, fevm)
resources/                             warehouse, schemas, build job + its deploy-time run, dashboard
                                       + refresh job, Genie agent, alerts
src/build/                             build job notebooks (bronze, silver, gold, semantic layer, validate,
                                       saved assets) and a shared helper module
src/assets/                            saved query SQL, the visual data prep flow and their manifest
```
