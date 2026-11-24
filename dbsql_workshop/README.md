# Bean & Brew Coffee Co.: Databricks SQL workshop in a box

[![Open in Databricks](https://www.databricks.com/open-in-databricks.png)](https://login.databricks.com/signin?destination_url=%2Fopen%3Fp%3DeyJhY3Rpb24iOiJpbXBvcnRkYWIiLCJwYXlsb2FkIjp7ImdpdFVybCI6Imh0dHBzOi8vZ2l0aHViLmNvbS9kYXRhYnJpY2tzLXNvbHV0aW9ucy9kYXRhYnJpY2tzLXdvcmtzaG9wcyIsInN1YmRpciI6ImRic3FsX3dvcmtzaG9wIn19)

A Databricks SQL workshop built around a fictional 12-store coffee chain. The deploy loads a governed silver layer and
sets up the analytics assets on an empty gold layer. Participants tour the SQL warehouse and the catalog, build gold in
SQL (including a model built with AI functions on free-text customer comments), then watch the dashboard, the Genie
agent and the alerts come to life. One destroy removes everything the deploy created.

| Set up by the deploy | What it is |
|---|---|
| `dbsql_workshop_silver`, `dbsql_workshop_gold` schemas | In the catalog `workspace` unless you choose another ([Use your own catalog](#use-your-own-catalog)) |
| Silver tables, loaded | 8 tables: about 1.5M register tickets (2025-01-01 to 2026-10-19), menu, stores, loyalty members, inventory, labor, and free-text customer feedback. Column comments, primary and foreign keys, tags |
| Empty gold tables | 6 gold tables with their comments and keys, plus 3 metric views on them |
| AI/BI dashboard | 4-page store operations dashboard, published |
| Genie agent | Instructions, worked examples, joins, starter questions and benchmarks |
| 3 SQL alerts | Inventory below reorder point, store traffic drop, labor cost above target. Weekly, created paused |
| Setup job | `[dbsql_workshop] Bean & Brew - Set up demo data`, which creates the tables, loads silver and adds comments and tags during the deploy |

| Built in `Bean & Brew DBSQL Workshop.sql` | What participants do |
|---|---|
| Steps 1-2 | Tour the SQL warehouse (size, scaling, auto-stop, connection details) and the governed silver layer in Catalog |
| Step 3 | Build the gold models in SQL (`INSERT OVERWRITE ... BY NAME`), one of them with `ai_extract()`, `ai_classify()` and `ai_analyze_sentiment()` |
| Step 4 | Test gold: every demo scenario, plus a fixed data fingerprint |
| Step 5 | Query gold: one feature per query, including `ai_query()` and `ai_summarize()`; query profile in the SQL editor |
| Steps 6-9 | Metric views, the dashboard, the Genie agent and the alerts, now full of data |
| Step 10 | Tear down |

The notebook's text cells double as the demo guide: each step outside the notebook lists the clicks to get there.

The deploy uses the workspace's **Serverless Starter Warehouse**; it doesn't create or change a warehouse. The data is
deterministic (every random value is a hash of the row key), so every deployment produces the same rows and numbers.
Steps 3 and 5 call AI Functions (`ai_extract`, `ai_classify`, `ai_analyze_sentiment`, `ai_summarize`, `ai_query`) on
about 100 rows; they need Foundation Model APIs in your workspace's region. Without them, skip those cells: everything
else works.

> **Before you deploy into an existing workspace:** the deploy needs a catalog named `workspace` (the Free Edition
> default) and a SQL warehouse named `Serverless Starter Warehouse`. If either is missing, the deploy fails with
> `Catalog 'workspace' does not exist` or a warehouse lookup error. Each is a one-line change:
> [Use your own catalog](#use-your-own-catalog) · [Use another SQL warehouse](#use-another-sql-warehouse).
> The full list of permissions is in [USER_GUIDE.md](../USER_GUIDE.md).

## Run the workshop

1. Deploy: click the Open in Databricks button above, or see [Deploy from a terminal](#deploy-from-a-terminal).
2. Open `Bean & Brew DBSQL Workshop` in the `dbsql_workshop` folder (in the Git folder that Open in Databricks
   created, or under the bundle's `files/` folder after a terminal deploy).
3. Attach the bundle's SQL warehouse (the **Serverless Starter Warehouse** unless you
   [chose another](#use-another-sql-warehouse)) or serverless compute, and follow the steps in order. Every cell is
   SQL and safe to re-run. The cells take about 3 minutes in all, one of them the AI model.
4. `DEMO_GUIDE.md`, next to the notebook, has a one-page overview of each step.

## Use your own catalog

The deploy needs an existing Unity Catalog catalog in which you have `USE CATALOG` and `CREATE SCHEMA`. It uses
`workspace` unless you change it.

1. **Choose a catalog.** To find your workspace's default catalog, run `SELECT current_catalog()` in the SQL editor.
   To make a new one, open **Catalog**, click **+**, then **Create a catalog**. Creating a catalog needs the
   `CREATE CATALOG` privilege, and some workspaces also ask for a storage location.
2. **Point the bundle at it.** In the Git folder that Open in Databricks created
   (`/Users/<you>/databricks-solutions-databricks-workshops`), open `dbsql_workshop/databricks.yml` and replace
   `workspace` in the `catalog` variable's `default`:

   ```yaml
   variables:
     catalog:
       default: <your_catalog>
   ```

3. **Deploy again.** Click the **Deployments** icon, then **Deploy**.

From a terminal, pass the catalog instead of editing the file: `databricks bundle deploy --var catalog=<your_catalog>`.

The workshop notebook finds the schemas in whichever catalog you chose. Keep the edit as a local change in your Git
folder rather than committing it; if you later pull updates, Git may ask you to keep your version of that line.

## Use another SQL warehouse

The deploy finds its SQL warehouse by name, `Serverless Starter Warehouse`, and only runs queries on it: it never
creates, resizes or deletes a warehouse. If your workspace renamed or removed that warehouse, the deploy stops with
`failed to resolve warehouse: Serverless Starter Warehouse` before it creates anything.

1. **Choose a warehouse.** Open **Compute → SQL warehouses** and pick one you have **Can use** on (its
   **Permissions** tab shows this). A serverless one is best: the setup and the notebook start in seconds instead of
   minutes, and the smallest size is plenty. If you can't use any warehouse, ask a workspace admin for **Can use** on
   one, or to create a 2X-Small serverless warehouse.
2. **Point the bundle at it.** In the Git folder that Open in Databricks created, open `dbsql_workshop/databricks.yml`
   and replace `Serverless Starter Warehouse` with your warehouse's name:

   ```yaml
   variables:
     warehouse_id:
       lookup:
         warehouse: <your warehouse name>
   ```

3. **Deploy again.** Click the **Deployments** icon, then **Deploy**.

From a terminal, pass the warehouse's ID instead of editing the file (it's on the warehouse's **Overview** tab):
`databricks bundle deploy --var warehouse_id=<warehouse_id>`.

When you run the workshop notebook, attach this same warehouse (or serverless compute). As with the catalog, keep the
edit as a local change in your Git folder rather than committing it.

## Deploy from a terminal

Requires Databricks CLI 1.7.0 or later.

```bash
cd dbsql_workshop
databricks bundle deploy --profile <PROFILE>               # default target: dev
databricks bundle deploy --var catalog=<catalog> --profile <PROFILE>
```

The deploy takes about 2-3 minutes, most of it the setup job. Re-running it is safe: tables are created only if they
don't exist (so the gold the notebook built stays), silver is reloaded with the same rows, and the setup job re-runs
only when a setup notebook changes.

### Variables

| Variable | Default | Purpose |
|---|---|---|
| `catalog` | `workspace` (target `fevm`: `for_startups_demos_catalog`) | Existing catalog for the two schemas. The catalog itself is never changed |
| `warehouse_id` | lookup of `Serverless Starter Warehouse` | Runs the setup job, the dashboard, the Genie agent and the alerts |
| `alert_email` | the deploying user | Alert subscriber |

## Tear down

- **In the workspace:** open `databricks.yml`, click the **Deployments** icon, then **Destroy**.
- **From a terminal:** `databricks bundle destroy`

This removes the two schemas with every table and metric view in them, the setup job, the dashboard, the Genie
agent and the alerts. The dashboard and the Genie agent go to the workspace trash, like any deleted asset. The catalog
and the warehouse stay, because the bundle never created them.

Delete these by hand if you made them: any query you saved in the SQL editor, and the Git folder that Open in
Databricks cloned.

## Names and safety

The names are fixed, so a workspace holds **one copy** of this workshop:

- schemas `dbsql_workshop_silver`, `dbsql_workshop_gold`
- everything else `[dbsql_workshop] …`

Nothing that already exists is reused or overwritten. If the schemas already exist (someone else in the workspace
deployed it, or an old copy was never destroyed), the deploy fails with `SCHEMA_ALREADY_EXISTS` before it creates
anything. Only the person who deployed can destroy the copy, from their own bundle folder.

Want several people in one workspace? Deploy once, then share: grant the participants `USE CATALOG`, `USE SCHEMA` and
`SELECT` on the two schemas, **Can run** on the dashboard and Genie agent, and **Can use** on the warehouse.

## Layout

```
databricks.yml                     bundle, variables, presets (fixed names), targets (dev, fevm)
Bean & Brew DBSQL Workshop.sql     the workshop notebook (a SQL notebook, Databricks source format)
DEMO_GUIDE.md                      an overview of each step of the demo
resources/                         schemas, setup job and its deploy-time run, dashboard, Genie agent, alerts
src/setup/01_create_tables.sql     deploy-time: the empty silver and gold tables, keys and metric views
src/setup/02_build_silver.sql      deploy-time: generates the source data in SQL and loads silver
src/setup/03_document.sql          deploy-time: column comments and tags
```

Tested on AWS (fevm-for-startups-demos, a non-default catalog), deploying from the CLI and running the notebook on both
serverless and the Starter warehouse. Not yet tested in Databricks Free Edition.

## How to get help

Databricks support does not cover this content. For questions or bugs, open a GitHub issue. The team will help on a best effort basis.
