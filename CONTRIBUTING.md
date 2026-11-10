# Contributing a workshop

Every workshop in this repository has to work for someone who has never seen it: one click on **Open in Databricks**
in a brand-new Databricks Free Edition workspace deploys it, a notebook walks through it, and **Destroy** removes it.
Any number of different workshops must be able to sit in the same workspace at the same time without touching each
other.

The rules below make that true. `scripts/check_workshops.py` checks them on every pull request; run it yourself first:

```bash
pip install pyyaml
python scripts/check_workshops.py
```

## The rules

### 1. Layout

- One top-level folder per workshop, named in `snake_case`. If the workshop deploys anything, its bundle
  (`databricks.yml`) sits at the root of that folder, and `bundle.name` equals the folder name.
- A `README.md` with: what the deploy creates, how to run the workshop, a `## Tear down` section, and the support
  disclaimer under `## How to get help` (copy it from an existing workshop).
- Notebooks in **source format**: `.py` or `.sql` files that start with `# Databricks notebook source` (or
  `-- Databricks notebook source`). No `.ipynb`: they're JSON, unreadable in review and in diffs.
- No file over 5 MB. Open in Databricks clones the whole repository with a 3-minute limit, so generate data in SQL
  instead of committing it.
- Presenter-only material goes in a `presenter/` folder listed in the workshop's `.gitignore`. It is never committed
  and never deployed.

### 2. One namespace per workshop

The workshop's name is its namespace. Everything it creates carries it, so two workshops can never collide:

- No `mode:` on any target. Instead, set `presets.name_prefix: "[<workshop>] "`. Bundle resources are named
  `[<workshop>] ...` and schemas `<workshop>_<name>`.
- Code only writes inside the workshop's own schemas (`<workshop>_*`) and never refers to another workshop's objects.
- The names are fixed, so a workspace holds one copy of a workshop. A second deploy fails on the existing schemas
  before it creates anything: make any deploy-time job wait for the schemas (`lifecycle.depends_on`), so it can't be
  created first.

### 3. One click, no input

Open in Databricks deploys the default target without asking anything:

- Exactly one target has `default: true`, and it sets no `workspace.host`, so it deploys wherever it's opened.
  Other targets (an internal team workspace, say) may set a host.
- Every variable has a default. The one allowed lookup is the SQL warehouse every workspace has:
  `lookup: {warehouse: Serverless Starter Warehouse}`.
- A `catalog` variable defaults to `workspace`, the Free Edition default catalog. The README explains how to point it
  at another catalog (copy "Use your own catalog" from `dbsql_workshop`).
- No Python bundle configuration (`python:` in `databricks.yml`): workspace deploys can't run it. Python in
  notebooks and job tasks is fine.

### 4. A full destroy path

**Destroy** must remove everything the workshop created, and the README must list anything it can't:

- Create schemas only as bundle `schemas` resources in `${var.catalog}`. Destroy drops them with every table and view
  inside. Never `CREATE CATALOG`, and never `CREATE SCHEMA` in code.
- Create workspace objects (jobs, dashboards, Genie agents, alerts, pipelines) as bundle resources. Destroy removes
  them.
- Saved queries and visual data prep flows have no bundle resource type. Prefer notebook cells for queries. For
  anything else, a short "build it in the UI" step is the last resort: it must take a few clicks, and the README's
  `## Tear down` section must say to delete it by hand.
- Don't create workspace objects through the REST API or SDK from a notebook: destroy doesn't know about them.

### 5. Minimal DBUs

- Run SQL on the `Serverless Starter Warehouse` lookup. Free Edition allows a single warehouse, so don't create one.
  If a workshop truly needs its own: serverless, `2X-Small`, one cluster, `auto_stop_mins: 1`.
- Serverless compute only: no `clusters`, `job_clusters`, `new_cluster` or `existing_cluster_id`. Serverless jobs
  use `performance_target: STANDARD` unless a person waits on them during deploy (then prefer a SQL warehouse task).
- Pipelines set `serverless: true`. Schedules start paused (`presets.trigger_pause_status: PAUSED`, and
  `pause_status: PAUSED` on alert schedules).
- Models: only those in `ALLOWED_MODELS` in `scripts/check_workshops.py`, the cheapest that does the job and that
  isn't legacy or deprecated. Call `ai_query()` on a few rows: pick the rows first (`LIMIT` in a CTE), then call it.
- Anything that bills while idle (model serving or vector search endpoints, apps, Lakebase) must be stated in the
  README.

### 6. The base case is Free Edition

Nothing a workshop needs may go beyond what a Free Edition or trial workspace gives the person who signed up: no
workspace admin rights, no `CREATE CATALOG`, no classic compute, no account console. Free Edition limits: one SQL
warehouse (2X-Small), 5 concurrent job tasks per account, a daily compute quota and a reduced set of models. Put any
extra requirement in the workshop's README and in [USER_GUIDE.md](USER_GUIDE.md).

## Where things go

| What | Where | Why |
|---|---|---|
| Schemas | Bundle `schemas` resources | Destroy drops them with everything inside |
| Tables and views the bundle's assets read | SQL in a deploy-time job (`job_runs`) on the warehouse | Genie agents need their tables to exist: create them empty, with keys and comments |
| The data participants build | SQL cells in the workshop notebook (`INSERT OVERWRITE ... BY NAME` into the deployed tables) | Readable, re-runnable, and it keeps the tables the assets point at |
| Dashboards, Genie agents, alerts, jobs, pipelines | Bundle resources | Destroy removes them, and the bundle fills in catalog and schema names |
| Queries to show in the SQL editor | Notebook cells | Nothing to clean up |
| Anything else with no bundle resource type | A short UI step in the notebook (last resort) | Listed under `## Tear down` |

## Starting a new workshop

Copy `dbsql_workshop` as a model, or start from this `databricks.yml`:

```yaml
bundle:
  name: my_workshop                     # = the folder name
  databricks_cli_version: ">= 1.7.0"

include:
  - resources/*.yml

variables:
  catalog:
    description: Existing catalog for the workshop's schemas. See "Use your own catalog" in README.md.
    default: workspace
  warehouse_id:
    description: SQL warehouse for setup and the workshop's assets.
    lookup:
      warehouse: Serverless Starter Warehouse

presets:
  name_prefix: "[my_workshop] "         # schemas become my_workshop_<name>
  trigger_pause_status: PAUSED
  tags:
    workshop: my_workshop

targets:
  dev:
    default: true                        # no host, no mode
```

## Test before you open a pull request

The check script covers the files. These need a real workspace, ideally a fresh Free Edition one, as a user who is
not an admin of anything you didn't create:

1. `databricks bundle validate`, then deploy (Open in Databricks, or `databricks bundle deploy`).
2. Run the notebook top to bottom.
3. Deploy again: it should report `0 created, 0 changed, 0 deleted`.
4. Deploy a second copy into the same workspace (a scratch target with the same host): it should fail on the existing
   schemas and create nothing.
5. Deploy another workshop from this repository alongside: both work.
6. Destroy, then check that nothing named after the workshop is left (search the workspace for `[<workshop>]`, and
   look for `<workshop>_*` schemas).

## Publishing an Open in Databricks link

Create the link after the workshop is merged to `main`: links always clone the default branch. In the admin tool
(go/openindatabricks/create), enter exactly `https://github.com/databricks-solutions/databricks-workshops` and the
workshop's folder as the subdirectory. Updates merged to `main` reach every published button immediately, so a
breaking commit breaks them all: test on a branch first.
