# User guide: run these workshops in your workspace

Every workshop here deploys with one click (Open in Databricks) or one command (`databricks bundle deploy`), and
removes itself with **Destroy**. This guide lists what your workspace and your user need for that to work.

**The base case is a Databricks Free Edition or free trial workspace.** Nothing a workshop needs goes beyond what
such a workspace gives the person who signed up for it, so in Free Edition or a trial you can skip straight to the
workshop. In an existing company workspace, check the two tables below, or send them to your workspace admin.

## Workspace requirements

Set by a workspace admin. All are on by default in Free Edition and trial workspaces.

| Requirement | Why | How to check |
|---|---|---|
| Unity Catalog | Workshops create their schemas, tables and views in a catalog | **Catalog** in the sidebar lists catalogs other than `hive_metastore` |
| Serverless compute | Open in Databricks deploys bundles on serverless; workshops run on serverless notebooks and SQL warehouses | **Compute → SQL warehouses** offers a serverless warehouse |
| Git folders, with github.com allowed | Open in Databricks clones this repository into your workspace | **Workspace → Create → Git folder** accepts `https://github.com/databricks-solutions/databricks-workshops` |
| Databricks SQL, AI/BI dashboards, Genie, SQL alerts | Workshops that deploy these assets | **SQL editor**, **Dashboards**, **Genie** and **Alerts** appear in the sidebar |
| Foundation Model APIs (pay-per-token) in your region | Workshops that call `ai_query()` | **Serving** lists endpoints named `databricks-...` |

## Your permissions

| Permission | On | Needed for | In Free Edition and trials |
|---|---|---|---|
| **Workspace access** and **Databricks SQL access** entitlements | Your user | Everything, including warehouses, dashboards, Genie and alerts | You have both |
| `USE CATALOG` and `CREATE SCHEMA` | The catalog the workshop deploys into (`workspace` by default) | Creating the workshop's schemas. You then own them, which covers every table, view, comment and tag inside | You own `workspace` |
| **Can use** | The SQL warehouse the workshop runs on (`Serverless Starter Warehouse` by default) | The deploy-time setup, dashboards, Genie, alerts, and running SQL | You're a workspace admin |
| Create jobs, dashboards, Genie agents and alerts | The workspace | The workshop's bundle resources | Every user can by default |
| **Can query** | The pay-per-token model endpoint a workshop names | `ai_query()` steps | Every user can by default |

You don't need to be a workspace admin, create a catalog, create a SQL warehouse, or use classic compute for any
workshop in this repository.

## Limits to know

- **Free Edition:** one SQL warehouse, limited to 2X-Small (that's the Serverless Starter Warehouse), at most 5
  concurrent job tasks per account, and a daily fair-use compute quota. Some models aren't available. Workshops stay
  within these limits; running several at once shares the one warehouse.
- **Trials with a personal email:** one SQL warehouse, scaling to 50 DBUs/hour.
- **One copy per workspace:** each workshop uses fixed names, so a workspace holds one copy of it. A second deploy in
  the same workspace fails without creating anything. Different workshops can run side by side.

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `Catalog 'workspace' does not exist` | Your workspace's default catalog has another name (common outside Free Edition) | Point the workshop at a catalog you can create schemas in: see "Use your own catalog" in its README |
| `failed to resolve warehouse: Serverless Starter Warehouse` | The Starter warehouse was renamed or removed. Nothing was created | Put the name of a SQL warehouse you can use in `databricks.yml`: see "Use another SQL warehouse" in its README |
| `SCHEMA_ALREADY_EXISTS` | The workshop is already deployed in this workspace, maybe by someone else | Use that copy, or ask whoever deployed it to destroy it. Nothing was created, so there's nothing to clean up |
| `PERMISSION_DENIED` on `CREATE SCHEMA` | You lack `CREATE SCHEMA` on the catalog | Ask the catalog owner for `USE CATALOG` and `CREATE SCHEMA`, or choose another catalog |
| A tag statement fails with a policy error | Your account has governed tag policies | Skip the tags cell; nothing later depends on it |
| `ai_query()` can't find the endpoint | That model isn't served in your region or edition | Replace the endpoint name in the query with one listed under **Serving** |

If the automatic deploy fails after Open in Databricks, the repository is still cloned. Fix the cause, then open the
workshop's `databricks.yml`, click the **Deployments** icon and deploy again.
