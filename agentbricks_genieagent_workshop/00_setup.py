# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup (pre-work, about 5 minutes)
# MAGIC
# MAGIC Welcome to **Kasterwell Health**, a fictional virtual-care company whose platform is used by 120 clinics. In the workshop you'll build two things on Kasterwell's data:
# MAGIC 1. a **Genie Agent** that answers questions about the data in plain language (notebook 01), and
# MAGIC 2. your own **AI agent**, built and deployed with the **Agent Bricks CLI**, that uses the Genie Agent as one of its tools (notebook 02).
# MAGIC
# MAGIC This notebook gets everything ready. You'll run it once, before the session. It takes about 5 minutes and you don't need to change any code.
# MAGIC
# MAGIC | What it creates | What it's for |
# MAGIC |---|---|
# MAGIC | 8 data tables (clinics, visits, claims, customer health) | The data your Genie Agent and agent will answer questions about |
# MAGIC | 1 documents table (939 sections of Kasterwell policies and playbooks) | Lets your agent explain *why* something happened and *what to do* |
# MAGIC | 4 **personas** and a row filter | Show how the same question gets a different answer depending on who asks |
# MAGIC | 3 **Unity Catalog functions** | Tools your agent will call in notebook 02 |
# MAGIC
# MAGIC Everything is fictional. No real patients, clinics or payers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before you start
# MAGIC
# MAGIC Check these three things first. They prevent almost every setup problem.
# MAGIC
# MAGIC - **You're in a Databricks workspace you signed up for with your work email.** A free trial created with a personal email (Gmail, Outlook and so on) blocks some features the workshop needs.
# MAGIC - **You imported the whole kit folder**, not just this notebook. Open the folder this notebook is in: you should see the three notebooks (`00_setup`, `01_genie_agent`, `02_agent_bricks_cli`) and a `resources` folder. You only open the three notebooks; `resources` holds the helpers, data and screenshots they use.
# MAGIC - **This notebook is open from inside that folder.** The notebooks find `resources` by looking next to themselves.
# MAGIC
# MAGIC ![The kit folder after import](./resources/images/00_kit_folder.png)
# MAGIC
# MAGIC ### New to Databricks notebooks?
# MAGIC A notebook is a list of **cells**. Grey cells hold code. Cells like this one hold instructions. When you run a code cell, its result appears directly underneath it. You can run one cell with the **▶** button at its top left (or **Shift + Enter**), or every cell from top to bottom with **Run all**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Connect to Serverless compute
# MAGIC
# MAGIC A notebook needs compute to run code. **Serverless** starts for you and needs no setup.
# MAGIC
# MAGIC 1. At the top right of the notebook, click the **compute** dropdown. It's to the right of **Run all**. It may say *Connect* or show a compute name.
# MAGIC 2. Click **Serverless**.
# MAGIC 3. Wait a few seconds for the dot next to *Serverless* to turn **green**.
# MAGIC
# MAGIC ![Choose Serverless from the compute dropdown](./resources/images/00_connect_serverless.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Click Run all
# MAGIC
# MAGIC 1. Click **Run all** in the toolbar.
# MAGIC 2. The notebook runs every cell from top to bottom. Each cell shows a spinning icon while it runs, then a **green check mark** and how long it took.
# MAGIC 3. Scroll down as it runs and compare each result with the screenshots below.
# MAGIC
# MAGIC ![Run all is in the toolbar, next to the compute dropdown](./resources/images/00_run_all.png)
# MAGIC
# MAGIC > **The first cell can take about a minute.** Serverless is starting up. That only happens once per session.
# MAGIC
# MAGIC > **It's safe to run this notebook again.** If anything fails, fix it and click **Run all** again. It replaces what it made last time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Your names for the workshop
# MAGIC The next cell runs a small helper notebook, `_config`. It works out **your** names from your login, so nothing you create clashes with anyone else's. Every notebook in the kit starts with this cell.
# MAGIC
# MAGIC **What you should see:** your login, your **schema** (the folder in the catalog that holds your tables) and your agent name.
# MAGIC - In your own workspace or free trial, your schema is `workspace.agentbricks_genieagent_workshop_lab`.
# MAGIC - In the presenters' workshop workspace it's `kasterwell_labs.<your_name>_<4 characters>`, which is what the screenshots show.

# COMMAND ----------

# MAGIC %run ./resources/_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Load the data (about 2 minutes)
# MAGIC
# MAGIC **What this cell does**
# MAGIC 1. Creates your schema.
# MAGIC 2. Reads each data file from the kit's `data` folder and saves it as a table in your schema.
# MAGIC 3. Adds a **description to every table and column**. Genie reads these descriptions to understand the data, so good descriptions mean better answers. You'll see this in notebook 01.
# MAGIC
# MAGIC **What you should see:** one line per table with its row count. Check that your numbers match:
# MAGIC
# MAGIC ![Expected output: nine tables with their row counts](./resources/images/00_load_output.png)
# MAGIC
# MAGIC **If something goes wrong**
# MAGIC | Message | What it means | Fix |
# MAGIC |---|---|---|
# MAGIC | `FileNotFoundError: .../resources/data/...parquet` | The `resources` folder wasn't imported with the notebooks | Import the whole kit folder again, including `resources` |
# MAGIC | `PERMISSION_DENIED` on `CREATE SCHEMA` | You aren't allowed to create schemas in this workspace | Use your own free trial, where you're the admin |
# MAGIC | `ModuleNotFoundError: genie_config` | This notebook was moved away from its `resources` folder | Open `00_setup` from inside the kit folder |

# COMMAND ----------

import json
from pathlib import Path

import pandas as pd

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {S} COMMENT 'Kasterwell Health workshop (fictional data)'")
import genie_config  # table and column descriptions, shared with the Genie Agent

data = Path(RESOURCES) / "data"
comments = json.loads((data / "_comments.json").read_text())  # documents table

for name in comments:
    spark.sql(f"DROP TABLE IF EXISTS {S}.{name}")  # also drops a row filter from an earlier run
    spark.createDataFrame(pd.read_parquet(data / f"{name}.parquet")).write.saveAsTable(f"{S}.{name}")
    table_text, cols = (genie_config.TABLES[name][0], genie_config.column_comments(name)) if name in genie_config.TABLES \
        else (comments[name]["table"], comments[name]["columns"])
    if table_text:
        spark.sql(f"COMMENT ON TABLE {S}.{name} IS {json.dumps(table_text)}")
    for col, text in cols.items():
        spark.sql(f"ALTER TABLE {S}.{name} ALTER COLUMN {col} COMMENT {json.dumps(text)}")
    print(f"{name:32} {spark.table(f'{S}.{name}').count():>8,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4. Personas: same question, different answer for each person
# MAGIC
# MAGIC In real companies, people see different data. A billing analyst sees every clinic; a regional care coordinator only sees their region. Unity Catalog enforces that with a **row filter**: a rule attached to a table that hides the rows the current person isn't allowed to see. It applies everywhere: notebooks, SQL, Genie and agents.
# MAGIC
# MAGIC **What this cell does**
# MAGIC 1. Creates four personas, from *sees everything* to *sees one clinic*.
# MAGIC 2. Attaches a row filter to every clinic table, so each table only shows the clinics your **current persona** may see.
# MAGIC
# MAGIC You start as the **billing analyst**, who sees all 120 clinics. In notebooks 01 and 02 you'll switch personas with `set_persona(...)` and watch the answers change.
# MAGIC
# MAGIC **What you should see:** a table with the four personas.
# MAGIC
# MAGIC ![Expected output: the four personas](./resources/images/00_personas_output.png)
# MAGIC
# MAGIC | Persona | Sees |
# MAGIC |---|---|
# MAGIC | `billing_analyst` (default) | All 120 clinics |
# MAGIC | `care_coordinator_west` | Only clinics in the West region |
# MAGIC | `clinic_admin_vellamy` | Only one clinic, Vellamy Family Health |
# MAGIC | `compliance_officer` | All 120 clinics |

# COMMAND ----------

vellamy = spark.sql(f"SELECT tenant_id FROM {S}.dim_clinic WHERE clinic_name = 'Vellamy Family Health'").first()[0]
spark.sql(f"""CREATE OR REPLACE TABLE {S}.personas AS SELECT * FROM VALUES
  ('billing_analyst',       'Billing analyst',                      'ALL',  'ALL'),
  ('care_coordinator_west', 'Care coordinator (West)',              'West', 'ALL'),
  ('clinic_admin_vellamy',  'Clinic admin (Vellamy Family Health)', 'ALL',  '{vellamy}'),
  ('compliance_officer',    'Compliance officer',                   'ALL',  'ALL')
  AS t(persona, display_name, region_scope, tenant_scope)""")
spark.sql(f"CREATE TABLE IF NOT EXISTS {S}.persona_assignments (user_key STRING, persona STRING, assigned_at TIMESTAMP)")
spark.sql(f"CREATE TABLE IF NOT EXISTS {S}.kit_settings (key STRING, value STRING)")
spark.sql(f"""CREATE OR REPLACE VIEW {S}.my_persona AS
  SELECT p.* FROM {S}.personas p
  WHERE p.persona = COALESCE((SELECT max_by(persona, assigned_at) FROM {S}.persona_assignments
                              WHERE user_key = sha2(current_user(), 256)), 'billing_analyst')""")
spark.sql(f"CREATE OR REPLACE TABLE {S}.clinic_regions AS SELECT tenant_id, region FROM {S}.dim_clinic")
spark.sql(f"""CREATE OR REPLACE FUNCTION {S}.rf_clinic_scope(tid STRING) RETURNS BOOLEAN
  COMMENT 'Row filter: keep clinics within the current persona''s region and clinic scope'
  RETURN EXISTS (SELECT 1 FROM {S}.my_persona p JOIN {S}.clinic_regions c
                   ON (p.region_scope = 'ALL' OR p.region_scope = c.region)
                  AND (p.tenant_scope = 'ALL' OR p.tenant_scope = c.tenant_id)
                 WHERE c.tenant_id = tid)""")
for t in [n for n in comments if n not in ("dim_payer", "documents")]:
    spark.sql(f"ALTER TABLE {S}.{t} SET ROW FILTER {S}.rf_clinic_scope ON (tenant_id)")
display(spark.table(f"{S}.personas"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5. Create the tools your agent will use
# MAGIC
# MAGIC An agent is more useful when it can *do* things, like look up a clinic or search a policy. These actions are called **tools**. Here, the tools are **Unity Catalog functions**: small SQL functions saved next to your tables and governed the same way.
# MAGIC
# MAGIC | Function | What your agent uses it for |
# MAGIC |---|---|
# MAGIC | `clinic_snapshot(name)` | A one-row health check for one clinic, including whether it's a renewal risk |
# MAGIC | `my_access()` | What the person asking is allowed to see (their persona and how many clinics) |
# MAGIC | `search_docs(question)` | Finds the Kasterwell policies and playbooks that answer a question |
# MAGIC
# MAGIC Your agent will call these functions **as the person chatting with it**, so the row filter from Step 4 applies inside them too.
# MAGIC
# MAGIC > `search_docs` is a simple keyword search, so your workspace doesn't need an always-on search service. The presenters' demo agent uses **AI Search** (vector search, which matches by meaning) for the same job. The *End-to-End AI Agents on Databricks* workshop covers it.
# MAGIC
# MAGIC **What you should see:** the cell finishes with a green check mark and no output. The next cell tries the tools.

# COMMAND ----------

spark.sql(genie_config.clinic_snapshot_sql(S))  # metrics + the CS-02 verdict + network context

spark.sql(f"""CREATE OR REPLACE FUNCTION {S}.my_access()
RETURNS TABLE (persona STRING, role STRING, region_scope STRING, clinic_scope STRING, visible_clinics BIGINT, total_clinics BIGINT)
COMMENT 'What the signed-in user is allowed to see: persona, region/clinic scope and how many of all clinics are visible. Call it when a clinic is not found or before describing results as company-wide.'
RETURN
  SELECT persona, display_name, region_scope, tenant_scope,
         (SELECT count(*) FROM {S}.dim_clinic), (SELECT count(*) FROM {S}.clinic_regions)
  FROM {S}.my_persona""")

spark.sql(f"""CREATE OR REPLACE FUNCTION {S}.search_docs(question STRING COMMENT 'What to look up, in plain language')
RETURNS TABLE (doc_id STRING, title STRING, doc_type STRING, content STRING)
COMMENT 'Searches Kasterwell policies, payer bulletins, clinical operations protocols, product help articles and release notes. Use it to explain WHY something happened or WHAT the recommended action is.'
RETURN
  WITH words AS (SELECT DISTINCT regexp_replace(w, '(ies|es|s)$', '') AS term
                 FROM (SELECT explode(split(lower(question), '[^a-z0-9-]+')) AS w)
                 WHERE length(w) > 3 AND w NOT IN ('what', 'does', 'should', 'there', 'with', 'have', 'this', 'that',
                       'from', 'about', 'which', 'when', 'were', 'they', 'health', 'clinic', 'clinics', 'kasterwell'))
  SELECT d.doc_id, d.title, d.doc_type, d.content
  FROM {S}.documents d JOIN words w ON lower(concat(d.title, ' ', d.content)) LIKE concat('%', w.term, '%')
  GROUP BY d.chunk_id, d.doc_id, d.title, d.doc_type, d.content
  ORDER BY count(*) + 2 * count_if(lower(d.title) LIKE concat('%', w.term, '%')) DESC, length(d.content) DESC
  LIMIT 5""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6. Try two of the tools
# MAGIC
# MAGIC The next cell calls two of the functions you just made, so you can see what your agent will see.
# MAGIC
# MAGIC **What you should see**
# MAGIC 1. **`clinic_snapshot('Vellamy')`** returns one row for *Vellamy Family Health*: renewal date **2026-12-15**, `cs02_at_risk` = **true**, with **5** red signals listed. (CS-02 is Kasterwell's renewal-risk rule; you'll teach it to Genie in notebook 01.)
# MAGIC 2. **`search_docs(...)`** returns five sections of the Larkspan payer bulletin `PB-LARK-2026-07`, the document that explains a rise in denied claims. You'll ask about it in the workshop.
# MAGIC
# MAGIC ![Expected output: the Vellamy snapshot and the matching document sections](./resources/images/00_tools_output.png)

# COMMAND ----------

display(spark.sql(f"SELECT clinic_name, region, renewal_date, cs02_at_risk, cs02_red_signals, cs02_red_signal_list FROM {S}.clinic_snapshot('Vellamy')"))
display(spark.sql(f"SELECT doc_id, title, left(content, 120) AS excerpt FROM {S}.search_docs('Larkspan prior authorization denials psychotherapy')"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ You're set
# MAGIC
# MAGIC The last cell prints **your values for the workshop**. You don't need to write them down: notebooks 01 and 02 work them out again from your login.
# MAGIC
# MAGIC ![Expected output: your values for the workshop](./resources/images/00_values_output.png)
# MAGIC
# MAGIC **Final check:** scroll up. Every code cell should have a green check mark. If one shows a red error, read the *If something goes wrong* table in that step, fix it, and click **Run all** again.
# MAGIC
# MAGIC **Next:** open **01_genie_agent** from the same folder. See you in the workshop!

# COMMAND ----------

print(f"""
Your values for the workshop
  login            {ME}
  schema           {S}
  agent name       {AGENT_NAME}
  app name         {APP_NAME}
  memory store     {MEMORY_STORE}
  sessions store   {SESSION_STORE}
""")
