# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Build and tune a Genie Agent (about 20 minutes)
# MAGIC
# MAGIC A **Genie Agent** answers questions about your data in plain language. It writes the SQL, runs it **as the person asking**, and explains the result. (Genie Agents used to be called *Genie spaces*.)
# MAGIC
# MAGIC You make a Genie Agent good the same way you'd onboard a new analyst: give it the right tables, teach it your business definitions, and check its work. That's exactly what you'll do here.
# MAGIC
# MAGIC | Step | You'll do | You'll learn |
# MAGIC |---|---|---|
# MAGIC | 1 | Create a Genie Agent on five tables | Setup takes a few minutes; the table and column descriptions from notebook 00 do a lot of the work |
# MAGIC | 2 | Ask a business question it can't answer well yet | Without your definition, Genie invents one. **Show code** reveals it |
# MAGIC | 3 | Teach it the definition and one example query | Instructions and example SQL fix the answer |
# MAGIC | 4 | Add a benchmark | A repeatable test that the answer stays right as you change things |
# MAGIC | 5 | Switch persona and ask again | Answers follow each person's data permissions |
# MAGIC | 6 | Call it from code | The same Genie Agent becomes a tool for your agent in notebook 02 |
# MAGIC
# MAGIC **How to work through this notebook:** you'll switch between this notebook and the Genie Agent. Open the Genie Agent in a **new browser tab** so you can keep these instructions open. Run the code cells here one at a time, from top to bottom, when a step tells you to.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before you start
# MAGIC - You've run **00_setup** and every cell finished with a green check mark.
# MAGIC - This notebook is connected to **Serverless** (top right). If not, pick **Serverless** from the compute dropdown, as in notebook 00.
# MAGIC
# MAGIC Run the next three cells, one at a time:
# MAGIC 1. **Install** the latest Databricks SDK, which has the Genie API used in Step 6 (about 20 seconds).
# MAGIC 2. **Restart Python** so the new version loads. The cell shows a green check mark; anything you ran before it is cleared.
# MAGIC 3. **Load your names**, as in notebook 00.

# COMMAND ----------

# MAGIC %pip install -q -U "databricks-sdk>=0.60"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./resources/_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Create your Genie Agent
# MAGIC
# MAGIC First, run the cell below. It prints your **schema** (where your tables are) and the **starting instructions** you'll paste in a moment.

# COMMAND ----------

import genie_config  # the finished Genie Agent's configuration, shared with the presenters' version

BASE_INSTRUCTIONS = genie_config.BASE_INSTRUCTIONS

print(f"Your schema:  {S}\nYour tables:  {', '.join(GENIE_TABLES)}\n\n"
      f"── Starting instructions (copy everything below this line) ──\n{BASE_INSTRUCTIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Open Genie Agents and click New
# MAGIC 1. Right-click **Genie Agents** in the left sidebar and open it in a **new tab**. (It's under **SQL**.)
# MAGIC 2. Click **New** at the top right.
# MAGIC
# MAGIC ![Genie Agents in the sidebar, then New](./resources/images/01_new_genie.png)
# MAGIC
# MAGIC ### 1b. Choose to pick your tables yourself
# MAGIC A box asks *"What should your Genie agent do?"*. Skip it: click **Select tables manually** underneath.
# MAGIC
# MAGIC ![Click Select tables manually](./resources/images/01_select_tables_manually.png)
# MAGIC
# MAGIC ### 1c. Add your five tables
# MAGIC 1. In **Connect your data**, click **All**.
# MAGIC 2. Click your catalog, then your schema. Both were printed by the cell above (**Your schema** is `catalog.schema`).
# MAGIC    - Free trial: click **workspace**, then **agentbricks_genieagent_workshop_lab**.
# MAGIC    - Presenters' workspace: click **kasterwell_labs**, then your own schema (named after you; the screenshots show a presenter's).
# MAGIC 3. Tick these **five** tables: `appointment_access`, `claims_monthly`, `dim_clinic`, `revenue_monthly`, `tenant_health_weekly`. The bottom of the box should say *Selected:* with 5 tables (2 shown plus **+3**).
# MAGIC 4. Click **Create manually**.
# MAGIC
# MAGIC ![Tick the five tables, then Create manually](./resources/images/01_pick_tables.png)
# MAGIC
# MAGIC > Why only five tables? A Genie Agent works best with a focused set of tables for one business area. These five cover clinics, customer health, claims, appointments and invoices.
# MAGIC
# MAGIC ### 1d. Name it
# MAGIC Genie gives the new agent a generated name. Rename it so you can find it later:
# MAGIC 1. In the right-hand panel, open the **About** tab and scroll to **About this agent**. Click its **pencil** icon.
# MAGIC 2. Replace the **Name** with **Kasterwell Clinic Ops**. Leave **Default warehouse** as it is: Genie picks one for you (in a free trial it's the *Serverless Starter Warehouse*).
# MAGIC 3. Click **Save**.
# MAGIC
# MAGIC ![Rename the agent, then Save](./resources/images/01_rename.png)
# MAGIC
# MAGIC ### 1e. Paste the starting instructions
# MAGIC 1. In the right-hand panel, click the **Instructions** tab.
# MAGIC 2. Click inside the **General Instructions** box and paste the starting instructions printed by the cell above (everything under the *Starting instructions* line).
# MAGIC 3. Click **Save**.
# MAGIC
# MAGIC ![Instructions tab, paste, Save](./resources/images/01_instructions.png)
# MAGIC
# MAGIC 💡 Notice what you *didn't* have to write: Genie already reads the table and column descriptions loaded in notebook 00. Good descriptions are the cheapest way to a good Genie Agent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Ask a question it can't answer well yet
# MAGIC
# MAGIC 1. Under the question box there are two modes: **Agent** and **Chat**. Click **Chat**. Chat answers in one step, usually in under a minute, and it's the mode your agent in notebook 02 uses.
# MAGIC
# MAGIC    ![Switch the question box to Chat](./resources/images/01_chat_mode.png)
# MAGIC
# MAGIC 2. Type this question and press **Enter**:
# MAGIC
# MAGIC    > **Which clinics are our biggest renewal risk?**
# MAGIC
# MAGIC 3. Wait for **Analysis complete**. The answer looks convincing: a ranked **top 10** with a "risk score" and a chart.
# MAGIC 4. Click **Show code** on the result table and read the SQL.
# MAGIC
# MAGIC ![The before answer: an invented risk score and a top 10](./resources/images/01_before_answer.png)
# MAGIC
# MAGIC ![Show code reveals the formula Genie made up](./resources/images/01_before_code.png)
# MAGIC
# MAGIC **What just happened?** Kasterwell has a precise definition of renewal risk, but Genie was never told it. So it **invented** its own "risk score" from whatever columns looked relevant, and returned ten clinics. Most of them aren't at risk by Kasterwell's rules, and its numbers for Vellamy (for example 46.8% seat utilization) don't match the playbook's. A new analyst would make the same guess, and nobody would notice unless they read the SQL.
# MAGIC
# MAGIC > Your answer may look a little different: Genie's guesses vary from run to run. Look for the same pattern, a made-up score or threshold in **Show code**.
# MAGIC >
# MAGIC > **Curious about Agent mode?** It reasons over several steps and usually names Vellamy, but it still makes up its own rule (for example an 8-week window and a 50% cutoff) and skips signals such as late invoices. Same lesson: without your definition, the numbers aren't yours.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Teach it your definition
# MAGIC
# MAGIC Kasterwell's renewal-risk playbook (**CS-02**) is precise: score five signals over the last 4 weeks, and a clinic is **at risk** with **two or more** red signals. You'll teach it to Genie in two ways:
# MAGIC - a **text instruction** that explains the rule in words, and
# MAGIC - an **example query**: a question paired with SQL that answers it correctly. Genie reuses the pattern for similar questions.
# MAGIC
# MAGIC Run the cell below to print both.

# COMMAND ----------

RENEWAL_DEFINITION = genie_config.RENEWAL_DEFINITION
RENEWAL_QUESTION = "Which clinics are our biggest renewal risk?"
RENEWAL_SQL = genie_config.renewal_risk_sql(S)

print(f"── 1. Definition (paste at the end of General Instructions) ──\n{RENEWAL_DEFINITION}\n\n"
      f"── 2. Example question (paste as the example's title) ──\n{RENEWAL_QUESTION}\n\n"
      f"── 3. Example SQL (paste as the example's SQL) ──\n{RENEWAL_SQL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Add the definition to the instructions
# MAGIC 1. In the Genie Agent, open the **Instructions** tab again.
# MAGIC 2. Click at the very **end** of the existing text, press **Enter** twice, and paste the **Definition** printed above.
# MAGIC 3. Click **Save**.
# MAGIC
# MAGIC ![Paste the definition at the end, then Save](./resources/images/01_add_definition.png)
# MAGIC
# MAGIC ### 3b. Add the example query
# MAGIC 1. Open the **Examples** tab. Click **Add**, then **Example query**.
# MAGIC
# MAGIC    ![Examples tab, Add, Example query](./resources/images/01_add_example_menu.png)
# MAGIC
# MAGIC 2. In the title box at the top, paste the **Example question**: `Which clinics are our biggest renewal risk?`
# MAGIC 3. In the SQL box below it, paste the **Example SQL**.
# MAGIC 4. Click **Save**. The example now appears in the list, marked **QUERY**.
# MAGIC
# MAGIC ![The saved example query](./resources/images/01_example_saved.png)
# MAGIC
# MAGIC ### 3c. Ask again in a new chat
# MAGIC 1. Click **New chat** (the pencil icon at the top right). A new chat makes sure Genie isn't reusing the earlier answer.
# MAGIC 2. Check that **Chat** is still selected under the question box, then ask the same question again:
# MAGIC
# MAGIC    > **Which clinics are our biggest renewal risk?**
# MAGIC
# MAGIC **What you should see:** exactly **one** clinic, **Vellamy Family Health**, red on all **5** signals. These are the playbook's numbers:
# MAGIC
# MAGIC | Signal (last 4 weeks) | Vellamy | Red when |
# MAGIC |---|---|---|
# MAGIC | Seat utilization | 51.1% | below 55% |
# MAGIC | P1 tickets | 12 | 3 or more |
# MAGIC | CSAT | 2.2 | below 3.5 |
# MAGIC | Visits vs the 12 weeks before | down 28% | down more than 20% |
# MAGIC | Worst late invoice | 30 days | more than 15 days |
# MAGIC
# MAGIC It renews on **2026-12-15**, with **$213,192** of ARR at stake. Click **Show code**: the SQL now matches the playbook.
# MAGIC
# MAGIC ![The after answer: Vellamy only, with the playbook's numbers](./resources/images/01_after_answer.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4. Add a benchmark
# MAGIC
# MAGIC Every change to the instructions can fix one answer and quietly break another. A **benchmark** is a test question with a known-correct SQL answer. Genie answers the question fresh, and its result is compared with yours. You can re-run all benchmarks any time.
# MAGIC
# MAGIC The benchmark asks the question in **different words** ("at risk of churning"), to check Genie applies the definition rather than memorizing one phrase. Its correct answer only needs the columns that matter (clinic, renewal date, ARR), so any correct answer passes.
# MAGIC
# MAGIC Run the cell below to print the benchmark question and its correct SQL.

# COMMAND ----------

BENCHMARK_QUESTION = "Which clinics are at risk of churning?"
BENCHMARK_SQL = f"SELECT clinic_name, renewal_date, arr_usd\nFROM (\n{RENEWAL_SQL}\n)\nORDER BY renewal_date"

print(f"── Benchmark question ──\n{BENCHMARK_QUESTION}\n\n── Ground truth SQL answer ──\n{BENCHMARK_SQL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Add the benchmark
# MAGIC 1. In the Genie Agent, click **Benchmark** in the bar at the top of the page.
# MAGIC 2. Click **Add benchmark**.
# MAGIC 3. Paste the **Benchmark question** into **Question**, and the **Ground truth SQL answer** into the box below it. Leave **Evaluation note** empty.
# MAGIC 4. Click **Add benchmark**.
# MAGIC
# MAGIC ![Fill in the question and the ground truth SQL, then Add benchmark](./resources/images/01_benchmark_add.png)
# MAGIC
# MAGIC ### 4b. Run it
# MAGIC 1. Click **Mode: Agent** and choose **Chat**, to test the same mode you've been using.
# MAGIC 2. Click **Run all benchmarks**. It takes about a minute.
# MAGIC
# MAGIC ![Choose Chat mode, then Run all benchmarks](./resources/images/01_benchmark_mode.png)
# MAGIC
# MAGIC **What you should see:** **100% accurate (1/1)**, with the assessment **Good**. Genie's answer and your ground truth both return Vellamy Family Health.
# MAGIC
# MAGIC ![100% accurate, assessment Good](./resources/images/01_benchmark_result.png)
# MAGIC
# MAGIC > If it says **Bad**, click the question to compare Genie's SQL with yours. The usual cause is a definition or example that wasn't saved: check the **Instructions** and **Examples** tabs, then click **Start new run**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5. Same question, different person
# MAGIC
# MAGIC Genie runs every query **as the person asking**, so the row filter from notebook 00 applies to every answer. You'll switch persona from this notebook, then ask Genie the same question each time.
# MAGIC
# MAGIC For each persona below:
# MAGIC 1. Run its `set_persona(...)` cell (just below these instructions).
# MAGIC 2. In the Genie Agent, click **New chat** and ask: **How many clinics do we have in each region?**
# MAGIC
# MAGIC | Persona cell | What you should see |
# MAGIC |---|---|
# MAGIC | `set_persona("care_coordinator_west")` | Only **West: 52 clinics** |
# MAGIC | `set_persona("clinic_admin_vellamy")` | Only **Central: 1 clinic** (Vellamy Family Health) |
# MAGIC | `set_persona("billing_analyst")` | All three regions: **West 52, Central 34, East 34** (120 clinics). This is the default, so finish on it |
# MAGIC
# MAGIC Always click **New chat** after switching, so Genie runs a fresh query.
# MAGIC
# MAGIC ![As the West care coordinator: only West, 52 clinics](./resources/images/01_persona_west.png)
# MAGIC
# MAGIC ![As the Vellamy clinic admin: only Central, 1 clinic](./resources/images/01_persona_vellamy.png)
# MAGIC
# MAGIC Nothing in the Genie Agent changed. The row filter on the tables did the work, and it applies the same way in SQL, dashboards, Genie and the agent you'll build next.

# COMMAND ----------

set_persona("care_coordinator_west")

# COMMAND ----------

set_persona("clinic_admin_vellamy")

# COMMAND ----------

set_persona("billing_analyst")  # back to the default: all 120 clinics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6. Call your Genie Agent from code
# MAGIC
# MAGIC Everything you just did in the browser is also available to code through the **Conversation API**. It's how apps embed Genie, and how your agent in notebook 02 will use this Genie Agent as a tool.
# MAGIC
# MAGIC 1. In the Genie Agent, open the **About** tab and scroll to **About this agent**. Copy the **Agent ID**. (It's also the last part of the page address: `.../genie/rooms/<agent id>`.)
# MAGIC
# MAGIC    ![The Agent ID is on the About tab](./resources/images/01_agent_id.png)
# MAGIC
# MAGIC 2. In the cell below, paste it between the quotes: `GENIE_SPACE_ID = "01f1..."`.
# MAGIC 3. Run the cell. It asks your Genie Agent the renewal-risk question and prints the answer and the SQL it ran. It also **saves the ID** so notebook 02 can find it.
# MAGIC
# MAGIC **What you should see:** an answer naming Vellamy Family Health with the same five signals, then the SQL it ran (the CS-02 query). It takes up to a minute.
# MAGIC
# MAGIC ![Expected output: the answer, then the SQL it ran](./resources/images/01_step6_output.png)

# COMMAND ----------

GENIE_SPACE_ID = ""  # paste your Genie Agent id here

GENIE_SPACE_ID = setting("genie_space_id", GENIE_SPACE_ID) if GENIE_SPACE_ID else setting("genie_space_id")

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
if not GENIE_SPACE_ID:
    print("Paste your Genie Agent id above and re-run this cell (or run the 'Skip the clicks' cell at the bottom).")
else:
    msg = w.genie.start_conversation_and_wait(GENIE_SPACE_ID, RENEWAL_QUESTION)
    attachments = msg.attachments or []
    texts = [a.text.content for a in attachments if a.text and a.text.content]
    answer = max(texts, key=len) if texts else "(no text answer; see the SQL below)"  # skip short follow-up suggestions
    print("Answer:", answer.replace("**", ""))
    for a in attachments:
        if a.query:
            print("\nSQL it ran (as you):\n", a.query.query)
    print(f"\n✅ Saved Genie Agent id {GENIE_SPACE_ID} for notebook 02.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### If something goes wrong
# MAGIC | Symptom | Cause | Fix |
# MAGIC |---|---|---|
# MAGIC | Your tables aren't listed in **Connect your data** | Wrong catalog or schema, or 00_setup didn't finish | Use the schema printed in Step 1; re-run 00_setup if it's empty |
# MAGIC | The "after" answer still shows a risk score or several clinics | The definition or example wasn't saved, or you asked in the old chat | Check the **Instructions** and **Examples** tabs, then click **New chat** and ask again |
# MAGIC | Persona doesn't change the answer | You asked in the same chat | Click **New chat** after running a `set_persona` cell |
# MAGIC | `'WorkspaceClient' object has no attribute 'genie'` | The install or restart cell was skipped | Run the install cell, then the restart cell, then continue from `%run ./resources/_config` |
# MAGIC | `Paste your Genie Agent id above` | Step 6 has no ID yet | Paste the Agent ID from the **About** tab between the quotes |
# MAGIC | Behind or stuck | | Run the **Skip the clicks** cell at the bottom: it builds a finished Genie Agent and saves its ID |
# MAGIC
# MAGIC ### Key takeaways
# MAGIC - A Genie Agent is the fastest way to give people governed, plain-language access to data.
# MAGIC - Quality comes from **context**: table descriptions, instructions and example SQL. **Benchmarks** keep it honest as you change things.
# MAGIC - It answers **as the user**, and the same Genie Agent can be a **tool** inside a larger agent. That's next, in **02_agent_bricks_cli**.
# MAGIC
# MAGIC ---
# MAGIC ### Skip the clicks (optional)
# MAGIC Short on time, or following along later? The cell below creates the finished, production version of the Genie Agent through the API and saves its ID for notebook 02. You don't need it if you finished Steps 1 to 6.
# MAGIC
# MAGIC It goes further than the lesson, with what a Genie Agent needs before real users rely on it (all defined in `genie_config.py`):
# MAGIC - descriptions and synonyms for every column, and **value matching** on names and categories ("behavioral health" → `behavioral_health`);
# MAGIC - **joins** between the tables, so Genie never guesses a join key;
# MAGIC - **measures** such as denial rate and weighted wait time, so a rate is never averaged row by row, plus common **filters** (psychotherapy claims, last 4 weeks, renewing in 90 days);
# MAGIC - instructions for the data window, unfinished months and limited access, 10 worked **example queries**, and 23 **benchmarks** with known answers.
# MAGIC
# MAGIC After it runs, click the link it prints, then open the **Instructions**, **Examples** and **Benchmark** pages to explore them.

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
warehouses = list(w.warehouses.list())
warehouse = next((wh for wh in warehouses if wh.enable_serverless_compute), warehouses[0])  # prefer serverless
r = w.api_client.do("POST", "/api/2.0/genie/spaces", body={
    "warehouse_id": warehouse.id, "title": "Kasterwell Clinic Ops", "parent_path": f"/Users/{ME}",
    "description": "Clinic operations at Kasterwell Health (fictional). Answers follow your persona's access.",
    "serialized_space": json.dumps(genie_config.build_space(S, GENIE_TABLES))})
GENIE_SPACE_ID = setting("genie_space_id", r["space_id"])
host = spark.conf.get("spark.databricks.workspaceUrl")
displayHTML(f'<a href="https://{host}/genie/rooms/{GENIE_SPACE_ID}" target="_blank">Open your Genie Agent →</a> (id {GENIE_SPACE_ID}, saved for notebook 02)')
