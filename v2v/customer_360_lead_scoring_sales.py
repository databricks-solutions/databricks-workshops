# Databricks notebook source

# MAGIC %md
# MAGIC # Customer 360, Lead Scoring, Churn & Upsell Prediction for Sales
# MAGIC **For sales leaders:** one view of the customer drives three sales plays — **know the account, rank the leads, and catch churn and upsell before the quarter ends.** AI Functions turn raw CRM, billing and support text into signals; MLflow handles the model that goes back into the CRM; Genie and an AI/BI dashboard put the answers in front of reps and managers.
# MAGIC
# MAGIC **Industry:** Cross-industry — any B2B business with a CRM, a billing system and a support desk (Financial Services, Healthcare & Life Sciences, Manufacturing, Technology)
# MAGIC
# MAGIC Every QBR starts the same way: *What do we actually know about this account? Which leads should reps call this week? Which renewals are at risk — and which accounts could buy more?* The answers are spread across Salesforce, billing, product telemetry, the support desk and NPS surveys — and none of them share an account ID.
# MAGIC This notebook joins them into **one governed customer view in Unity Catalog**, scores leads and renewals, and hands the results to the business through **Genie** — with **MLflow** proving every match and model is accurate.
# MAGIC
# MAGIC | Section | Question | Databricks capability |
# MAGIC |---|---|---|
# MAGIC | **1 · Customer 360** | What do we know about this account? | Identity resolution with `ai_similarity` + `ai_classify`, `ai_enrich` firmographics, a **Lakeflow Job** with an **`ai_classify` AI Function task** for tickets, `ai_analyze_sentiment`, `ai_summarize` · **MLflow** matching accuracy |
# MAGIC | **2 · Lead scoring** | Which leads should reps call this week, and why? | **MLflow** model (scikit-learn) in the **Unity Catalog registry** with **SHAP** reasons, compared with `ai_predict_class`; `ai_gen` talking points and outreach |
# MAGIC | **3 · Churn & upsell** | Which accounts do we need to save, and which can we grow, before the quarter ends? | `ai_predict_class` ×2, `ai_detect_anomalies`, `ai_top_drivers`, `ai_gen` next-best actions · **MLflow** hold-out accuracy |
# MAGIC | **4 · Wrap-up** | How do reps and managers use this every day? | **Genie** agent, one-page **AI/BI dashboard**, **lineage**, the **MLflow** trust layer |
# MAGIC | 💬 | After sections 0–3: **ask Genie** the same question in plain English — no SQL | **Genie** agent on Unity Catalog |
# MAGIC | 🛠 | After sections 0–3: a **Try it yourself** exercise, and a *Build more* list at the end | |
# MAGIC
# MAGIC ---
# MAGIC ### How to run
# MAGIC 1. Keep the default **Serverless** compute. Steps that use `ai_detect_anomalies` or `ai_top_drivers` (which run on Databricks SQL) automatically send their SQL to a **Serverless or Pro SQL warehouse** in your workspace (the notebook finds one for you).
# MAGIC 2. A **workspace admin** enables these on the **Previews** page (profile menu → *Previews*): **Predictive AI Functions** (Beta — `ai_predict_class`, `ai_detect_anomalies`, `ai_top_drivers`), **Ai Enrich** (Beta — `ai_enrich`), **Lakeflow AI Function Task** (Beta — the ticket-classification job) and the **AI Functions** preview for `ai_gen`. **If any AI Function cell fails, this is the first thing to check.**
# MAGIC 3. Run **S.0–S.3** and set the widgets (`catalog`, `schema`, …) to a location where you can create tables. Then either **Run all** (about 45 minutes — mostly the ticket-classification job, which starts in the background in step 0.7), or — better for learning — run section by section: each section ends with a **💬 Ask Genie** cell and a **🛠 Try it yourself** exercise.
# MAGIC 4. **Every section and cell is numbered** (S.1, 0.1, 1.1 …) — look for the number in each cell's title.
# MAGIC 5. **New to some of the code?** Ask **Genie Code** to explain any cell: click into it, open Genie Code and type `/explain` (or ask *"what does this cell do?"*).
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews listed in step 2 above. Then re-run the failed cell. The `ai_detect_anomalies` step also falls back automatically to a rolling-baseline detector so the rest of the notebook keeps running — check the `method` column.
# MAGIC
# MAGIC > **No data? No problem.** Section 0 generates a realistic, fully synthetic go-to-market dataset — a Salesforce export with duplicate accounts, a billing system and a support desk that don't share its account IDs, product telemetry, NPS surveys, 6,000 leads and three years of renewals — plus an answer key (`ground_truth`) so MLflow can score every match and prediction. **Have your own data?** See *"Bring your own data"* (step 0.11).
# MAGIC
# MAGIC **Terms used in this notebook:** **identity resolution** = deciding which records in different systems are the same customer · **master account** = the one surviving CRM record per customer · **ARR** = annual recurring revenue · **seat utilization** = active users ÷ seats purchased · **SHAP** = per-row explanation of a model's score · **AUC** = how well a score ranks positives above negatives (0.5 = random, 1 = perfect) · **top-decile lift** = conversion rate in the top 10% of scores ÷ overall rate.
# MAGIC
# MAGIC *The selling company, people and most customers are fictitious. Twenty account rows deliberately use the names of real public companies (with no industry or segment filled in) so `ai_enrich` has real firmographics to find; all of their numbers are synthetic.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## S. Setup — choose where to write the demo tables
# MAGIC Four short setup steps. The **widgets** (the notebook's parameters) drive everything below.
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | **S.0** | Installs the Python libraries for the MLflow steps (MLflow 3, Databricks Feature Engineering, scikit-learn, SHAP, Faker) in **one** `%pip` call and restarts Python |
# MAGIC | **S.1** | Creates the widgets |
# MAGIC | **S.2** | Creates / selects the catalog, the `sales` schema and the `crm_export` volume, and turns the widget values into SQL session variables |
# MAGIC | **S.3** | Loads the Python helper functions used later (Unity Catalog links, SQL warehouse runner, Genie, MLflow, dashboard) |
# MAGIC
# MAGIC | Widget | Meaning | Default |
# MAGIC |---|---|---|
# MAGIC | `catalog` / `schema` | Where tables are created in **Unity Catalog**. It's created with `IF NOT EXISTS`, so other demo notebooks can share it. If you can't create catalogs, the current catalog is used | `demo_cfo_sales` / `sales` |
# MAGIC | `match_threshold` | `ai_similarity` score at or above which two records are auto-matched | `0.85` |
# MAGIC | `review_floor` | Pairs between this and `match_threshold` are borderline — `ai_classify` decides | `0.60` |
# MAGIC | `enrich_with_web_search` | Ground `ai_enrich` in a web search (if your workspace has it) | `no` |
# MAGIC | `anomaly_threshold` | 0–1. Lower = more sensitive usage-drop alerts | `0.95` |
# MAGIC | `question` / `custom_question` | What to ask Genie in step 6.1 | starter questions |
# MAGIC
# MAGIC > 🧭 **Where are the widgets?** They show up once you run **S.1**, in one of two places depending on your layout:
# MAGIC > * **A bar across the top** of the notebook (the classic layout), or
# MAGIC > * **A panel on the left-hand side** — click the **Parameters** icon (the parameters symbol in the left sidebar) to open it.
# MAGIC >
# MAGIC > Same widgets either way — use whichever you prefer.
# MAGIC
# MAGIC > 🔁 **Changed a widget?** Re-run **S.2 and S.3**, then the cells you want to refresh (or just **Run all**).

# COMMAND ----------

# MAGIC %pip install -q --upgrade databricks-feature-engineering "mlflow[databricks]>=3.9" "scikit-learn>=1.5" shap faker "databricks-sdk>=0.70"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Widgets — they appear in the bar at the top of the notebook, or in the left-hand Parameters panel
GENIE_QUESTIONS = [
    "Show me everything about Acme.",
    "Which accounts grew usage but not spend?",
    "Who are the top 20 leads this week?",
    "What's the win rate by score decile?",
    "How much ARR is at risk next quarter?",
    "Which accounts are best for upsell?",
    "Why is ARR at risk going up?",
]
dbutils.widgets.text("catalog", "demo_cfo_sales", "1. Catalog")
dbutils.widgets.text("schema", "sales", "2. Schema")
dbutils.widgets.dropdown("match_threshold", "0.85", ["0.80", "0.85", "0.90"], "3. Auto-match at similarity ≥")
dbutils.widgets.dropdown("review_floor", "0.60", ["0.50", "0.60", "0.70"], "4. ai_classify borderline pairs from")
dbutils.widgets.dropdown("enrich_with_web_search", "no", ["no", "yes"], "5. Ground ai_enrich in web search")
dbutils.widgets.dropdown("anomaly_threshold", "0.95", ["0.90", "0.95", "0.99"], "6. Anomaly sensitivity (lower = more alerts)")
dbutils.widgets.dropdown("question", GENIE_QUESTIONS[0], GENIE_QUESTIONS, "7. Ask Genie (any time)")
dbutils.widgets.text("custom_question", "", "8. ...or type your own Genie question")

# COMMAND ----------

# Changed a widget? Re-run this cell (S.2) and the helpers (S.3), then the cells you want to refresh.
cat = dbutils.widgets.get("catalog").strip() or "demo_cfo_sales"
sch = dbutils.widgets.get("schema").strip() or "sales"
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS `{cat}` COMMENT 'AI Functions demos (synthetic data)'")
except Exception as e:
    fallback = spark.sql("SELECT current_catalog()").first()[0]
    print(f"⚠️ You don't have permission to create catalog '{cat}' — using the current catalog '{fallback}' instead.")
    cat = fallback
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{cat}`.`{sch}` COMMENT 'Sales: customer 360, lead scoring, churn and upsell (synthetic data)'")
spark.sql(f"USE `{cat}`.`{sch}`")
spark.sql("CREATE VOLUME IF NOT EXISTS crm_export COMMENT 'Salesforce export (CSV): accounts, contacts, leads, engagement, opportunities'")

num = lambda name, default: float(dbutils.widgets.get(name) or default)
spark.sql(f"DECLARE OR REPLACE VARIABLE match_threshold DOUBLE DEFAULT {num('match_threshold', 0.85)}")
spark.sql(f"DECLARE OR REPLACE VARIABLE review_floor DOUBLE DEFAULT {num('review_floor', 0.6)}")
spark.sql(f"DECLARE OR REPLACE VARIABLE anomaly_threshold DOUBLE DEFAULT {num('anomaly_threshold', 0.95)}")
spark.sql("DECLARE OR REPLACE VARIABLE enrich_sources STRING DEFAULT " +
          ("'[{\"type\": \"web_search\"}]'" if dbutils.widgets.get("enrich_with_web_search") == "yes" else "NULL"))
spark.sql("DECLARE OR REPLACE VARIABLE lme DATE DEFAULT last_day(add_months(current_date(), -1))")
spark.sql("DECLARE OR REPLACE VARIABLE next_q_start DATE DEFAULT CAST(date_trunc('QUARTER', add_months(current_date(), 3)) AS DATE)")
spark.sql("DECLARE OR REPLACE VARIABLE next_q_end DATE DEFAULT date_sub(add_months(next_q_start, 3), 1)")
spark.sql("DECLARE OR REPLACE VARIABLE last_q_start DATE DEFAULT add_months(CAST(date_trunc('QUARTER', current_date()) AS DATE), -3)")
display(spark.sql("""SELECT current_catalog() AS catalog, current_schema() AS schema, match_threshold, review_floor,
       enrich_sources IS NOT NULL AS ai_enrich_web_search, anomaly_threshold,
       concat('Q', quarter(next_q_start), ' ', year(next_q_start)) AS next_quarter, next_q_start, next_q_end"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### S.3 Load the helper functions
# MAGIC Step S.3 **doesn't create any tables or change any data** — it only defines Python helpers the rest of the notebook calls, so later cells stay short and readable:
# MAGIC
# MAGIC | Helper | Used in | What it does |
# MAGIC |---|---|---|
# MAGIC | `WorkspaceClient()`, `CATALOG`, `SCHEMA`, `CRM_EXPORT` | everywhere | Connects to this workspace with **your** identity and remembers where the demo tables and the `crm_export` volume live (from S.2) |
# MAGIC | `pick_warehouse()` / `run_on_sql_warehouse()` | steps 3.4, 3.5 | Finds a **Serverless or Pro SQL warehouse** and sends `ai_detect_anomalies` / `ai_top_drivers` SQL to it *exactly as written*, passing in your widget values |
# MAGIC | `uc_url()`, `button()`, `show_uc_cards()` | step 0.6 and later | Builds clickable **Unity Catalog** links and the table cards (layer · rows · sample · lineage) |
# MAGIC | `ask_genie()`, `show_genie_answer()` — and `ask()`, used once the agent exists (step 0.8) | every 💬 Ask Genie step | Asks the **Genie agent** a question through the Genie API and renders the answer, the SQL Genie wrote and a *Continue in Genie* link |
# MAGIC | `experiment()`, `show_run()` | steps 1.7, 2.2, 2.5, 3.7 | Opens the notebook's **MLflow experiment** — created with **Unity Catalog trace storage** (traces land in UC tables you can query with SQL), reused only if it's active — and shows a link to each run |
# MAGIC | `eval_dataset()` | step 1.7 | Creates (or updates) a versioned **MLflow evaluation dataset stored in Unity Catalog** — it shows in the experiment's *Datasets* tab |
# MAGIC | `set_alias()` | step 2.2 | Sets the **`@champion`** alias on the registered lead model version that step 2.3 scores with |
# MAGIC | `create_or_update_job()`, `show_task()`, `wait_for_run()`, `run_job()`, `publish_dashboard()` | steps 0.7, 1.4a, 1.8, 4.2 | Creates the Lakeflow Job, renders an AI Function task definition, waits for / runs the job and builds the AI/BI dashboard |
# MAGIC
# MAGIC When it finishes it prints the catalog / schema and the SQL warehouse it will use. **Re-run this cell whenever you change the `catalog` or `schema` widget** (after S.2).
# MAGIC
# MAGIC > 💡 **Don't fully understand a cell?** Ask **Genie Code** to explain it: click into the cell, open Genie Code and type `/explain` — or just ask *"what does this cell do?"* in plain English.

# COMMAND ----------

import json, time, uuid, html, re
import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
CATALOG, SCHEMA = spark.sql("SELECT current_catalog(), current_schema()").first()
HOST = w.config.host.rstrip("/")
try:
    ORG = f"?o={w.get_workspace_id()}"
except Exception:
    ORG = ""
NB_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
NB_FOLDER = NB_PATH.rsplit("/", 1)[0]
fq = lambda t: f"{CATALOG}.{SCHEMA}.{t}"
_q = lambda s: str(s).replace("'", "''")

def _api(method, path, body=None, query=None):
    return w.api_client.do(method, path, body=body, query=query)

# ---------------------------------------------------------------- SQL warehouse (ai_forecast / ai_detect_anomalies / Genie)
def pick_warehouse():
    # Serverless or Pro SQL warehouses only (skip Classic and real-time / special-purpose warehouses)
    whs = [x for x in w.warehouses.list() if str(x.warehouse_type).endswith("PRO")]
    if not whs:
        raise Exception("Create a Serverless or Pro SQL warehouse — ai_forecast, ai_detect_anomalies and Genie run on Databricks SQL.")
    whs.sort(key=lambda x: ("RUNNING" not in str(x.state), not x.enable_serverless_compute))
    return whs[0]
WAREHOUSE = pick_warehouse()

def run_on_sql_warehouse(statement, **variables):
    """Runs the SQL exactly as written on the SQL warehouse. Keyword args become SQL variables: name=(type, value or SQL expression)."""
    decl = "".join(f"  DECLARE {k} {t} DEFAULT {v};\n" for k, (t, v) in variables.items())
    wrapped = f"BEGIN\n{decl}{statement.strip()}\nEND" if decl else statement.strip()
    r = w.statement_execution.execute_statement(statement=wrapped, warehouse_id=WAREHOUSE.id, catalog=CATALOG, schema=SCHEMA, wait_timeout="50s")
    while r.status.state.value in ("PENDING", "RUNNING"):
        time.sleep(5)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise Exception(f"SQL warehouse '{WAREHOUSE.name}': {r.status.error.message if r.status.error else r.status.state}")
    print(f"✓ Ran on SQL warehouse '{WAREHOUSE.name}'")

# ---------------------------------------------------------------- Unity Catalog links and cards
def uc_url(*parts, tab=None):
    q = ORG + ((("&" if ORG else "?") + f"activeTab={tab}") if tab else "")
    return f"{HOST}/explore/data/" + "/".join(parts) + q

def volume_url(name):
    return f"{HOST}/explore/data/volumes/{CATALOG}/{SCHEMA}/{name}{ORG}"

FONT = "font-family:DM Sans,Helvetica,Arial,sans-serif"
def button(label, url, color="#FF3621"):
    return (f'<a href="{url}" target="_blank" style="display:inline-block;margin:4px 8px 4px 0;padding:8px 14px;'
            f'background:{color};color:white;border-radius:6px;text-decoration:none;font-weight:600;{FONT};font-size:13px">'
            f'{html.escape(label)} ↗</a>')

ROLE_COLOR = {}
def existing_tables():
    return {r.tableName for r in spark.sql(f"SHOW TABLES IN `{CATALOG}`.`{SCHEMA}`").collect() if not r.isTemporary}

def show_uc_cards(roles=None, title=None, volume=None):
    have, cards = existing_tables(), []
    if volume:
        n = len(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{volume}"))
        cards.append(f"""<div style="border:1px solid #e0e0e0;border-radius:10px;padding:12px 14px;width:280px;{FONT}">
          <div style="font-size:11px;font-weight:700;color:#FF3621;text-transform:uppercase;letter-spacing:.04em">Volume · unstructured files</div>
          <div style="font-size:15px;font-weight:700;margin:4px 0"><a href="{volume_url(volume)}" target="_blank" style="color:#1B3139">{volume}</a></div>
          <div style="font-size:12px;color:#555;min-height:32px">{n:,} files governed like a table</div>
          <div style="font-size:12px;margin-top:6px"><a href="{volume_url(volume)}" target="_blank">browse &amp; preview the files</a></div></div>""")
    for t, (role, desc, _) in TABLE_INFO.items():
        if t not in have or (roles and role not in roles):
            continue
        n = spark.sql(f"SELECT count(*) FROM `{CATALOG}`.`{SCHEMA}`.`{t}`").first()[0]
        cards.append(f"""<div style="border:1px solid #e0e0e0;border-radius:10px;padding:12px 14px;width:280px;{FONT}">
          <div style="font-size:11px;font-weight:700;color:{ROLE_COLOR.get(role, '#FF3621')};text-transform:uppercase;letter-spacing:.04em">{role}</div>
          <div style="font-size:15px;font-weight:700;margin:4px 0"><a href="{uc_url(CATALOG, SCHEMA, t)}" target="_blank" style="color:#1B3139">{t}</a></div>
          <div style="font-size:12px;color:#555;min-height:32px">{desc}</div>
          <div style="font-size:12px;margin-top:6px">{n:,} rows · <a href="{uc_url(CATALOG, SCHEMA, t, tab='sample')}" target="_blank">sample</a> ·
          <a href="{uc_url(CATALOG, SCHEMA, t, tab='lineage')}" target="_blank">lineage</a></div></div>""")
    displayHTML(f"""<div style="{FONT}">{f'<div style="font-size:16px;font-weight:700;margin-bottom:6px">{title}</div>' if title else ''}
      {button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139")}
      <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:10px">{''.join(cards)}</div></div>""")

def describe_tables():
    """(Re)applies business descriptions — CREATE OR REPLACE resets them, and Genie relies on them."""
    have = existing_tables()
    for t, (role, desc, _) in TABLE_INFO.items():
        if t not in have:
            continue
        try:
            spark.sql(f"COMMENT ON TABLE {fq(t)} IS '{_q(desc)}'")
        except Exception:
            continue   # e.g. a table managed by an AI Function task
        cols = None
        for (tt, col), cc in globals().get("COLUMN_COMMENTS", {}).items():
            if tt != t:
                continue
            cols = cols or [r.col_name for r in spark.sql(f"SHOW COLUMNS IN {fq(t)}").collect()]   # SQL, so VARIANT columns are fine
            if col in cols:
                spark.sql(f"ALTER TABLE {fq(t)} ALTER COLUMN `{col}` COMMENT '{_q(cc)}'")

# ---------------------------------------------------------------- Genie
def md_to_html(t):
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    out, in_list = [], False
    for line in t.splitlines():
        if re.match(r"^\s*[-*] ", line):
            if not in_list: out.append("<ul style='margin:4px 0 4px 18px;padding:0'>"); in_list = True
            out.append("<li>" + re.sub(r"^\s*[-*] ", "", line) + "</li>")
        else:
            if in_list: out.append("</ul>"); in_list = False
            out.append(line + "<br>")
    if in_list: out.append("</ul>")
    return "".join(out)

hexid = lambda: uuid.uuid4().hex
def genie_space(tables, functions):
    return {
        "version": 2,
        "config": {"sample_questions": sorted([{"id": hexid(), "question": [x]} for x in GENIE_QUESTIONS], key=lambda x: x["id"])},
        "data_sources": {"tables": [{"identifier": fq(t)} for t in sorted(tables)]},
        "instructions": {
            "text_instructions": [{"id": hexid(), "content": [GENIE_INSTRUCTIONS]}],
            "example_question_sqls": sorted([{"id": hexid(), "question": [q], "sql": [s]} for q, s in GENIE_EXAMPLES], key=lambda x: x["id"]),
            **({"sql_functions": sorted([{"id": hexid(), "identifier": fq(f)} for f in functions], key=lambda x: (x["id"], x["identifier"]))} if functions else {}),
        },
    }

def _genie_wanted():
    have = existing_tables()
    tables = sorted(t for t, (_, _, g) in TABLE_INFO.items() if g and t in have)
    funcs = [f for f in globals().get("GENIE_FUNCTIONS", [])
             if spark.sql(f"SHOW USER FUNCTIONS IN `{CATALOG}`.`{SCHEMA}` LIKE '{f}'").count() > 0]
    return tables, funcs

def create_or_sync_genie():
    """Creates (or reuses) the Genie agent and adds tables / functions that later steps have built."""
    global GENIE_SPACE_ID, GENIE_URL
    describe_tables()
    tables, funcs = _genie_wanted()
    existing = next((s for s in _api("GET", "/api/2.0/genie/spaces").get("spaces", []) if s.get("title") == GENIE_TITLE), None)
    if existing:
        GENIE_SPACE_ID = existing["space_id"]
        g = _api("GET", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}", query={"include_serialized_space": "true"})
        cur = json.loads(g["serialized_space"])
        have_t = sorted(x["identifier"] for x in cur["data_sources"].get("tables", []))
        have_f = sorted(x["identifier"] for x in cur.get("instructions", {}).get("sql_functions", []))
        have_i = [x["content"][0] for x in cur.get("instructions", {}).get("text_instructions", [])]
        have_q = sorted(x["question"][0] for x in cur.get("instructions", {}).get("example_question_sqls", []))
        if (have_t != [fq(t) for t in tables] or have_f != sorted(fq(f) for f in funcs) or have_i != [GENIE_INSTRUCTIONS]
                or have_q != sorted(q for q, _ in GENIE_EXAMPLES)):
            _api("PATCH", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}", body={"serialized_space": json.dumps(genie_space(tables, funcs))})
    else:
        GENIE_SPACE_ID = _api("POST", "/api/2.0/genie/spaces", body={
            "title": GENIE_TITLE, "description": GENIE_INSTRUCTIONS.splitlines()[0], "warehouse_id": WAREHOUSE.id,
            "parent_path": f"/Workspace/Users/{w.current_user.me().user_name}",
            "serialized_space": json.dumps(genie_space(tables, funcs))})["space_id"]
        print(f"Created Genie agent '{GENIE_TITLE}'")
    GENIE_URL = f"{HOST}/genie/rooms/{GENIE_SPACE_ID}{ORG}"

def ask_genie(question, timeout_s=300):
    """Asks the Genie agent a question and returns answer text, generated SQL, result DataFrame and chat link."""
    t0 = time.time()
    m = _api("POST", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation", body={"content": question})
    conv, mid = m["conversation_id"], m["message_id"]
    while True:
        msg = _api("GET", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conv}/messages/{mid}")
        if msg.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED") or time.time() - t0 > timeout_s:
            break
        time.sleep(3)
    res = {"question": question, "texts": [], "sql": "", "df": None, "followups": [], "status": msg.get("status"),
           "seconds": round(time.time() - t0), "url": f"{HOST}/genie/rooms/{GENIE_SPACE_ID}/chats/{conv}{ORG}"}
    for a in msg.get("attachments", []) or []:
        if a.get("text"):
            res["texts"].append(a["text"].get("content", ""))
        if a.get("query"):
            q = a["query"]
            if q.get("description"):
                res["texts"].append(q["description"])
            res["sql"] = q.get("query", "")
            try:
                r = _api("GET", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conv}/messages/{mid}/attachments/{a['attachment_id']}/query-result")
                sr = r.get("statement_response", {})
                cols = [c["name"] for c in sr.get("manifest", {}).get("schema", {}).get("columns", [])]
                rows = sr.get("result", {}).get("data_array", []) or []
                if cols and rows:
                    res["df"] = spark.createDataFrame(rows, schema=", ".join(f"`{c}` string" for c in cols))
            except Exception as e:
                res["texts"].append(f"(Could not fetch the result table: {e})")
        if a.get("suggested_questions"):
            res["followups"] += a["suggested_questions"].get("questions", [])
    if msg.get("status") == "FAILED" and not res["texts"]:
        res["texts"].append(str(msg.get("error", "Genie could not answer this question.")))
    return res

def show_genie_answer(res, before=None):
    """Renders a Genie answer. `before` = (step label, lines of hand-written SQL it replaces)."""
    sql = res["sql"] or ""
    sql_lines = len([l for l in sql.splitlines() if l.strip()])
    tables = sorted(set(re.findall(r"\b" + re.escape(SCHEMA) + r"\.`?(\w+)", sql))) if sql else []
    answer = md_to_html("\n".join(res["texts"])) or ("Genie ran the query below — its result is in the table under this card."
                                                      if res["sql"] else "Genie is still working on this one — open it in Genie to see the answer.")
    compare = ""
    if before:
        compare = f"""<div style="display:flex;gap:12px;margin:10px 0 12px">
          <div style="flex:1;background:#F9F7F4;border-radius:8px;padding:10px 12px">
            <div style="font-size:11px;font-weight:700;color:#5A6F77;text-transform:uppercase">Without Genie</div>
            <div style="font-size:14px;margin-top:4px">{html.escape(before[0])} needed <b>{before[1]} lines of hand-written code</b> — written and maintained by an analyst.</div></div>
          <div style="flex:1;background:#FFF1EE;border-radius:8px;padding:10px 12px">
            <div style="font-size:11px;font-weight:700;color:#FF3621;text-transform:uppercase">With Genie</div>
            <div style="font-size:14px;margin-top:4px">One plain-English question. Genie wrote <b>{sql_lines} line{"s" if sql_lines != 1 else ""} of SQL</b>{f" across {len(tables)} governed table" + ("s" if len(tables) != 1 else "") if tables else ""} in <b>{res['seconds']} seconds</b>.</div></div></div>"""
    sql_html = (f'<details style="margin-top:8px"><summary style="cursor:pointer;color:#1B3139;font-weight:600">Show the SQL Genie wrote</summary>'
                f'<pre style="background:#f6f6f6;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:12px">{html.escape(sql)}</pre></details>') if sql else ""
    fu = ("<div style='margin-top:10px;font-size:12px;color:#555'><b>Try next:</b> " + " · ".join(html.escape(f) for f in res["followups"][:3]) + "</div>") if res["followups"] else ""
    displayHTML(f"""<div style="{FONT};border:1px solid #e0e0e0;border-left:5px solid #FF3621;border-radius:10px;padding:16px;max-width:1100px">
      <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.04em">Business user asks Genie</div>
      <div style="font-size:18px;font-weight:700;margin:2px 0 4px">“{html.escape(res['question'])}”</div>
      {compare}
      <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.04em">Genie answers</div>
      <div style="font-size:14px;margin-top:4px;line-height:1.45">{answer}</div>
      {sql_html}{fu}
      <div style="margin-top:12px">{button("Continue this conversation in Genie", res["url"])}{button("Open the Genie agent", GENIE_URL, "#1B3139")}</div></div>""")
    if res["df"] is not None:
        display(res["df"])

def ask(question, before=None):
    """Ask the Genie agent a question in plain English and show the answer here."""
    create_or_sync_genie()
    show_genie_answer(ask_genie(question), before=before)

# ---------------------------------------------------------------- MLflow — the trust layer (AI agent & ML demo blueprint patterns)
import os
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = WAREHOUSE.id    # MLflow reads the UC trace tables through this warehouse

def experiment():
    """Opens this notebook's MLflow experiment. A new one is created with Unity Catalog trace storage (traces become UC tables
    you can query with SQL); an existing, active one is used as-is."""
    from mlflow.entities.trace_location import UnityCatalog
    mlflow.set_registry_uri("databricks-uc")
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp and exp.lifecycle_stage == "active":
        return mlflow.set_experiment(experiment_id=exp.experiment_id)
    if exp:
        raise Exception(f"An experiment named '{EXPERIMENT_NAME}' is in the trash — restore it in the Experiments UI or change EXPERIMENT_NAME in S.3.")
    return mlflow.set_experiment(experiment_name=EXPERIMENT_NAME, trace_location=UnityCatalog(catalog_name=CATALOG, schema_name=SCHEMA))

def show_run(run, label):
    url = f"{HOST}/ml/experiments/{run.info.experiment_id}/runs/{run.info.run_id}{ORG}"
    displayHTML(f'<div style="{FONT};font-size:13px">MLflow · {html.escape(label)}: <b>{html.escape(run.info.run_name)}</b> '
                f'{button("Open the run", url, "#1B3139")}</div>')

def eval_dataset(name, records):
    """A versioned MLflow evaluation dataset stored in Unity Catalog (shows in the experiment's Datasets tab).
    merge_records de-duplicates by inputs, so re-running only updates expectations."""
    from mlflow.genai.datasets import create_dataset, get_dataset
    experiment()
    try:
        ds = get_dataset(name=fq(name))
    except Exception:
        ds = create_dataset(name=fq(name))
    ds.merge_records(records)
    return ds

def set_alias(model_name, alias, version):
    mlflow.MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(model_name, alias, str(version))

# ---------------------------------------------------------------- Lakeflow Jobs with AI Function tasks
def show_task(task, ui_name, what):
    """Renders an AI Function task definition: what it does, its input and output, and the full JSON the Jobs API receives."""
    a = task["ai_function_task"]
    src = a["volume"]["volume_path"] if "volume" in a else f"{a['delta_table']['table_name']} · column {a['delta_table']['input_column']}"
    out = f"{a['output']['catalog']}.{a['output']['schema']}.{a['output']['table']}"
    displayHTML(f"""<div style="{FONT};border:1px solid #e0e0e0;border-left:5px solid #1B3139;border-radius:10px;padding:10px 14px;margin:6px 0;max-width:1100px">
      <div style="font-size:11px;font-weight:700;color:#5A6F77;text-transform:uppercase;letter-spacing:.04em">AI Function task · {ui_name}</div>
      <div style="font-size:15px;font-weight:700;margin:2px 0">{task['task_key']}</div><div style="font-size:13px">{html.escape(what)}</div>
      <div style="font-size:12px;color:#555;margin-top:4px"><b>Input</b> {html.escape(src)} &nbsp;→&nbsp; <b>Output</b> {out}</div>
      <details style="margin-top:6px"><summary style="cursor:pointer;font-size:12px;font-weight:600">Show the task JSON</summary>
      <pre style="background:#f6f6f6;padding:8px;border-radius:6px;white-space:pre-wrap;font-size:11px">{html.escape(json.dumps(task, indent=2))}</pre></details></div>""")

def create_or_update_job(settings):
    global JOB_URL
    jobs = _api("GET", "/api/2.2/jobs/list", query={"name": settings["name"]}).get("jobs", [])
    if jobs:
        job_id = jobs[0]["job_id"]
        _api("POST", "/api/2.2/jobs/reset", body={"job_id": job_id, "new_settings": settings})
    else:
        job_id = _api("POST", "/api/2.2/jobs/create", body=settings)["job_id"]
    JOB_URL = f"{HOST}/jobs/{job_id}{ORG}"
    return job_id

def start_job_run(job_id):
    run_id = _api("POST", "/api/2.2/jobs/run-now", body={"job_id": job_id})["run_id"]
    return _api("GET", "/api/2.2/jobs/runs/get", query={"run_id": run_id})

def wait_for_run(run_id, label):
    """Waits for a job run and shows every task's result and duration."""
    t0, last = time.time(), ""
    while True:
        r = _api("GET", "/api/2.2/jobs/runs/get", query={"run_id": run_id})
        if r["state"]["life_cycle_state"] in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        running = ", ".join(t["task_key"] for t in r.get("tasks", []) if t.get("state", {}).get("life_cycle_state") == "RUNNING") or r["state"]["life_cycle_state"].lower()
        if running != last:
            print(f"{round((time.time() - t0) / 60, 1):>5} min  running: {running}")
            last = running
        time.sleep(20)
    tasks_html = "".join(
        f"<tr><td style='padding:2px 10px'>{t['task_key']}</td><td style='padding:2px 10px'>{t['state'].get('result_state', '')}</td>"
        f"<td style='padding:2px 10px' align=right>{round((t.get('end_time', 0) - t.get('start_time', 0)) / 60000, 1)} min</td></tr>"
        for t in sorted(r.get("tasks", []), key=lambda t: t.get("start_time", 0)))
    displayHTML(f"""<div style="{FONT};font-size:14px">Run <b>{label}</b> finished: <b>{r['state'].get('result_state')}</b>
      ({round((r.get('end_time', 0) - r.get('start_time', 0)) / 60000, 1)} minutes in total)
      <table style="font-size:12px;margin:6px 0">{tasks_html}</table>{button("Open the job run (see the graph)", r["run_page_url"])}</div>""")
    assert r["state"].get("result_state") == "SUCCESS", f"Job run did not succeed — open {r['run_page_url']} and use Repair run"
    return r

def run_job(job_id, label):
    return wait_for_run(start_job_run(job_id)["run_id"], label)

# ---------------------------------------------------------------- AI/BI dashboard
_wid = lambda: uuid.uuid4().hex[:8]
def _pos(x, y, wd, h):
    return {"x": x, "y": y, "width": wd, "height": h}
def text_widget(md, x, y, wd, h):
    return {"widget": {"name": _wid(), "multilineTextboxSpec": {"lines": [md]}}, "position": _pos(x, y, wd, h)}
def counter_widget(ds, col, title, x, y, wd, h=3):
    return {"widget": {"name": _wid(), "queries": [{"name": "main_query", "query": {"datasetName": ds, "disaggregated": True,
            "fields": [{"name": col, "expression": f"`{col}`"}]}}],
            "spec": {"version": 2, "widgetType": "counter", "encodings": {"value": {"fieldName": col, "displayName": title}},
                     "frame": {"title": title, "showTitle": True}}}, "position": _pos(x, y, wd, h)}
def table_widget(ds, cols, title, x, y, wd, h):
    return {"widget": {"name": _wid(), "queries": [{"name": "main_query", "query": {"datasetName": ds, "disaggregated": True,
            "fields": [{"name": c, "expression": f"`{c}`"} for c in cols]}}],
            "spec": {"version": 2, "widgetType": "table", "encodings": {"columns": [{"fieldName": c, "displayName": c.replace("_", " ")} for c in cols]},
                     "frame": {"title": title, "showTitle": True}}}, "position": _pos(x, y, wd, h)}
def _chart(kind, ds, xcol, ycol, color, title, x, y, wd, h, xscale, horizontal=False):
    fields = [{"name": xcol, "expression": f"`{xcol}`"}, {"name": f"sum({ycol})", "expression": f"SUM(`{ycol}`)"}]
    if color:
        fields.append({"name": color, "expression": f"`{color}`"})
    cat_enc = {"fieldName": xcol, "scale": {"type": xscale}, "displayName": xcol.replace("_", " ")}
    val_enc = {"fieldName": f"sum({ycol})", "scale": {"type": "quantitative"}, "displayName": ycol.replace("_", " ")}
    enc = {"x": val_enc, "y": cat_enc} if horizontal else {"x": cat_enc, "y": val_enc}
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color.replace("_", " ")}
    return {"widget": {"name": _wid(), "queries": [{"name": "main_query", "query": {"datasetName": ds, "disaggregated": False, "fields": fields}}],
            "spec": {"version": 3, "widgetType": kind, "encodings": enc, "frame": {"title": title, "showTitle": True}}}, "position": _pos(x, y, wd, h)}
def bar_widget(ds, xcol, ycol, color, title, x, y, wd, h, horizontal=False):
    return _chart("bar", ds, xcol, ycol, color, title, x, y, wd, h, "categorical", horizontal)
def line_widget(ds, xcol, ycol, color, title, x, y, wd, h):
    return _chart("line", ds, xcol, ycol, color, title, x, y, wd, h, "temporal")

def publish_dashboard(name, datasets, widgets):
    """Creates (or updates) a one-page AI/BI dashboard with the Genie agent attached, publishes it and returns its URL."""
    spec = {"datasets": [{"name": k, "displayName": k, "queryLines": [v]} for k, v in datasets.items()],
            "pages": [{"name": "main", "displayName": name, "pageType": "PAGE_TYPE_CANVAS", "layout": widgets}],
            "uiSettings": {"genieSpace": {"isEnabled": True, "overrideId": GENIE_SPACE_ID, "enablementMode": "ENABLED"}}}
    body = {"display_name": name, "warehouse_id": WAREHOUSE.id, "serialized_dashboard": json.dumps(spec)}
    found = [d for d in _api("GET", "/api/2.0/lakeview/dashboards", query={"page_size": 200}).get("dashboards", [])
             if d.get("display_name") == name and d.get("lifecycle_state") != "TRASHED"]
    if found:
        did = found[0]["dashboard_id"]
        _api("PATCH", f"/api/2.0/lakeview/dashboards/{did}", body=body)
    else:
        did = _api("POST", "/api/2.0/lakeview/dashboards", body=dict(body, parent_path=NB_FOLDER))["dashboard_id"]
    _api("POST", f"/api/2.0/lakeview/dashboards/{did}/published", body={"embed_credentials": True, "warehouse_id": WAREHOUSE.id})
    return f"{HOST}/dashboardsv3/{did}/published{ORG}"
CRM_EXPORT = f"/Volumes/{CATALOG}/{SCHEMA}/crm_export"
JOB_NAME = f"Support ticket classification ({CATALOG}.{SCHEMA})"
DASHBOARD_NAME = f"Sales — account 360, leads & renewals ({SCHEMA})"
EXPERIMENT_NAME = f"{NB_FOLDER}/Sales — AI Functions & MLflow trust layer ({SCHEMA})"

# table -> (layer shown on the card, description, include in the Genie agent?)
TABLE_INFO = {
    "crm_accounts": ("Bronze · source", "Salesforce accounts (with duplicates and missing firmographics)", False),
    "crm_contacts": ("Bronze · source", "Salesforce contacts with email", False),
    "leads": ("Bronze · source", "Salesforce leads", False),
    "engagement": ("Bronze · source", "Marketing engagement per lead", False),
    "opportunities": ("Bronze · source", "Opportunities: converted / closed-won per lead", False),
    "billing": ("Bronze · source", "Billing system: monthly revenue, products, seats — its own account IDs and names", False),
    "product_events": ("Bronze · source", "Product telemetry: one row per user event", False),
    "support_tickets": ("Bronze · source", "Support desk tickets — only the requester's email, no account ID", False),
    "nps_surveys": ("Bronze · source", "NPS survey responses with free-text comments", False),
    "contract_renewals": ("Bronze · source", "Renewals: ARR and renewed ARR (NULL until the renewal date)", True),
    "ground_truth": ("Answer key", "True identity mappings, duplicates, ticket themes and planted signals", False),
    "match_pairs": ("ai_similarity · ai_classify", "Candidate record pairs scored with ai_similarity", False),
    "match_candidates": ("ai_similarity · ai_classify", "Candidate record pairs with similarity score and decision", False),
    "account_xref": ("Silver", "Every CRM, billing and support record mapped to one master account_id", True),
    "account_enrichment": ("ai_enrich", "Industry, segment and HQ region filled in, with rationale", False),
    "ticket_classifications": ("AI Function task", "classify_tickets task output: theme + confidence per ticket", False),
    "tickets_resolved": ("Silver", "Tickets with master account and ai_classify theme", True),
    "nps_sentiment": ("ai_analyze_sentiment", "NPS responses with sentiment", False),
    "account_weekly_usage": ("Silver", "Weekly active users per account", False),
    "customer_360": ("Gold", "One row per master account: revenue, usage, tickets, NPS, products and an ai_summarize brief", True),
    "lead_features": ("Feature table", "Governed feature table (primary key lead_id): engagement recency / frequency, firmographic fit, source", False),
    "lead_labels": ("Silver", "Lead labels kept apart from features: converted, closed_won and the training / hold-out / open split", False),
    "lead_scores": ("MLflow model", "Open leads: conversion probability, decile, top-3 SHAP reasons, ai_gen talking point", True),
    "lead_score_backtest": ("MLflow model", "Hold-out leads with score decile and outcome (win rate by decile)", True),
    "lead_outreach": ("ai_gen", "Drafted outreach emails for the top 20 leads", True),
    "crm_lead_writeback": ("Gold", "What gets written back to Salesforce (reverse ETL stub)", False),
    "renewal_features": ("Silver", "Renewal features as of 90 days before each renewal, with churn / expansion labels", False),
    "churn_predictions": ("ai_predict_class", "Churn prediction and confidence per renewal", False),
    "expansion_predictions": ("ai_predict_class", "Expansion prediction and confidence per renewal", False),
    "usage_anomalies": ("ai_detect_anomalies", "Weekly active users per account with anomaly flags", False),
    "arr_drivers": ("ai_top_drivers", "Which segments drive next quarter's ARR at risk", True),
    "account_actions": ("ai_gen", "Save / Expand / Nurture / Healthy bucket and next-best action per upcoming renewal", True),
}
ROLE_COLOR.update({"Feature table": "#7A3E9D", "Bronze · source": "#5A6F77", "Silver": "#2272B4", "Gold": "#00A972", "Answer key": "#98102A", "AI Function task": "#1B3139", "MLflow model": "#0B2026"})
print(f"Writing to {CATALOG}.{SCHEMA} · CRM export volume {CRM_EXPORT} · SQL warehouse for ai_detect_anomalies / Genie: '{WAREHOUSE.name}'")
displayHTML(button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139") + button("Open the crm_export volume", volume_url("crm_export"), "#5A6F77"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Generate synthetic data *(skip 0.1–0.4 if you bring your own)*
# MAGIC
# MAGIC **Halcyon Group** (fictitious) sells a data platform — *Platform, Data API, Analytics Pro, Managed Storage* — to about 500 B2B accounts. Step 0.1 writes what its go-to-market systems look like after a few years — **messy on purpose**:
# MAGIC
# MAGIC | Source | Table(s) | The mess |
# MAGIC |---|---|---|
# MAGIC | **Salesforce** (CSV export in the `crm_export` volume — loaded in step 1.1) | `crm_accounts`, `crm_contacts`, `leads`, `engagement`, `opportunities` | **15 duplicate accounts** (*"Acme Corp"* next to *"Acme Corporation"*); **20 accounts with no industry or segment** — real public company names, so `ai_enrich` can fill in true firmographics |
# MAGIC | **Billing** | `billing` | Its own IDs and names — *"ACME Inc."* in billing is *"Acme Corporation"* in CRM; long names are shortened; some customers have a separate EMEA billing account; the bill-to email is often at an outsourced payables provider, or missing |
# MAGIC | **Support desk** | `support_tickets` | Only the requester's **email** — no account ID. `ai_gen` writes each ticket description (step 0.2) |
# MAGIC | **Product** | `product_events` | ~2 million user events over 2½ years |
# MAGIC | **Surveys / renewals** | `nps_surveys`, `contract_renewals` | NPS comments written by `ai_gen`; renewed ARR is NULL until the renewal date |
# MAGIC
# MAGIC **Planted signals**, every one recorded in **`ground_truth`**: the **true CRM-to-billing and ticket-to-account mappings** and the **CRM duplicate pairs** (section 1); **accounts whose usage is falling while NPS still looks healthy** (section 3); and every past renewal outcome.
# MAGIC
# MAGIC > ⚠️ **Step 0.1 starts fresh:** it replaces the demo tables in the schema from S.2, empties the `crm_export` volume and deletes this notebook's job (if it exists). **Don't run it against a schema that holds your own data** — skip to step 0.11 instead.
# MAGIC
# MAGIC All randomness is seeded, so results are reproducible, and dates are relative to today, so the data always looks current.

# COMMAND ----------

# Start fresh: delete this notebook's job, its output table and the CRM export
for j in _api("GET", "/api/2.2/jobs/list", query={"name": JOB_NAME}).get("jobs", []):
    _api("POST", "/api/2.2/jobs/delete", body={"job_id": j["job_id"]})
spark.sql(f"DROP TABLE IF EXISTS {fq('ticket_classifications')}")
for f in dbutils.fs.ls(CRM_EXPORT):
    dbutils.fs.rm(f.path, True)

# Synthetic sales data for Halcyon Group (fictitious seller). Seeded, and dated relative to today.
import calendar, datetime as dt, re
import numpy as np, pandas as pd
from faker import Faker

rng = np.random.default_rng(11); fake = Faker("en_US"); Faker.seed(11)
TODAY = dt.date.today()
def month_end(y, m): return dt.date(y, m, calendar.monthrange(y, m)[1])
def add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    return dt.date(d.year + y, m + 1, min(d.day, calendar.monthrange(d.year + y, m + 1)[1]))
LME = TODAY.replace(day=1) - dt.timedelta(days=1)
MONTHS = [month_end(add_months(LME, -i).year, add_months(LME, -i).month) for i in range(29, -1, -1)]   # 30 months of billing
WEEK0 = TODAY - dt.timedelta(days=TODAY.weekday()) - dt.timedelta(weeks=130)                          # 130 weeks of telemetry
WEEKS = [WEEK0 + dt.timedelta(weeks=i) for i in range(130)]
PRODUCTS = ["Platform", "Data API", "Analytics Pro", "Managed Storage"]
INDUSTRIES = ["Financial Services", "Insurance", "Healthcare", "Life Sciences", "Manufacturing", "Retail & CPG", "Technology", "Energy & Utilities"]
SEGMENTS = ["Enterprise", "Mid-Market", "SMB"]
sig = lambda x: 1 / (1 + np.exp(-x))
slug = lambda s: re.sub(r"[^a-z0-9]", "", re.sub(r"\b(inc|llc|ltd|plc|group|corp|corporation|co|company|and|the)\b", "", s.lower()))

# ---------------------------------------------------------------- master accounts (the truth)
REAL = [  # real public companies, left without industry / segment in CRM so ai_enrich has real work
    ("JPMorgan Chase & Co.", "jpmorganchase.com", "Financial Services"), ("Capital One Financial", "capitalone.com", "Financial Services"),
    ("Prudential Financial", "prudential.com", "Insurance"), ("MetLife", "metlife.com", "Insurance"), ("The Travelers Companies", "travelers.com", "Insurance"),
    ("Humana", "humana.com", "Healthcare"), ("Cigna Group", "thecignagroup.com", "Healthcare"), ("HCA Healthcare", "hcahealthcare.com", "Healthcare"),
    ("Pfizer", "pfizer.com", "Life Sciences"), ("Moderna", "modernatx.com", "Life Sciences"), ("Regeneron Pharmaceuticals", "regeneron.com", "Life Sciences"),
    ("Medtronic", "medtronic.com", "Life Sciences"), ("Stryker", "stryker.com", "Life Sciences"), ("Caterpillar", "caterpillar.com", "Manufacturing"),
    ("Deere & Company", "deere.com", "Manufacturing"), ("Honeywell International", "honeywell.com", "Manufacturing"),
    ("Parker Hannifin", "parker.com", "Manufacturing"), ("Rockwell Automation", "rockwellautomation.com", "Manufacturing"),
    ("Kroger", "kroger.com", "Retail & CPG"), ("Duke Energy", "duke-energy.com", "Energy & Utilities"),
]
names = {"Acme Corporation"}
while len(names) < 480:
    names.add(fake.company().replace(",", ""))
owners = [fake.name() for _ in range(14)]
acc, used_domains = [], {d for _, d, _ in REAL}
for i, nm in enumerate(sorted(names)):
    seg = str(rng.choice(SEGMENTS, p=[0.22, 0.43, 0.35]))
    dom = f"{slug(nm)[:22]}.com"
    k = 1
    while dom in used_domains:          # two different companies never share a website
        k += 1; dom = f"{slug(nm)[:20]}{k}.com"
    used_domains.add(dom)
    acc.append(dict(name=nm, domain=dom, industry=str(rng.choice(INDUSTRIES)), segment=seg, real=False))
for nm, dom, ind in REAL:
    acc.append(dict(name=nm, domain=dom, industry=ind, segment="Enterprise", real=True))
rng.shuffle(acc)
for i, a in enumerate(acc):
    a["account_id"] = f"A-{10001 + i}"
    a["owner"] = owners[i % len(owners)]
    a["hq_region"] = str(rng.choice(["North America", "EMEA", "APAC"], p=[0.7, 0.2, 0.1]))
A = pd.DataFrame(acc).set_index("account_id")
ACME = A.index[A.name == "Acme Corporation"][0]

# ---------------------------------------------------------------- lifecycle: products, seats, ARR, renewals, usage trajectory
seat_rng = {"Enterprise": (50, 250), "Mid-Market": (15, 60), "SMB": (4, 15)}
arr_per_seat = {"Enterprise": (1500, 2600), "Mid-Market": (1300, 2200), "SMB": (1100, 1900)}
NPS_HEALTHY_DROP = []          # planted: usage falling while NPS still looks healthy
renewals, bill, weekly, gt = [], [], [], []
wk_idx = {d: i for i, d in enumerate(WEEKS)}
mo_of_week = lambda d: month_end(d.year, d.month)
state = {}
upcoming = [a for a in A.index]
for aid, a in A.iterrows():
    seg = a.segment
    since = TODAY - dt.timedelta(days=int(rng.integers(200, 1900)))
    primary = str(rng.choice(PRODUCTS, p=[0.4, 0.15, 0.3, 0.15]))
    prods = [primary] + [p for p in PRODUCTS if p != primary and rng.random() < 0.25]
    seats = int(rng.integers(*seat_rng[seg]))
    arr = round(seats * rng.uniform(*arr_per_seat[seg]) * (1 + 0.15 * (len(prods) - 1)), -2)
    util = rng.uniform(0.45, 0.9)
    state[aid] = dict(since=since, primary=primary, prods=prods, seats=seats, arr=arr, util=util, alive=True)

def renewal_dates(since):
    d, out = since, []
    while add_months(d, 12) < TODAY + dt.timedelta(days=200):
        d = add_months(d, 12)
        if d >= add_months(TODAY, -30):
            out.append(d)
    return out

mult = {aid: np.ones(130) for aid in A.index}         # weekly usage multiplier
risk_flags = {}                                       # aid -> list of (start_week, end_week, kind) for tickets / NPS
next_q_end = month_end(add_months(LME, 3).year, add_months(LME, 3).month)
picked_drop = 0
for aid, a in A.iterrows():
    s = state[aid]
    events = []
    for r in renewal_dates(s["since"]):
        if not s["alive"]:
            break
        future = r > TODAY
        seg_risk = 2.0 if (a.industry in ("Healthcare", "Life Sciences") and s["primary"] == "Analytics Pro" and future) else 0.0   # planted: new churn driver
        p_churn = sig(-2.4 + seg_risk + rng.normal(0, 0.6))
        churn = rng.random() < p_churn
        expand = (not churn) and rng.random() < sig(-1.2 + 1.5 * (s["util"] - 0.65))
        drop_case = (future and r <= add_months(TODAY, 6) and not churn and picked_drop < 10 and a.segment != "SMB" and rng.random() < 0.12)
        ri = (r - WEEK0).days // 7
        lo = max(0, ri - 26)
        if churn and rng.random() < 0.75:     # most churners' usage decays before the renewal (only what has happened by today shows)
            depth = rng.uniform(0.15, 0.45)
            for i in range(lo, min(ri, 130)):
                mult[aid][i] *= 1 - depth * (i - lo) / max(ri - lo, 1)
        elif not churn and not expand and rng.random() < 0.2:   # healthy accounts dip too (seasonality, reorgs)
            depth = rng.uniform(0.1, 0.3)
            for i in range(lo, min(ri, 130)):
                mult[aid][i] *= 1 - depth * (i - lo) / max(ri - lo, 1)
        elif expand:
            up = rng.uniform(0.1, 0.35)
            for i in range(lo, min(ri, 130)):
                mult[aid][i] *= 1 + up * (i - lo) / max(ri - lo, 1)
        if drop_case:
            picked_drop += 1
            NPS_HEALTHY_DROP.append(aid)
            for i in range(122, 130):
                mult[aid][i] *= 1 - 0.45 * (i - 121) / 8
        events.append(dict(r=r, churn=churn, expand=expand, future=future, drop=drop_case))
        arr_before = s["arr"]
        if not future:
            if churn:
                partial = rng.random() < 0.3
                renewed = round(arr_before * rng.uniform(0.5, 0.8), -2) if partial else 0.0
                if not partial:
                    s["alive"] = False
                    s["end"] = r
                    if ri < 130:
                        mult[aid][max(ri, 0):] = 0
                else:
                    s["seats"] = int(s["seats"] * renewed / arr_before); s["arr"] = renewed
            else:
                renewed = round(arr_before * rng.uniform(1.0, 1.06), -2)
                s["arr"] = renewed
            if expand:   # seats or a new product within 6 months after the renewal
                s.setdefault("expansions", []).append(add_months(r, int(rng.integers(1, 6))))
        else:
            renewed = None
        renewals.append(dict(account_id=aid, renewal_date=r, arr=float(arr_before), renewed_arr=None if renewed is None else float(renewed),
                             _churn=churn, _expand=expand, _drop=drop_case))
        risk_flags.setdefault(aid, []).append((max(lo, 0), min(ri, 130), "churn" if churn else "expand" if expand else "ok"))
for aid in NPS_HEALTHY_DROP:
    gt.append(dict(use_case="Churn and upsell", anomaly_type="usage_falling_nps_healthy", entity_type="account", entity_id=aid,
                   true_value="true", detail="Weekly active users down ~45% in the last 8 weeks; NPS still 9-10"))

# weekly active users + monthly billing (with expansions applied)
for aid, a in A.iterrows():
    s = state[aid]
    seats0 = s["seats"] if "end" in s else s["seats"]
    exp_dates = sorted(s.get("expansions", []))
    prods, seats = list(s["prods"]), int(s["seats"])
    for i, wk in enumerate(WEEKS):
        if wk < s["since"]:
            continue
        n_exp = sum(1 for d in exp_dates if d <= wk)
        seats_w = int(seats * (1 + 0.25 * n_exp))
        wau = max(0, int(round(seats_w * s["util"] * mult[aid][i] * rng.lognormal(0, 0.05))))
        if wau:
            weekly.append((aid, wk, wau))
    late_base = max(0, rng.normal(4, 5)) + (4 if any(k == "churn" for _, _, k in risk_flags.get(aid, [])) else 0)
    for m in MONTHS:
        if m < s["since"] or ("end" in s and m > s["end"]):
            continue
        n_exp = sum(1 for d in exp_dates if d <= m)
        p = prods + [x for x in PRODUCTS if x not in prods][:max(0, n_exp - 1)] if n_exp >= 2 else prods
        seats_m = int(seats * (1 + 0.25 * n_exp))
        bill.append(dict(account_id=aid, month=m, revenue=round(s["arr"] / 12 * (seats_m / max(seats, 1)), 2), products=list(p),
                         seats_purchased=seats_m, days_late=int(max(0, rng.normal(late_base, 4)))))

# ---------------------------------------------------------------- CRM (accounts with duplicates & gaps), contacts, billing names
crm = []
for aid, a in A.iterrows():
    crm.append(dict(account_id=aid, name=a["name"], website_domain=a.domain, industry=None if a.real else a.industry,
                    segment=None if a.real else a.segment, owner=a.owner))
VARIANTS = [lambda n: n.upper(), lambda n: re.sub(r"\b(Inc|LLC|Ltd|PLC|Group)\b\.?", "Corp", n), lambda n: n.replace(" and ", " & "),
            lambda n: n + " (EMEA)", lambda n: re.sub(r"\s+(Inc|LLC|Ltd|PLC|Group|Corporation)\.?$", "", n) + " Inc.",
            lambda n: n.replace(",", "").replace("-", " "), lambda n: n.split(" ")[0] + " " + " ".join(w[0] + "." for w in n.split(" ")[1:])]
dups = [x for x in A.index if x != ACME and not A.loc[x, "real"]]
dups = list(rng.choice(dups, 14, replace=False)) + [ACME]
for k, aid in enumerate(dups):
    a = A.loc[aid]
    dn = "Acme Corp" if aid == ACME else VARIANTS[k % len(VARIANTS)](a["name"])
    did = f"A-{19001 + k}"
    crm.append(dict(account_id=did, name=dn, website_domain=a.domain if k % 2 == 0 else None, industry=a.industry if k % 3 else None,
                    segment=a.segment, owner=owners[(k + 3) % len(owners)]))
    gt.append(dict(use_case="Customer 360", anomaly_type="crm_duplicate", entity_type="crm_account", entity_id=did, true_value=aid,
                   detail=f"Duplicate CRM record of {a['name']}"))
crm_accounts = pd.DataFrame(crm).sample(frac=1, random_state=3).reset_index(drop=True)

contacts, cno, second = [], 0, {}
TITLES = ["CFO", "VP Data", "Director of Analytics", "Data Engineering Manager", "CIO", "Head of BI", "Procurement Manager", "Data Scientist", "IT Director"]
for aid, a in A.iterrows():
    if rng.random() < 0.15:
        second[aid] = re.sub(r"\.com$", "", a.domain) + str(rng.choice(["-group.com", "health.com", "corp.net", "-eu.com"]))
    for j in range(int(rng.integers(2, 7))):
        cno += 1
        dom = second[aid] if aid in second and j == 0 else a.domain
        fn, ln = fake.first_name(), fake.last_name()
        contacts.append(dict(contact_id=f"CT-{cno:06d}", account_id=aid, email=f"{fn[0].lower()}{ln.lower()}@{dom}", title=str(rng.choice(TITLES))))
crm_contacts = pd.DataFrame(contacts)

import hashlib
_h = lambda *k: int(hashlib.md5("|".join(map(str, k)).encode()).hexdigest(), 16) % 100   # deterministic, doesn't touch rng
def BILLING_ABBREV(n):
    """Billing systems shorten long names but keep the distinctive part: 'Davis Lopez and Hill' -> 'Davis Lopez & H. Inc.'"""
    w = n.replace(",", "").split(" ")
    if len(w) <= 2:
        return n                                   # short names are left as they are
    return " ".join(w[:2]) + " " + " ".join("&" if x == "and" else x[0] + "." for x in w[2:])
def bill_to_email(aid, bid):
    a, h = A.loc[aid], _h(bid, "billto")
    if h < 55:   return f"{['ap', 'accounts.payable', 'invoices', 'finance'][h % 4]}@{a.domain}"                       # the customer's own domain
    if h < 70 and aid in second: return f"ap@{second[aid]}"                                                              # a subsidiary domain
    if h < 88:   return f"{slug(a['name'])[:12]}@{['payables-hub.com', 'apservices.net', 'invoicecloud.io'][h % 3]}"   # outsourced AP
    return None                                                                                                          # never captured
billing_names, bno = {}, 0
billing = []
for aid, a in A.iterrows():
    splits = ["main", "emea"] if (a.segment == "Enterprise" and rng.random() < 0.25) else ["main"]
    for sp in splits:
        bno += 1
        bid = f"B-{70001 + bno}"
        if aid == ACME:
            bname = "ACME Inc." if sp == "main" else "Acme Corp (EMEA)"
        else:
            if rng.random() < 0.7:          # same random draws, in the same order, as before
                vi = int(rng.integers(0, len(VARIANTS)))
                bname = BILLING_ABBREV(a["name"]) if vi == len(VARIANTS) - 1 else VARIANTS[vi](a["name"])
            else:
                bname = a["name"]
            if sp == "emea":
                bname = re.sub(r"\s*\(EMEA\)", "", bname) + " (EMEA)"
        billing_names[(aid, sp)] = (bid, bname, bill_to_email(aid, bid))
        gt.append(dict(use_case="Customer 360", anomaly_type="billing_mapping", entity_type="billing_account", entity_id=bid, true_value=aid,
                       detail=f"'{bname}' is {a['name']}"))
for b in bill:
    keys = [k for k in billing_names if k[0] == b["account_id"]]
    for k in keys:
        share = 1.0 if len(keys) == 1 else (0.7 if k[1] == "main" else 0.3)
        bid, bname, bemail = billing_names[k]
        billing.append(dict(billing_account_id=bid, billing_account_name=bname, bill_to_email=bemail, month=b["month"], revenue=round(b["revenue"] * share, 2),
                            products=b["products"], seats_purchased=int(round(b["seats_purchased"] * share)), days_late=b["days_late"]))
billing = pd.DataFrame(billing)

# ---------------------------------------------------------------- support tickets (descriptions are written by ai_gen in step 0.3)
THEMES = ["billing", "outage", "feature_request", "onboarding"]
emails_by_acc = crm_contacts.groupby("account_id").email.apply(list).to_dict()
tickets, tno = [], 0
for aid, a in A.iterrows():
    s = state[aid]
    base = {"Enterprise": 0.9, "Mid-Market": 0.45, "SMB": 0.18}[a.segment]
    for m in MONTHS:
        if m < s["since"] or ("end" in s and m > s["end"]):
            continue
        wi = min(129, max(0, (m - WEEK0).days // 7))
        flags = [k for lo, hi, k in risk_flags.get(aid, []) if lo <= wi <= hi]
        churny = "churn" in flags
        n = rng.poisson(base * (1.6 if churny else 1.0) * 0.2)
        for _ in range(n):
            tno += 1
            months_in = (m - s["since"]).days / 30
            p = np.array([0.22, 0.2, 0.38, 0.2]) if not churny else np.array([0.3, 0.35, 0.22, 0.13])
            if months_in < 4:
                p = np.array([0.15, 0.1, 0.15, 0.6])
            theme = str(rng.choice(THEMES, p=p / p.sum()))
            sev = str(rng.choice(["Sev 1", "Sev 2", "Sev 3"], p=[0.45, 0.4, 0.15] if theme == "outage" else [0.05, 0.35, 0.6]))
            opened = dt.datetime.combine(m.replace(day=1), dt.time(8)) + dt.timedelta(days=int(rng.integers(0, 27)), minutes=int(rng.integers(0, 600)))
            closed = opened + dt.timedelta(hours=float(rng.gamma(2, {"Sev 1": 6, "Sev 2": 20, "Sev 3": 40}[sev])))
            r = rng.random()
            if r < 0.68:
                email = str(rng.choice(emails_by_acc[aid]))
            elif r < 0.9:
                email = f"{fake.user_name()}@{a.domain}"
            elif r < 0.97 and aid in second:
                email = f"{fake.user_name()}@{second[aid]}"
            else:
                email = f"{fake.user_name()}@gmail.com"
            tid = f"TCK-{100000 + tno}"
            tickets.append(dict(ticket_id=tid, requester_email=email, severity=sev, opened=opened, closed=closed if closed < dt.datetime.now() else None,
                                _theme=theme, _product=s["primary"], _company=a["name"], _churny=churny))
            gt.append(dict(use_case="Customer 360", anomaly_type="ticket_mapping", entity_type="ticket", entity_id=tid,
                           true_value=aid if not email.endswith("@gmail.com") else None, detail=theme))
tickets = pd.DataFrame(tickets)
support_tickets = tickets[["ticket_id", "requester_email", "severity", "opened", "closed"]].assign(description="")
ticket_prompts = tickets[["ticket_id", "_theme", "_product", "severity", "_churny"]].rename(columns=lambda c: c.strip("_"))

# ---------------------------------------------------------------- NPS (scores now; comments from an ai_gen-written pool in step 0.3)
nps = []
for aid, a in A.iterrows():
    s = state[aid]
    d = TODAY - dt.timedelta(days=int(rng.integers(5, 60)))
    while d > max(s["since"], WEEK0):
        if not ("end" in s and d > s["end"]) and rng.random() < 0.6:
            wi = min(129, max(0, (d - WEEK0).days // 7))
            flags = [k for lo, hi, k in risk_flags.get(aid, []) if lo <= wi <= hi]
            mu = 8.2 if "expand" in flags else 6.3 if "churn" in flags else 7.5
            if aid in NPS_HEALTHY_DROP and d > TODAY - dt.timedelta(days=120):
                mu = 9.6
            score = int(np.clip(round(rng.normal(mu, 1.4)), 0, 10))
            topic = str(rng.choice(["performance", "support", "pricing", "ease_of_use", "reliability", "roadmap"]))
            nps.append(dict(account_id=aid, date=d, score=score, _bucket="promoter" if score >= 9 else "passive" if score >= 7 else "detractor", _topic=topic))
        d -= dt.timedelta(days=int(rng.integers(80, 100)))
nps = pd.DataFrame(nps)

# ---------------------------------------------------------------- leads, engagement, opportunities
SOURCES = {"Referral": 0.9, "Webinar": 0.4, "Trade Show": 0.3, "Content Download": 0.0, "Paid Search": -0.2, "Purchased List": -1.1}
SIZES = {"1-50": -0.9, "51-200": -0.2, "201-1000": 0.3, "1001-5000": 0.5, "5000+": 0.35}
FIT = {"Financial Services": 0.5, "Insurance": 0.35, "Healthcare": 0.45, "Life Sciences": 0.4, "Manufacturing": 0.45,
       "Retail & CPG": -0.1, "Technology": 0.0, "Energy & Utilities": -0.2}
leads, eng, opps = [], [], []
for i in range(6000):
    lid = f"L-{300001 + i}"
    created = TODAY - dt.timedelta(days=int(rng.integers(1, 730)))
    size, ind, src = str(rng.choice(list(SIZES))), str(rng.choice(INDUSTRIES)), str(rng.choice(list(SOURCES), p=[0.08, 0.17, 0.1, 0.25, 0.25, 0.15]))
    opens, visits = int(rng.poisson(4 + (3 if src in ("Webinar", "Referral") else 0))), int(rng.poisson(3))
    demo, event = bool(rng.random() < 0.16), bool(rng.random() < 0.14)
    last = created + dt.timedelta(days=int(rng.integers(0, 40)))
    last = min(last, TODAY)
    logit = -2.9 + 1.7 * demo + 0.09 * visits + 0.05 * opens + 0.6 * event + SIZES[size] + FIT[ind] + SOURCES[src] - 0.012 * (last - created).days
    leads.append(dict(lead_id=lid, company_size=size, industry=ind, source=src, created_date=created))
    eng.append(dict(lead_id=lid, email_opens=opens, web_visits=visits, demo_requested=demo, event_attended=event, last_activity_date=last))
    if (TODAY - created).days > 45:
        conv = bool(rng.random() < sig(logit))
        won = bool(conv and rng.random() < 0.38)
        opps.append(dict(lead_id=lid, converted=conv, closed_won=won,
                         amount=float(round(rng.uniform(15000, 250000) * (2 if size in ("1001-5000", "5000+") else 1), -2)) if conv else None))
leads, engagement, opportunities = pd.DataFrame(leads), pd.DataFrame(eng), pd.DataFrame(opps)

# ---------------------------------------------------------------- outputs
crm_accounts = crm_accounts[["account_id", "name", "website_domain", "industry", "segment", "owner"]]
product_weekly = pd.DataFrame(weekly, columns=["account_id", "week_start", "active_users"])
contract_renewals = pd.DataFrame(renewals)[["account_id", "renewal_date", "arr", "renewed_arr"]]
for r in renewals:
    if r["renewal_date"] <= TODAY:
        gt.append(dict(use_case="Churn and upsell", anomaly_type="renewal_outcome", entity_type="account", entity_id=r["account_id"],
                       true_value=f"churn={r['_churn']},expand={r['_expand']}", detail=str(r["renewal_date"])))
for aid in A.index[A.real]:
    gt.append(dict(use_case="Customer 360", anomaly_type="true_firmographics", entity_type="account", entity_id=aid,
                   true_value=A.loc[aid, "industry"], detail="Enterprise"))
ground_truth = pd.DataFrame(gt)
for c in ["true_value", "detail"]:
    ground_truth[c] = ground_truth[c].astype(object).where(ground_truth[c].notna(), None)
print(f"{len(A)} accounts (+{len(dups)} CRM duplicates) · {billing.billing_account_id.nunique()} billing accounts · {len(crm_contacts):,} contacts · "
      f"{len(support_tickets):,} tickets · {len(nps):,} NPS responses · {len(leads):,} leads · {len(contract_renewals):,} renewals · "
      f"{product_weekly.active_users.sum():,} weekly-active-user events to expand")

# Salesforce objects land as CSV files in the crm_export volume (step 1.1 ingests them)
for pdf, name in [(crm_accounts, "Account"), (crm_contacts, "Contact"), (leads, "Lead"), (engagement, "Lead_Engagement__c"), (opportunities, "Opportunity")]:
    pdf.to_csv(f"{CRM_EXPORT}/{name}.csv", index=False)

def save(pdf, name, props=""):
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq(name))
    if props:
        spark.sql(f"ALTER TABLE {fq(name)} SET TBLPROPERTIES ({props})")
save(billing, "billing")
save(support_tickets, "support_tickets", "delta.enableChangeDataFeed = true")   # the classify task reads only new / changed rows
save(nps.drop(columns=["_bucket", "_topic"]).assign(comment=""), "nps_surveys")
save(contract_renewals, "contract_renewals")
save(ground_truth, "ground_truth")
spark.createDataFrame(ticket_prompts).createOrReplaceTempView("ticket_briefs")         # hidden inputs for ai_gen in step 0.2
spark.createDataFrame(nps[["account_id", "date", "_bucket", "_topic"]].rename(columns=lambda c: c.strip("_"))).createOrReplaceTempView("nps_briefs")
# Product telemetry: expand weekly active users into one event row per active user per week
spark.createDataFrame(product_weekly).createOrReplaceTempView("weekly_active")
spark.sql(f"""
CREATE OR REPLACE TABLE {fq('product_events')} AS
SELECT account_id, concat(account_id, '-u', lpad(CAST(u AS STRING), 4, '0')) AS user_id,
       element_at(array('login', 'query_run', 'dashboard_view', 'api_call', 'notebook_run', 'model_train'),
                  CAST(pmod(xxhash64(account_id, week_start, u), 6) AS INT) + 1) AS event,
       CAST(week_start AS TIMESTAMP) + make_interval(0, 0, 0, CAST(pmod(xxhash64(u, week_start), 5) AS INT), CAST(pmod(xxhash64(week_start, u, 'h'), 10) AS INT) + 8,
                                                   CAST(pmod(xxhash64(u, account_id), 60) AS INT), 0) AS timestamp
FROM weekly_active LATERAL VIEW explode(sequence(1, active_users)) e AS u
""")
displayHTML(f'<div style="{FONT};font-size:14px">✓ Salesforce export written to <b>{CRM_EXPORT}</b> and 6 source tables to <b>{CATALOG}.{SCHEMA}</b>. '
            f'{button("Browse the CRM export", volume_url("crm_export"), "#5A6F77")}</div>')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- One description per ticket, written by ai_gen from its (hidden) theme, product and severity
# MAGIC MERGE INTO support_tickets t
# MAGIC USING (
# MAGIC   SELECT ticket_id, ai_gen(concat(
# MAGIC     'Write the body of a customer support ticket (2-3 sentences, first person, plain text, no greeting, no signature, no ticket number) ',
# MAGIC     'for a B2B data platform product called "', product, '". Severity: ', severity, '. The ticket is about ',
# MAGIC     CASE theme WHEN 'billing' THEN 'an invoice, a charge, seats billed or a renewal quote that looks wrong'
# MAGIC                WHEN 'outage' THEN 'the service being down, very slow or failing jobs'
# MAGIC                WHEN 'feature_request' THEN 'a capability they wish the product had'
# MAGIC                ELSE 'getting started: setup, access, SSO, training or migrating their first workloads' END,
# MAGIC     CASE WHEN churny THEN '. The customer sounds frustrated and mentions evaluating alternatives.' ELSE '.' END)) AS description
# MAGIC   FROM ticket_briefs
# MAGIC ) s ON t.ticket_id = s.ticket_id
# MAGIC WHEN MATCHED THEN UPDATE SET description = s.description;
# MAGIC
# MAGIC -- NPS comments: ai_gen writes a pool of comments per score bucket and topic; each response draws one
# MAGIC CREATE OR REPLACE TABLE nps_comment_pool COMMENT 'NPS comment pool written by ai_gen' AS
# MAGIC SELECT bucket, topic, v,
# MAGIC        ai_gen(concat('Write one NPS survey comment (1-2 sentences, first person, no quotes) from a ', bucket,
# MAGIC                      ' (score ', CASE bucket WHEN 'promoter' THEN '9-10' WHEN 'passive' THEN '7-8' ELSE '0-6' END,
# MAGIC                      ') customer of a B2B data platform, mainly about ', replace(topic, '_', ' '), '. Variation #', v, '.')) AS comment
# MAGIC FROM (SELECT explode(array('promoter', 'passive', 'detractor')) AS bucket)
# MAGIC CROSS JOIN (SELECT explode(array('performance', 'support', 'pricing', 'ease_of_use', 'reliability', 'roadmap')) AS topic)
# MAGIC CROSS JOIN (SELECT explode(sequence(1, 6)) AS v);
# MAGIC
# MAGIC MERGE INTO nps_surveys n
# MAGIC USING (SELECT b.account_id, b.date, p.comment FROM nps_briefs b
# MAGIC        JOIN nps_comment_pool p ON p.bucket = b.bucket AND p.topic = b.topic AND p.v = CAST(pmod(xxhash64(b.account_id, b.date), 6) AS INT) + 1) s
# MAGIC ON n.account_id = s.account_id AND n.date = s.date
# MAGIC WHEN MATCHED THEN UPDATE SET comment = s.comment;
# MAGIC
# MAGIC SELECT t.ticket_id, t.severity, t.requester_email, t.description FROM support_tickets t ORDER BY t.opened DESC LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'billing' AS table_name, count(*) AS row_count FROM billing UNION ALL
# MAGIC SELECT 'support_tickets', count(*) FROM support_tickets UNION ALL
# MAGIC SELECT 'product_events', count(*) FROM product_events UNION ALL
# MAGIC SELECT 'nps_surveys', count(*) FROM nps_surveys UNION ALL
# MAGIC SELECT 'contract_renewals', count(*) FROM contract_renewals UNION ALL
# MAGIC SELECT 'ground_truth', count(*) FROM ground_truth;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT use_case, anomaly_type, count(*) AS records, any_value(detail) AS example
# MAGIC FROM ground_truth GROUP BY ALL ORDER BY use_case, anomaly_type;

# COMMAND ----------

COLUMN_COMMENTS = {
    ("billing", "billing_account_id"): "Billing system's own account ID (not the CRM account_id)", ("billing", "month"): "Month-end date",
    ("billing", "products"): "Products billed that month", ("billing", "bill_to_email"): "Where invoices are sent; often the customer's own domain, sometimes an outsourced payables provider; can be missing", ("billing", "days_late"): "Days the invoice was paid after its due date",
    ("support_tickets", "requester_email"): "Who opened the ticket — the only link to an account", ("contract_renewals", "renewed_arr"): "NULL until the renewal date; 0 = churned",
    ("account_xref", "match_method"): "self, rule (deterministic domain match), ai_similarity (auto-match) or ai_classify (borderline pair)",
    ("account_xref", "master_account_id"): "The surviving CRM account_id", ("customer_360", "seat_utilization"): "Active users (last 4 weeks) / seats purchased",
    ("customer_360", "revenue_trend_pct"): "Billing revenue, last 3 months vs the 3 months before (%)", ("customer_360", "usage_trend_pct"): "Weekly active users, last 8 weeks vs the 8 before (%)",
    ("customer_360", "account_brief"): "ai_summarize brief of everything known about the account",
    ("lead_scores", "conversion_probability"): "MLflow model probability (0-1) that the lead converts to an opportunity",
    ("lead_scores", "score_decile"): "1 = top 10% of open leads", ("lead_scores", "top_reasons"): "Top 3 SHAP reasons for the score",
    ("account_actions", "bucket"): "Save, Expand, Nurture or Healthy (deterministic SQL CASE)", ("account_actions", "churn_probability"): "ai_predict_class probability of churn at the next renewal",
    ("arr_drivers", "change_usd"): "ARR change between the control group (churned last quarter) and the test group (next quarter's renewals at churn risk, probability >= 10%)",
}
describe_tables()
print("✓ Comments applied")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.6 Your source data in Unity Catalog
# MAGIC Every table — and the folder of CSV files — is governed by **Unity Catalog**: permissions, lineage and audit come for free. Click a card to see the schema, sample data or lineage.

# COMMAND ----------

show_uc_cards(roles={"Bronze · source", "Answer key"}, volume="crm_export")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.7 Start classifying the support tickets — in the background
# MAGIC Tickets arrive all day, every day, so sorting them into **billing, outage, feature request or onboarding** is a **recurring pipeline** — a native **`ai_classify` AI Function task** (*Classification* in the Jobs UI) in a Lakeflow Job. It's **incremental**: the first run classifies every ticket (the backfill), and each scheduled run after that only **new or changed** tickets (step 1.8). A brand-new AI Function task has a **one-time start-up of about 15–20 minutes** on its first run, so step 0.7 creates the job and **starts it now, without waiting**. You'll use its output in step 1.4.

# COMMAND ----------

TICKET_LABELS = {
    "billing": "Invoices, charges, seats billed, renewal quotes, refunds or payment questions",
    "outage": "The service is down, slow, failing or returning errors",
    "feature_request": "The customer asks for a capability or improvement the product doesn't have",
    "onboarding": "Getting started: setup, access, SSO, training, migration of first workloads",
}
classify_task = {
    "task_key": "classify_tickets",
    "ai_function_task": {
        "delta_table": {"table_name": fq("support_tickets"), "input_column": "description", "merge_key_columns": ["ticket_id"]},
        "output": {"catalog": CATALOG, "schema": SCHEMA, "table": "ticket_classifications"},
        "ai_classify": {"version": "2.1", "enable_confidence_scores": True,
                        "instructions": "Support tickets from B2B customers of a data platform company.",
                        "inline_labels": {"labels": [{"name": k, "description": v} for k, v in TICKET_LABELS.items()]}},
    },
    "max_retries": 1, "min_retry_interval_millis": 60_000, "timeout_seconds": 7200,
}
JOB_ID = create_or_update_job({"name": JOB_NAME, "max_concurrent_runs": 1, "queue": {"enabled": True}, "tags": {"industry_play": "sales-c360"},
                               "schedule": {"quartz_cron_expression": "0 0 * * * ?", "timezone_id": "America/New_York", "pause_status": "PAUSED"},
                               "tasks": [classify_task]})
BACKFILL = start_job_run(JOB_ID)
displayHTML(f"""<div style="{FONT};font-size:14px">Ticket classification started in the background (AI Function task, serverless, hourly schedule <b>paused</b>) — continue with step 0.8.
  {button("Watch the backfill run", BACKFILL["run_page_url"])}{button("Open the job", JOB_URL, "#1B3139")}</div>""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.8 Meet your Genie agent
# MAGIC Each section ends with **💬 Ask Genie**: the same insight, asked in plain English instead of SQL. That's how reps, sales managers and customer success will use these results — no SQL, no waiting on RevOps, and every answer respects **Unity Catalog** permissions. As later steps build tables, the agent picks them up automatically before each question.

# COMMAND ----------

GENIE_TITLE = f"Sales Genie — accounts, leads & renewals ({SCHEMA})"
GENIE_INSTRUCTIONS = f"""This agent answers account, lead and renewal questions for Halcyon Group's sales team.
- Accounts: {fq('customer_360')} has ONE row per master account (account_name, owner, segment, industry, revenue, usage, seats, tickets, NPS, products owned / not owned, account_brief). Match account names with ILIKE '%name%' — "Acme" means account_name ILIKE '%acme%'.
- Other names for an account (billing, CRM duplicates, support): {fq('account_xref')} (source_system, source_id, source_name, master_account_id).
- Tickets: {fq('tickets_resolved')} (theme from ai_classify, master_account_id).
- "Grew usage but not spend" = usage_trend_pct > 10 AND revenue_trend_pct <= 0 in {fq('customer_360')}.
- Leads: {fq('lead_scores')} (open leads; score_decile 1 = best; conversion_probability; top_reasons; talking_point). "This week" = all open leads ranked by conversion_probability. Win rate by decile: {fq('lead_score_backtest')} (closed_won / leads per score_decile).
- Renewals: {fq('account_actions')} (renewal_date, arr, churn_probability, expansion_probability, bucket Save / Expand / Nurture / Healthy, next_best_action). ARR at risk = sum(arr) WHERE bucket = 'Save'. Next quarter = renewal_date in the next calendar quarter.
- Why ARR at risk is going up: {fq('arr_drivers')} (ai_top_drivers: contributor, change_usd). arr_at_risk_next_quarter = ARR on next quarter's renewals predicted to churn (test); churned_arr_last_quarter = ARR that actually churned last quarter (control) — not ARR at risk.
- Round money to whole dollars and state the filters used."""
GENIE_EXAMPLES = [
    ("Show me everything about Acme",
     f"SELECT * FROM {fq('customer_360')} WHERE account_name ILIKE '%acme%'"),
    ("How much ARR is at risk next quarter?",
     f"SELECT round(sum(arr)) AS arr_at_risk, count(*) AS accounts\nFROM {fq('account_actions')}\n"
     f"WHERE bucket = 'Save' AND renewal_date BETWEEN date_trunc('QUARTER', add_months(current_date(), 3)) AND date_sub(add_months(date_trunc('QUARTER', add_months(current_date(), 3)), 3), 1)"),
]
create_or_sync_genie()
displayHTML(button("Open the Genie agent", GENIE_URL) + button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.9 💬 Ask Genie — explore the source data
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("How much ARR renews in each of the next three quarters?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.10 🛠 Try it yourself
# MAGIC Open the **`billing`** card from **step 0.6** → **Sample data**. Pick one `billing_account_name` and try to find the same company in the `Account.csv` file in the `crm_export` volume by eye. That's the problem section 1 solves for every record.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.11 Bring your own data
# MAGIC Everything after section 0 starts from these tables (the `ground_truth` answer key is only used by the MLflow checks in steps 1.7 and 3.7). To run on real data, skip steps 0.1–0.4 and point the names at your own tables with views — except `support_tickets`, which must be a **table with change data feed on**, because the classification task reads it incrementally (and step 1.8 appends to it). Minimum columns:
# MAGIC
# MAGIC | Table | Required columns | Typical source |
# MAGIC |---|---|---|
# MAGIC | `crm_accounts`, `crm_contacts` | `account_id, name, website_domain, industry, segment, owner` / `contact_id, account_id, email, title` | Salesforce via **Lakeflow Connect** (or a CSV export like the one in step 1.1) |
# MAGIC | `leads`, `engagement`, `opportunities` | `lead_id, company_size, industry, source, created_date` / `lead_id, email_opens, web_visits, demo_requested, event_attended, last_activity_date` / `lead_id, converted, closed_won, amount` | Salesforce, marketing automation (Marketo, HubSpot) |
# MAGIC | `billing` | `billing_account_id, billing_account_name, bill_to_email, month, revenue, products, seats_purchased, days_late` | Zuora, Stripe, NetSuite, SAP |
# MAGIC | `support_tickets` | `ticket_id, requester_email, severity, opened, closed, description` (change data feed on) | Zendesk, ServiceNow, Salesforce Service Cloud |
# MAGIC | `product_events`, `nps_surveys`, `contract_renewals` | `account_id, user_id, event, timestamp` / `account_id, date, score, comment` / `account_id, renewal_date, arr, renewed_arr` | Product analytics (Segment, Amplitude), Qualtrics / Medallia, CPQ |
# MAGIC
# MAGIC ```sql
# MAGIC -- Example: point the notebook at your Zendesk tickets
# MAGIC CREATE OR REPLACE VIEW support_tickets AS
# MAGIC SELECT CAST(id AS STRING) AS ticket_id, requester.email AS requester_email, priority AS severity,
# MAGIC        created_at AS opened, solved_at AS closed, description
# MAGIC FROM my_catalog.zendesk.tickets;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Customer 360 — what do we know about this account?
# MAGIC
# MAGIC | Step | What happens | Capability |
# MAGIC |---|---|---|
# MAGIC | 1.1 | Ingest the Salesforce export | **Lakeflow Connect for Salesforce** if a live org is connected; here `read_files` on the CSV export |
# MAGIC | 1.2 | **Resolve identities** — map every billing account, support ticket and duplicate CRM record to one master `account_id` | deterministic SQL · `ai_similarity` · `ai_classify` → `account_xref` |
# MAGIC | 1.3 | **Fill in firmographics** for accounts missing industry or segment | `ai_enrich` (with a rationale per value) |
# MAGIC | 1.4 | **Turn text into signals** — ticket themes and NPS sentiment | `ai_classify` **AI Function task** · `ai_analyze_sentiment` |
# MAGIC | 1.5–1.6 | Build the gold **`customer_360`** with an account brief | SQL · `ai_summarize` |
# MAGIC | 1.7 | **How good is the matching?** | **MLflow** `mlflow.genai.evaluate`: precision / recall by match method |
# MAGIC | 1.8 | Tomorrow — only the new tickets | the same job, scheduled |

# COMMAND ----------

# With a live Salesforce org, Lakeflow Connect for Salesforce lands these objects as tables incrementally (no code).
# Here they come from the CSV export in the crm_export volume.
LOADS = {
    "crm_accounts": ("Account.csv", "account_id STRING, name STRING, website_domain STRING, industry STRING, segment STRING, owner STRING"),
    "crm_contacts": ("Contact.csv", "contact_id STRING, account_id STRING, email STRING, title STRING"),
    "leads": ("Lead.csv", "lead_id STRING, company_size STRING, industry STRING, source STRING, created_date DATE"),
    "engagement": ("Lead_Engagement__c.csv", "lead_id STRING, email_opens INT, web_visits INT, demo_requested BOOLEAN, event_attended BOOLEAN, last_activity_date DATE"),
    "opportunities": ("Opportunity.csv", "lead_id STRING, converted BOOLEAN, closed_won BOOLEAN, amount DOUBLE"),
}
for table, (file, schema) in LOADS.items():
    sql = f"""CREATE OR REPLACE TABLE {table} COMMENT '{_q(TABLE_INFO[table][1])}' AS
SELECT * FROM read_files('{CRM_EXPORT}/{file}', format => 'csv', header => true, schema => '{schema}')"""
    spark.sql(sql)
print(sql)
display(spark.sql("""SELECT 'crm_accounts' AS table_name, count(*) AS rows, count_if(industry IS NULL OR segment IS NULL) AS missing_firmographics FROM crm_accounts
UNION ALL SELECT 'crm_contacts', count(*), NULL FROM crm_contacts UNION ALL SELECT 'leads', count(*), NULL FROM leads
UNION ALL SELECT 'engagement', count(*), NULL FROM engagement UNION ALL SELECT 'opportunities', count(*), NULL FROM opportunities"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Resolve identities — one master `account_id` for every record
# MAGIC Billing and support don't carry the CRM `account_id`, and CRM has duplicates. Three methods, cheapest first:
# MAGIC
# MAGIC | Records | Method | Why |
# MAGIC |---|---|---|
# MAGIC | **Support tickets** | **Rule** — the requester's email domain matches `crm_accounts.website_domain` or a contact's email domain | Deterministic SQL join, free and exact |
# MAGIC | **CRM pairs with a website on both records** | **Rule** — same website = same company; *different* websites = never the same company | Deterministic, and it stops look-alike names from being merged |
# MAGIC | **Billing accounts whose bill-to email is at the customer's domain** | **Rule** — the domain matches the CRM website or a contact's email domain | Deterministic, free and exact |
# MAGIC | **The rest of billing, and CRM duplicates** | Candidate pairs that share the **first word of the name** (or a domain), scored with **`ai_similarity`**; **auto-match at ≥ `match_threshold`** | One embedding-based score per pair |
# MAGIC | **Borderline pairs** (`review_floor` – `match_threshold`) | **`ai_classify`** on the evidence for both records — names, bill-to email domain, products billed, website and industry: *same company* or *different company* | A one-time, ad-hoc step on the borderline pairs only — a direct SQL call is simpler than a job task |
# MAGIC
# MAGIC Everything lands in **`account_xref`** (`source_system, source_id, master_account_id, match_method, score`).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE match_pairs COMMENT 'Candidate record pairs (shared first word or a shared domain) scored with ai_similarity' AS
# MAGIC WITH norm AS (
# MAGIC   SELECT account_id AS id, name, website_domain, industry,
# MAGIC          regexp_replace(lower(regexp_replace(name, '[^A-Za-z0-9 ]', ' ')), '^(the )', '') AS n FROM crm_accounts
# MAGIC ), crm AS (SELECT *, split(trim(n), ' +')[0] AS w1 FROM norm),
# MAGIC dom AS (   -- every domain we know for a CRM record: its website and its contacts' email domains
# MAGIC   SELECT account_id AS id, lower(website_domain) AS domain FROM crm_accounts WHERE website_domain IS NOT NULL
# MAGIC   UNION SELECT account_id, lower(split(email, '@')[1]) FROM crm_contacts
# MAGIC ), bil AS (    -- one row per billing account: latest name, bill-to email domain and products
# MAGIC   SELECT billing_account_id AS id, max_by(billing_account_name, month) AS name,
# MAGIC          lower(split(max_by(bill_to_email, month), '@')[1]) AS bill_to_domain, array_join(max_by(products, month), ', ') AS products
# MAGIC   FROM billing GROUP BY billing_account_id
# MAGIC ), bil2 AS (SELECT *, split(trim(regexp_replace(lower(regexp_replace(name, '[^A-Za-z0-9 ]', ' ')), '^(the )', '')), ' +')[0] AS w1 FROM bil),
# MAGIC pairs AS (
# MAGIC   SELECT 'billing' AS source_system, b.id AS source_id, b.name AS source_name, b.bill_to_domain AS source_domain, b.products AS source_products,
# MAGIC          c.id AS candidate_id, c.name AS candidate_name, c.website_domain AS candidate_domain, c.industry AS candidate_industry
# MAGIC   FROM bil2 b JOIN crm c ON b.w1 = c.w1                                       -- same first word of the name
# MAGIC   UNION
# MAGIC   SELECT 'billing', b.id, b.name, b.bill_to_domain, b.products, c.id, c.name, c.website_domain, c.industry
# MAGIC   FROM bil2 b JOIN dom d ON d.domain = b.bill_to_domain JOIN crm c ON c.id = d.id   -- or the bill-to email domain is one of theirs
# MAGIC   UNION
# MAGIC   SELECT 'crm', a.id, a.name, a.website_domain, NULL, c.id, c.name, c.website_domain, c.industry
# MAGIC   FROM crm a JOIN crm c ON a.id < c.id AND (a.w1 = c.w1 OR a.website_domain = c.website_domain)
# MAGIC )
# MAGIC SELECT *, round(ai_similarity(source_name, candidate_name), 3) AS similarity FROM pairs;
# MAGIC
# MAGIC SELECT source_system, count(*) AS candidate_pairs,
# MAGIC        count_if(similarity >= match_threshold) AS auto_match,
# MAGIC        count_if(similarity >= review_floor AND similarity < match_threshold) AS borderline_for_ai_classify,
# MAGIC        count_if(similarity < review_floor) AS rejected
# MAGIC FROM match_pairs GROUP BY ALL;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE match_candidates AS
# MAGIC WITH dom AS (
# MAGIC   SELECT account_id AS id, lower(website_domain) AS domain FROM crm_accounts WHERE website_domain IS NOT NULL
# MAGIC   UNION SELECT account_id, lower(split(email, '@')[1]) FROM crm_contacts
# MAGIC ), p AS (
# MAGIC   SELECT m.*,
# MAGIC          -- deterministic rules first: a billing account whose bill-to email is at one of the CRM account's domains, or two CRM records on the same website
# MAGIC          (m.source_system = 'billing' AND EXISTS (SELECT 1 FROM dom d WHERE d.id = m.candidate_id AND d.domain = m.source_domain))
# MAGIC            OR (m.source_system = 'crm' AND m.source_domain IS NOT NULL AND m.source_domain = m.candidate_domain) AS same_domain,
# MAGIC          -- guard: two CRM records with DIFFERENT websites are never the same company
# MAGIC          m.source_system = 'crm' AND m.source_domain IS NOT NULL AND m.candidate_domain IS NOT NULL
# MAGIC            AND m.source_domain <> m.candidate_domain AS different_domain
# MAGIC   FROM match_pairs m
# MAGIC ), borderline AS (   -- only the borderline pairs go to ai_classify — with the evidence, not just two names
# MAGIC   SELECT source_system, source_id, candidate_id,
# MAGIC          ai_classify(concat('Record A (', source_system, ' system): "', source_name, '"',
# MAGIC                             coalesce(concat(CASE WHEN source_system = 'billing' THEN ', bill-to email domain ' ELSE ', website ' END, source_domain), ''),
# MAGIC                             coalesce(concat(', products billed: ', source_products), ''),
# MAGIC                             '. Record B (CRM): "', candidate_name, '"', coalesce(concat(', website ', candidate_domain), ''),
# MAGIC                             coalesce(concat(', industry ', candidate_industry), ''),
# MAGIC                             '. Are these the same company? Suffixes like Inc / Corp / LLC, capitalization, abbreviated words and regional tags such as (EMEA) ',
# MAGIC                             'do not make them different. A bill-to email at an outsourced payables provider is neither evidence for nor against. ',
# MAGIC                             'A shared common surname (Davis, Jones, Smith, Williams ...) with a different legal suffix and no other matching evidence is NOT enough: answer different company.'),
# MAGIC                      array('same company', 'different company')) AS verdict
# MAGIC   FROM p WHERE NOT same_domain AND NOT different_domain AND similarity >= review_floor AND similarity < match_threshold
# MAGIC )
# MAGIC SELECT p.*,
# MAGIC   CASE WHEN p.same_domain OR p.different_domain THEN 'rule' WHEN p.similarity >= match_threshold THEN 'ai_similarity'
# MAGIC        WHEN b.verdict IS NOT NULL THEN 'ai_classify' END AS match_method,
# MAGIC   b.verdict AS ai_classify_verdict,
# MAGIC   p.same_domain OR (NOT p.different_domain AND (p.similarity >= match_threshold OR coalesce(b.verdict = 'same company', false))) AS is_match
# MAGIC FROM p LEFT JOIN borderline b USING (source_system, source_id, candidate_id);
# MAGIC
# MAGIC SELECT source_system, match_method, count(*) AS pairs, count_if(is_match) AS matched,
# MAGIC        max_by(CASE WHEN match_method = 'ai_classify' AND is_match THEN concat(source_name, coalesce(concat(' (', source_domain, ')'), ''), '  =  ', candidate_name) END,
# MAGIC               CASE WHEN match_method = 'ai_classify' AND is_match THEN CAST(source_domain IS NOT NULL AS INT) * 10 + similarity END) AS example_match,
# MAGIC        max_by(CASE WHEN match_method = 'ai_classify' AND NOT is_match THEN concat(source_name, '  ≠  ', candidate_name) END,
# MAGIC               CASE WHEN match_method = 'ai_classify' AND NOT is_match THEN similarity END) AS example_rejected
# MAGIC FROM match_candidates WHERE match_method IS NOT NULL GROUP BY ALL ORDER BY source_system, match_method;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE account_xref AS
# MAGIC WITH contacts AS (SELECT account_id, count(*) AS n FROM crm_contacts GROUP BY ALL),
# MAGIC dup AS (   -- CRM duplicate pairs: the record with contacts survives (ties: the older id)
# MAGIC   SELECT * FROM (
# MAGIC   SELECT CASE WHEN coalesce(cb.n, 0) > coalesce(ca.n, 0) THEN m.source_id ELSE m.candidate_id END AS duplicate_id,
# MAGIC          CASE WHEN coalesce(cb.n, 0) > coalesce(ca.n, 0) THEN m.candidate_id ELSE m.source_id END AS master_account_id,
# MAGIC          m.match_method, m.similarity
# MAGIC   FROM match_candidates m
# MAGIC   LEFT JOIN contacts ca ON ca.account_id = m.source_id LEFT JOIN contacts cb ON cb.account_id = m.candidate_id
# MAGIC   WHERE m.source_system = 'crm' AND m.is_match)
# MAGIC   QUALIFY row_number() OVER (PARTITION BY duplicate_id ORDER BY similarity DESC) = 1
# MAGIC ), crm_map AS (
# MAGIC   SELECT a.account_id AS source_id, a.name AS source_name, coalesce(d.master_account_id, a.account_id) AS master_account_id,
# MAGIC          coalesce(d.match_method, 'self') AS match_method, coalesce(d.similarity, 1.0) AS score
# MAGIC   FROM crm_accounts a LEFT JOIN dup d ON d.duplicate_id = a.account_id
# MAGIC ), bill AS (
# MAGIC   SELECT m.source_id, m.source_name, c.master_account_id, m.match_method, m.similarity AS score
# MAGIC   FROM match_candidates m JOIN crm_map c ON c.source_id = m.candidate_id
# MAGIC   WHERE m.source_system = 'billing' AND m.is_match
# MAGIC   QUALIFY row_number() OVER (PARTITION BY m.source_id ORDER BY (m.match_method = 'rule') DESC, (c.match_method = 'self') DESC, m.similarity DESC) = 1
# MAGIC ), domains AS (    -- every domain we know for a master account: website domains and contact email domains
# MAGIC   SELECT DISTINCT lower(a.website_domain) AS domain, c.master_account_id FROM crm_accounts a JOIN crm_map c ON c.source_id = a.account_id WHERE a.website_domain IS NOT NULL
# MAGIC   UNION SELECT DISTINCT lower(split(ct.email, '@')[1]), c.master_account_id FROM crm_contacts ct JOIN crm_map c ON c.source_id = ct.account_id
# MAGIC ), tix AS (
# MAGIC   SELECT t.ticket_id AS source_id, t.requester_email AS source_name, d.master_account_id, 'rule' AS match_method, 1.0 AS score
# MAGIC   FROM support_tickets t JOIN domains d ON d.domain = lower(split(t.requester_email, '@')[1])
# MAGIC   QUALIFY count(*) OVER (PARTITION BY t.ticket_id) = 1      -- skip ambiguous domains
# MAGIC )
# MAGIC SELECT 'crm' AS source_system, * FROM crm_map
# MAGIC UNION ALL SELECT 'billing', * FROM bill
# MAGIC UNION ALL SELECT 'support', * FROM tix;
# MAGIC
# MAGIC SELECT source_system, match_method, count(*) AS records, count(DISTINCT master_account_id) AS master_accounts
# MAGIC FROM account_xref GROUP BY ALL ORDER BY source_system, match_method;

# COMMAND ----------

# MAGIC %sql WITH q AS (SELECT source_system, match_method, count(*) AS records, count(DISTINCT master_account_id) AS master_accounts
# MAGIC FROM account_xref GROUP BY ALL ORDER BY source_system, match_method) SELECT `source_system`,SUM(`records`) `column_7864429402`,`match_method` FROM q GROUP BY `source_system`,`match_method`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT x.source_system, x.source_id, x.source_name, x.master_account_id, x.match_method, round(x.score, 3) AS score
# MAGIC FROM account_xref x
# MAGIC WHERE x.master_account_id IN (SELECT master_account_id FROM account_xref WHERE source_name ILIKE '%acme%')
# MAGIC   AND x.source_system <> 'support'
# MAGIC ORDER BY x.source_system;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Fill in firmographics — `ai_enrich`
# MAGIC Twenty accounts have no industry or segment — sales ops never filled them in. *(Step 1.1 counts a few more: some CRM duplicates are missing them too, but those merge into their master account in step 1.2, so only the 20 masters are enriched.)* `ai_enrich` generates the missing columns **from a schema you define** (here, enums matching the CRM picklists), and with **rationale on**, every value comes back with **its reasoning** so a sales-ops reviewer can accept it in seconds. Set the **`enrich_with_web_search`** widget to `yes` to ground the values in a **web search** (an MCP search source works too).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE account_enrichment AS
# MAGIC SELECT account_id, name, website_domain, enriched,
# MAGIC        enriched:response:industry:value::STRING      AS industry,
# MAGIC        enriched:response:industry:rationale::STRING  AS industry_rationale,
# MAGIC        enriched:response:segment:value::STRING       AS segment,
# MAGIC        enriched:response:segment:rationale::STRING   AS segment_rationale,
# MAGIC        enriched:response:hq_region:value::STRING     AS hq_region
# MAGIC FROM (
# MAGIC   SELECT a.*, ai_enrich(
# MAGIC            concat('Company: ', a.name, coalesce(concat(' (website ', a.website_domain, ')'), '')),
# MAGIC            '{"industry": {"type": "enum", "labels": ["Financial Services", "Insurance", "Healthcare", "Life Sciences", "Manufacturing", "Retail & CPG", "Technology", "Energy & Utilities"]},
# MAGIC              "segment": {"type": "enum", "labels": ["Enterprise", "Mid-Market", "SMB"], "description": "Enterprise = over $1B annual revenue, Mid-Market = $50M-$1B, SMB = under $50M"},
# MAGIC              "hq_region": {"type": "enum", "labels": ["North America", "EMEA", "APAC", "LATAM"]}}',
# MAGIC            enrich_sources,
# MAGIC            map('instructions', 'Fill in B2B firmographics for this customer account, as of today.', 'enableRationale', 'true')) AS enriched
# MAGIC   FROM crm_accounts a
# MAGIC   JOIN account_xref x ON x.source_system = 'crm' AND x.source_id = a.account_id AND x.match_method = 'self'
# MAGIC   WHERE a.industry IS NULL OR a.segment IS NULL
# MAGIC );
# MAGIC
# MAGIC SELECT name, industry, segment, hq_region, industry_rationale FROM account_enrichment ORDER BY name;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Turn text into signals — ticket themes and NPS sentiment
# MAGIC * **Ticket themes — an `ai_classify` AI Function task.** The job from step 0.7 has been classifying every ticket in the background. This step shows the task and waits for the backfill if it's still running.
# MAGIC * **NPS sentiment — `ai_analyze_sentiment`**, as a SQL call in the notebook: there's no dedicated job task type for sentiment, and it runs once over the whole table.

# COMMAND ----------

show_task(classify_task, "Classification", "Classifies every new or changed support ticket into billing, outage, feature request or onboarding.")
backfill = wait_for_run(BACKFILL["run_id"], "one-time backfill")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE tickets_resolved AS
# MAGIC SELECT t.ticket_id, x.master_account_id, t.requester_email, t.severity, t.opened, t.closed, t.description,
# MAGIC        c.response:response[0]:value::STRING AS theme, round(c.response:response[0]:confidence_score::DOUBLE, 3) AS theme_confidence
# MAGIC FROM support_tickets t
# MAGIC LEFT JOIN account_xref x ON x.source_system = 'support' AND x.source_id = t.ticket_id
# MAGIC LEFT JOIN ticket_classifications c USING (ticket_id);
# MAGIC
# MAGIC CREATE OR REPLACE TABLE nps_sentiment AS
# MAGIC SELECT account_id, date, score, comment, ai_analyze_sentiment(comment) AS sentiment FROM nps_surveys;
# MAGIC
# MAGIC SELECT 'ticket theme' AS signal, theme AS value, count(*) AS n FROM tickets_resolved GROUP BY ALL
# MAGIC UNION ALL SELECT 'NPS sentiment', sentiment, count(*) FROM nps_sentiment GROUP BY ALL
# MAGIC ORDER BY signal, n DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (SELECT 'ticket theme' AS signal, theme AS value, count(*) AS n FROM tickets_resolved GROUP BY ALL
# MAGIC UNION ALL SELECT 'NPS sentiment', sentiment, count(*) FROM nps_sentiment GROUP BY ALL
# MAGIC ORDER BY signal, n DESC) SELECT `value`,SUM(`n`) `column_7342938094`,`signal` FROM q GROUP BY `value`,`signal`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Build the gold `customer_360` table
# MAGIC One row per **master account**, joined through `account_xref`: **revenue trend**, **active users**, **seat utilization** (active users ÷ `seats_purchased`), **ticket volume and themes**, **NPS and sentiment**, **products owned and not yet owned** — and an **`ai_summarize` account brief** a rep can read in ten seconds before a call.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE account_weekly_usage AS
# MAGIC SELECT account_id, CAST(date_trunc('WEEK', timestamp) AS DATE) AS week_start, count(DISTINCT user_id) AS active_users
# MAGIC FROM product_events GROUP BY ALL;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE customer_360 AS
# MAGIC WITH m AS (
# MAGIC   SELECT a.account_id, a.name AS account_name, a.owner, a.website_domain,
# MAGIC          coalesce(a.industry, e.industry) AS industry, coalesce(a.segment, e.segment) AS segment, e.hq_region,
# MAGIC          CASE WHEN a.industry IS NULL OR a.segment IS NULL THEN 'ai_enrich' ELSE 'CRM' END AS firmographics_source
# MAGIC   FROM crm_accounts a JOIN account_xref x ON x.source_system = 'crm' AND x.source_id = a.account_id AND x.match_method = 'self'
# MAGIC   LEFT JOIN account_enrichment e USING (account_id)
# MAGIC ), b AS (
# MAGIC   SELECT x.master_account_id AS account_id,
# MAGIC          sum(CASE WHEN month > add_months(lme, -3) THEN revenue END) AS rev_3m,
# MAGIC          sum(CASE WHEN month BETWEEN add_months(lme, -5) AND add_months(lme, -3) THEN revenue END) AS rev_prev_3m,
# MAGIC          sum(CASE WHEN month > add_months(lme, -12) THEN revenue END) AS revenue_12m,
# MAGIC          sum(CASE WHEN month = lme THEN seats_purchased END) AS seats_purchased,
# MAGIC          array_distinct(flatten(collect_list(CASE WHEN month = lme THEN products END))) AS products_owned,
# MAGIC          round(avg(CASE WHEN month > add_months(lme, -6) THEN days_late END), 1) AS avg_days_late_6m,
# MAGIC          count(DISTINCT billing_account_id) AS billing_accounts
# MAGIC   FROM billing JOIN account_xref x ON x.source_system = 'billing' AND x.source_id = billing.billing_account_id
# MAGIC   GROUP BY ALL
# MAGIC ), u AS (
# MAGIC   SELECT account_id,
# MAGIC          avg(CASE WHEN week_start > date_sub(current_date(), 35) THEN active_users END) AS active_users_4w,
# MAGIC          avg(CASE WHEN week_start > date_sub(current_date(), 63) THEN active_users END) AS au_8w,
# MAGIC          avg(CASE WHEN week_start BETWEEN date_sub(current_date(), 119) AND date_sub(current_date(), 63) THEN active_users END) AS au_prev_8w
# MAGIC   FROM account_weekly_usage GROUP BY ALL
# MAGIC ), t AS (
# MAGIC   SELECT master_account_id AS account_id, count(*) AS tickets_90d, count_if(severity = 'Sev 1') AS sev1_tickets_90d,
# MAGIC          max_by(theme, opened) AS latest_ticket_theme
# MAGIC   FROM tickets_resolved WHERE opened > current_date() - INTERVAL 90 DAYS AND master_account_id IS NOT NULL GROUP BY ALL
# MAGIC ), tt AS (   -- ticket themes, most frequent first
# MAGIC   SELECT account_id, concat_ws(', ', transform(array_sort(collect_list(struct(-n AS k, theme))), x -> x.theme)) AS ticket_themes
# MAGIC   FROM (SELECT master_account_id AS account_id, theme, count(*) AS n FROM tickets_resolved
# MAGIC         WHERE opened > current_date() - INTERVAL 90 DAYS AND master_account_id IS NOT NULL AND theme IS NOT NULL GROUP BY ALL)
# MAGIC   GROUP BY ALL
# MAGIC ), nr AS (SELECT account_id, min(renewal_date) AS next_renewal_date FROM contract_renewals WHERE renewal_date > current_date() GROUP BY ALL
# MAGIC ), n AS (
# MAGIC   SELECT account_id, max_by(score, date) AS latest_nps, max_by(comment, date) AS latest_nps_comment,
# MAGIC          round(avg(CASE WHEN sentiment = 'negative' THEN 1.0 ELSE 0.0 END) * 100) AS negative_sentiment_pct_12m
# MAGIC   FROM nps_sentiment WHERE date > add_months(current_date(), -12) GROUP BY ALL
# MAGIC ), j AS (
# MAGIC   SELECT m.*, round(b.revenue_12m) AS revenue_12m, round((b.rev_3m / nullif(b.rev_prev_3m, 0) - 1) * 100, 1) AS revenue_trend_pct,
# MAGIC          b.seats_purchased, round(u.active_users_4w) AS active_users, round(u.active_users_4w / nullif(b.seats_purchased, 0), 2) AS seat_utilization,
# MAGIC          round((u.au_8w / nullif(u.au_prev_8w, 0) - 1) * 100, 1) AS usage_trend_pct,
# MAGIC          coalesce(t.tickets_90d, 0) AS tickets_90d, coalesce(t.sev1_tickets_90d, 0) AS sev1_tickets_90d, tt.ticket_themes,
# MAGIC          n.latest_nps, n.negative_sentiment_pct_12m, n.latest_nps_comment,
# MAGIC          b.products_owned, array_except(array('Platform', 'Data API', 'Analytics Pro', 'Managed Storage'), b.products_owned) AS products_not_owned,
# MAGIC          b.avg_days_late_6m, b.billing_accounts, nr.next_renewal_date
# MAGIC   FROM m JOIN b USING (account_id) LEFT JOIN u USING (account_id) LEFT JOIN t USING (account_id) LEFT JOIN tt USING (account_id)
# MAGIC   LEFT JOIN n USING (account_id) LEFT JOIN nr USING (account_id)
# MAGIC )
# MAGIC SELECT *, ai_summarize(concat(
# MAGIC   account_name, ' is a ', coalesce(segment, ''), ' ', coalesce(industry, ''), ' customer owned by ', owner, '. ',
# MAGIC   'Revenue last 12 months $', format_number(revenue_12m, 0), ', trend ', coalesce(revenue_trend_pct, 0), '% (last 3 months vs prior 3). ',
# MAGIC   'Products: ', array_join(products_owned, ', '), '; not yet owned: ', array_join(products_not_owned, ', '), '. ',
# MAGIC   coalesce(active_users, 0), ' weekly active users on ', coalesce(seats_purchased, 0), ' seats (utilization ', coalesce(seat_utilization, 0), '), usage trend ', coalesce(usage_trend_pct, 0), '%. ',
# MAGIC   tickets_90d, ' tickets in 90 days (', sev1_tickets_90d, ' Sev 1; themes: ', coalesce(ticket_themes, 'none'), '). ',
# MAGIC   'Latest NPS ', coalesce(CAST(latest_nps AS STRING), 'n/a'), ': "', coalesce(latest_nps_comment, ''), '". ',
# MAGIC   'Pays ', coalesce(avg_days_late_6m, 0), ' days late on average. Next renewal ', coalesce(CAST(next_renewal_date AS STRING), 'n/a'), '.'), 60) AS account_brief
# MAGIC FROM j;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT account_name, owner, segment, industry, revenue_12m, revenue_trend_pct, seats_purchased, active_users, seat_utilization, usage_trend_pct,
# MAGIC        tickets_90d, ticket_themes, latest_nps, products_owned, products_not_owned, billing_accounts, next_renewal_date, account_brief
# MAGIC FROM customer_360 WHERE account_name ILIKE '%acme%';

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.7 Check accuracy in MLflow — matching precision and recall
# MAGIC The answer key becomes a **versioned evaluation dataset in Unity Catalog** (`identity_resolution_eval` — it shows in the experiment's *Datasets* tab): one record per billing account, support ticket and CRM record, with the **true master account** as the expectation. `mlflow.genai.evaluate` runs a **traced** `predict_fn` that looks each record up in `account_xref`, so every prediction is a trace you can open, and custom scorers grade it. On top, the cell reports **precision by match method** (`rule`, `ai_similarity`, `ai_classify`), **recall by source** and scores the ticket themes from the `ai_classify` task.

# COMMAND ----------

import mlflow, pandas as pd
from mlflow.genai.scorers import scorer
pairs = spark.sql("""
WITH truth AS (
  SELECT CASE anomaly_type WHEN 'billing_mapping' THEN 'billing' WHEN 'crm_duplicate' THEN 'crm' ELSE 'support' END AS source_system,
         entity_id AS source_id, true_value AS true_master
  FROM ground_truth WHERE anomaly_type IN ('billing_mapping', 'crm_duplicate', 'ticket_mapping')
  UNION ALL   -- CRM records that are NOT duplicates should stay their own master (catches false merges)
  SELECT 'crm', account_id, account_id FROM crm_accounts
  WHERE account_id NOT IN (SELECT entity_id FROM ground_truth WHERE anomaly_type = 'crm_duplicate')
)
SELECT t.source_system AS kind, t.source_id, t.true_master, x.master_account_id AS predicted_master,
       coalesce(x.match_method, 'unmatched') AS match_method
FROM truth t LEFT JOIN account_xref x USING (source_system, source_id)
""").toPandas()

# 1) The answer key as a versioned evaluation dataset in Unity Catalog
dataset = eval_dataset("identity_resolution_eval", [
    {"inputs": {"source_system": r.kind, "source_id": r.source_id}, "expectations": {"master": r.true_master or "unmatched"}}
    for r in pairs.itertuples()])

# 2) A traced predict_fn: what account_xref says for each record (parameter names match the dataset's inputs)
XREF = {(r.kind, r.source_id): (r.predicted_master or "unmatched", r.match_method) for r in pairs.itertuples()}
@mlflow.trace(span_type="RETRIEVER")
def predict_fn(source_system, source_id):
    master, method = XREF.get((source_system, source_id), ("unmatched", "unmatched"))
    return {"master": master, "match_method": method}

@scorer
def correct_match(outputs, expectations):
    """Right master account — or correctly left unmatched (e.g. a ticket from a gmail.com address)."""
    return outputs["master"] == expectations["master"]
@scorer
def matched(outputs):
    return outputs["master"] != "unmatched"

experiment()
with mlflow.start_run(run_name="1.7 identity resolution vs ground truth") as run:
    res = mlflow.genai.evaluate(data=dataset, predict_fn=predict_fn, scorers=[correct_match, matched])
    pairs["correct"] = pairs.predicted_master == pairs.true_master
    pairs["link"] = ~pairs.match_method.isin(["self", "unmatched"])                       # a predicted cross-record link
    pairs["true_link"] = pairs.true_master.notna() & (pairs.true_master != pairs.source_id)
    precision = (pairs.link & pairs.correct).sum() / max(pairs.link.sum(), 1)
    recall = (pairs.true_link & pairs.link & pairs.correct).sum() / max(pairs.true_link.sum(), 1)
    by = (pairs[pairs.link].groupby(["kind", "match_method"])
              .agg(links=("correct", "size"), precision_pct=("correct", lambda x: round(100 * x.mean(), 1))).reset_index())
    rec = pairs[pairs.true_link].groupby("kind").apply(lambda d: round(100 * (d.link & d.correct).mean(), 1)).rename("recall_pct").reset_index()
    by = by.merge(rec, on="kind", how="left")
    mlflow.log_metrics({"precision_pct": round(100 * precision, 1), "recall_pct": round(100 * recall, 1),
                        "false_merges": int(((pairs.kind == "crm") & (pairs.true_master == pairs.source_id) & pairs.link).sum())})
    for r in by.itertuples():
        mlflow.log_metric(f"precision_pct_{r.kind}_{r.match_method}", r.precision_pct)
    for r in rec.itertuples():
        mlflow.log_metric(f"recall_pct_{r.kind}", r.recall_pct)
    # Ticket themes from the ai_classify task vs the theme each ticket was written about
    th = spark.sql("""SELECT avg(CASE WHEN t.theme = g.detail THEN 1.0 ELSE 0.0 END) AS acc FROM tickets_resolved t
                      JOIN ground_truth g ON g.anomaly_type = 'ticket_mapping' AND g.entity_id = t.ticket_id""").first().acc
    mlflow.log_metric("ticket_theme_accuracy_pct", round(100 * th, 1))
    mlflow.log_table(by, "matching_by_method.json")
    show_run(run, "Identity resolution: precision & recall")
display(by)
print(f"Overall precision {100 * precision:.1f}% · recall {100 * recall:.1f}% · {res.metrics.get('correct_match/mean', 0) * 100:.1f}% of records map correctly")
print(f"Ticket themes (ai_classify task): {100 * th:.1f}% match the theme each ticket was written about")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.8 Tomorrow — the same job, only the new tickets
# MAGIC A dozen new tickets from Acme arrive. The **same job** runs again and classifies **only those** — not the ones it already did. In production that's the job's **hourly schedule** (created paused in step 0.7 — unpause it on the job page). The cell then folds the new tickets into `account_xref`, `tickets_resolved` and Acme's `customer_360` row (ticket counts, themes and the account brief), so the next Genie question sees them.

# COMMAND ----------

before = spark.table("ticket_classifications").count()
spark.sql("""
INSERT INTO support_tickets
SELECT concat('TCK-9', lpad(CAST(k AS STRING), 5, '0')) AS ticket_id,
       concat('ops', k, '@', (SELECT website_domain FROM crm_accounts WHERE name = 'Acme Corporation')) AS requester_email,
       CASE WHEN k % 4 = 1 THEN 'Sev 1' ELSE 'Sev 3' END AS severity, current_timestamp() AS opened, CAST(NULL AS TIMESTAMP) AS closed,
       ai_gen(concat('Write the body of a customer support ticket (2 sentences, first person, plain text) for a B2B data platform about ',
                     element_at(array('dashboards timing out since this morning', 'a duplicate charge on this month invoice',
                                      'adding SAML single sign-on for a new team', 'wanting row-level security on shared dashboards'), k % 4 + 1), '.')) AS description
FROM (SELECT explode(sequence(1, 12)) AS k)
""")
daily = run_job(JOB_ID, "scheduled run")
after = spark.table("ticket_classifications").count()
displayHTML(f'<div style="{FONT};font-size:14px">Tickets classified: <b>{before:,}</b> after the backfill → <b>{after:,}</b> after the scheduled run — '
            f'it classified only the <b>{after - before}</b> new tickets.</div>')
# Fold today's tickets into the customer view (same rules as steps 1.2c, 1.4b and 1.5, for the new rows only)
spark.sql("""INSERT INTO account_xref
SELECT 'support', t.ticket_id, t.requester_email, x.master_account_id, 'rule', 1.0
FROM support_tickets t JOIN crm_accounts a ON lower(a.website_domain) = lower(split(t.requester_email, '@')[1])
JOIN account_xref x ON x.source_system = 'crm' AND x.source_id = a.account_id
WHERE t.ticket_id LIKE 'TCK-9%' AND t.ticket_id NOT IN (SELECT source_id FROM account_xref WHERE source_system = 'support')""")
spark.sql("""INSERT INTO tickets_resolved
SELECT t.ticket_id, x.master_account_id, t.requester_email, t.severity, t.opened, t.closed, t.description,
       c.response:response[0]:value::STRING, round(c.response:response[0]:confidence_score::DOUBLE, 3)
FROM support_tickets t LEFT JOIN account_xref x ON x.source_system = 'support' AND x.source_id = t.ticket_id
LEFT JOIN ticket_classifications c USING (ticket_id)
WHERE t.ticket_id LIKE 'TCK-9%' AND t.ticket_id NOT IN (SELECT ticket_id FROM tickets_resolved)""")
spark.sql("""MERGE INTO customer_360 c
USING (SELECT master_account_id AS account_id, count(*) AS tickets_90d, count_if(severity = 'Sev 1') AS sev1_tickets_90d,
              concat_ws(', ', array_distinct(collect_list(theme))) AS ticket_themes,
              count_if(ticket_id LIKE 'TCK-9%') AS new_today, count_if(ticket_id LIKE 'TCK-9%' AND severity = 'Sev 1') AS new_sev1
       FROM tickets_resolved WHERE opened > current_date() - INTERVAL 90 DAYS AND master_account_id IS NOT NULL
       GROUP BY ALL HAVING count_if(ticket_id LIKE 'TCK-9%') > 0) t
ON c.account_id = t.account_id
WHEN MATCHED THEN UPDATE SET tickets_90d = t.tickets_90d, sev1_tickets_90d = t.sev1_tickets_90d, ticket_themes = t.ticket_themes,
  account_brief = ai_summarize(concat('URGENT UPDATE TODAY: ', t.new_today, ' new support tickets (', t.new_sev1, ' Sev 1); ticket themes in the last 90 days: ',
                                      t.ticket_themes, '. Mention this first. Earlier brief: ', c.account_brief), 60)""")
display(spark.sql("""SELECT t.ticket_id, t.severity, c.response:response[0]:value::STRING AS theme, t.description
                     FROM support_tickets t JOIN ticket_classifications c USING (ticket_id) WHERE t.ticket_id LIKE 'TCK-9%' ORDER BY t.ticket_id"""))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.9 💬 Ask Genie — Customer 360
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Show me everything about Acme.", before=("Section 1 (identity resolution and customer_360)", 160))

# COMMAND ----------

ask("Which accounts grew usage but not spend?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.11 🛠 Try it yourself
# MAGIC Set **`match_threshold`** to `0.90`, re-run **S.2**, then **steps 1.2a–1.2c**, **1.4b**, **1.5** and **1.7**. More pairs now go to `ai_classify` — does precision change? Which method would you trust for auto-merging records in Salesforce?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Lead scoring — which leads should reps call this week, and why?
# MAGIC
# MAGIC **When an AI Function, when an MLflow model?** Default to an **AI Function** when a one-step SQL answer is the point and retraining on each run is fine — section 3 uses `ai_predict_class` for churn and upsell. Use an **MLflow model** when you need a **registered, versioned model** (for CRM write-back), **per-row explanations** reps can read, or **real-time serving** — that's lead scoring. `ai_predict_class` trains a new model on every call and doesn't save it, so step 2.4 scores the same hold-out with it and step 2.5 compares it with the MLflow `@champion`: *an analyst gets most of the way there in one SQL call, and MLflow is the production path.*
# MAGIC
# MAGIC The MLflow path here is built the production way: lead features live in a **governed feature table keyed by `lead_id`**, with labels kept in a separate table and joined through a `FeatureLookup` training set, so Unity Catalog shows the lineage from feature to model. Training is **autologged**, and the model is **registered in Unity Catalog** and scored by its **`@champion`** alias — promoting a better version later needs no code change. Reps get the **top 3 SHAP reasons** for every score. In the experiment, the model's training run holds the full hold-out evaluation — ROC, precision-recall, lift and calibration curves, the confusion matrix — plus SHAP feature importance and a beeswarm plot (*Artifacts* and *Model metrics* tabs).
# MAGIC
# MAGIC | Step | What happens | Capability |
# MAGIC |---|---|---|
# MAGIC | 2.1 | Engineer features (engagement recency & frequency, firmographic fit, source) into a **feature table** with primary key `lead_id`; keep labels in `lead_labels` | SQL + **Feature Engineering in Unity Catalog** |
# MAGIC | 2.2 | Train on a `FeatureLookup` training set, log with `fe.log_model`, register in Unity Catalog and set the **`@champion`** alias | scikit-learn + **MLflow** + UC registry |
# MAGIC | 2.3 | Batch-score open leads from **`@champion`** (features looked up automatically), with the **top 3 SHAP reasons**; turn them into a rep talking point | `fe.score_batch` + `shap` · `ai_gen` |
# MAGIC | 2.4 | The analyst path on the same hold-out | `ai_predict_class` |
# MAGIC | 2.5 | Compare AUC and top-decile lift | **MLflow** |
# MAGIC | 2.6 | Act — write scores back to CRM and draft outreach for the top 20 | reverse-ETL stub · `ai_gen` |

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Features (no labels): one row per lead. Step 2.2 turns this into a governed feature table with primary key lead_id.
# MAGIC CREATE OR REPLACE TEMP VIEW lead_feature_source AS
# MAGIC SELECT l.lead_id, l.created_date, l.company_size, l.industry, l.source,
# MAGIC        e.email_opens, e.web_visits, e.email_opens + e.web_visits AS engagement_frequency,
# MAGIC        datediff(current_date(), e.last_activity_date) AS days_since_last_activity,
# MAGIC        datediff(e.last_activity_date, l.created_date) AS days_engaged,
# MAGIC        CAST(e.demo_requested AS INT) AS demo_requested, CAST(e.event_attended AS INT) AS event_attended,
# MAGIC        CASE WHEN l.industry IN ('Financial Services', 'Insurance', 'Healthcare', 'Life Sciences', 'Manufacturing')
# MAGIC              AND l.company_size IN ('201-1000', '1001-5000', '5000+') THEN 1 ELSE 0 END AS icp_fit
# MAGIC FROM leads l JOIN engagement e USING (lead_id);
# MAGIC
# MAGIC -- Labels, kept apart from features: label = opportunities.converted; open leads have no opportunity yet
# MAGIC CREATE OR REPLACE TABLE lead_labels COMMENT 'Lead labels kept apart from features: converted, closed_won and the training / hold-out / open split' AS
# MAGIC SELECT l.lead_id, o.converted, o.closed_won,
# MAGIC        CASE WHEN o.lead_id IS NULL THEN 'open' WHEN pmod(xxhash64(l.lead_id), 5) = 0 THEN 'hold-out' ELSE 'training' END AS split
# MAGIC FROM leads l LEFT JOIN opportunities o USING (lead_id);
# MAGIC
# MAGIC SELECT split, count(*) AS leads, round(avg(CAST(converted AS INT)) * 100, 1) AS conversion_rate_pct FROM lead_labels GROUP BY ALL;

# COMMAND ----------

import mlflow, numpy as np, pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
CAT = ["company_size", "industry", "source"]
NUM = ["email_opens", "web_visits", "engagement_frequency", "days_since_last_activity", "days_engaged", "demo_requested", "event_attended", "icp_fit"]
X_cols = CAT + NUM
fe = FeatureEngineeringClient()
FEATURE_TABLE, LEAD_MODEL = fq("lead_features"), fq("lead_conversion_model")

# 1) Governed feature table (primary key lead_id) — reusable by other models and by Model Serving's automatic feature lookup
spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")
fe.create_table(name=FEATURE_TABLE, primary_keys=["lead_id"], df=spark.table("lead_feature_source"),
                description="Lead features: engagement recency / frequency, firmographic fit, source. One row per lead.")

# 2) Training set = labels + FeatureLookup (the same lookup Model Serving does at inference time)
labels = spark.table("lead_labels")
def training_set(split, keep_id=False):
    return fe.create_training_set(df=labels.filter(f"split = '{split}'").selectExpr("lead_id", "CAST(converted AS INT) AS converted"),
                                  feature_lookups=[FeatureLookup(table_name=FEATURE_TABLE, lookup_key="lead_id", feature_names=X_cols)],
                                  label="converted", exclude_columns=[] if keep_id else ["lead_id"])
TRAIN_SET = training_set("training")
train, hold = TRAIN_SET.load_df().toPandas(), training_set("hold-out", keep_id=True).load_df().toPandas()

class ConversionScorer(BaseEstimator, ClassifierMixin):
    """Wraps a classifier so the registered model's predict() returns P(converted) — what CRM write-back and fe.score_batch need."""
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
    def fit(self, X, y):
        self.pipeline.fit(X[X_cols], y); self.classes_ = self.pipeline.classes_; return self
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X[X_cols])
    def predict(self, X):
        return self.pipeline.predict_proba(X[X_cols])[:, 1]

def lift_top_decile(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(y[p >= np.quantile(p, 0.9)].mean() / y.mean())

LEAD_PIPELINE = Pipeline([("prep", ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
                                                      remainder="passthrough")),
                           ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_depth=3, min_samples_leaf=60,
                                                                    l2_regularization=2.0, random_state=42))])
experiment()
mlflow.sklearn.autolog(log_models=False, log_datasets=False, silent=True)   # params + training metrics automatically; the model is logged by fe.log_model
with mlflow.start_run(run_name="2.2 lead conversion — scikit-learn (production path)") as run:
    lead_model = ConversionScorer(LEAD_PIPELINE).fit(train, train.converted)
    p_hold = lead_model.predict(hold)
    AUC_MLFLOW, LIFT_MLFLOW = roc_auc_score(hold.converted, p_hold), lift_top_decile(hold.converted, p_hold)
    mlflow.log_params({"feature_table": FEATURE_TABLE, "training_leads": len(train), "holdout_leads": len(hold)})
    mlflow.log_metrics({"holdout_auc": round(AUC_MLFLOW, 4), "holdout_top_decile_lift": round(LIFT_MLFLOW, 2)})
    # fe.log_model records the feature lookups with the model, so Model Serving and fe.score_batch fetch features by lead_id
    fe.log_model(model=lead_model, artifact_path="model", flavor=mlflow.sklearn, training_set=TRAIN_SET, registered_model_name=LEAD_MODEL)
    # Traditional evaluation on the hold-out: ROC / precision-recall / lift / calibration curves, confusion matrix,
    # classification metrics and SHAP explainability, all logged to this run (Artifacts and Model metrics tabs)
    eval_copy = mlflow.sklearn.log_model(LEAD_PIPELINE, name="holdout_evaluation_model", input_example=hold[X_cols].head(3))
    eval_data = hold[X_cols + ["converted"]]
    try:
        evaluation = mlflow.models.evaluate(model=eval_copy.model_uri, data=eval_data, targets="converted", model_type="classifier",
                                            evaluator_config={"log_model_explainability": True, "explainability_nsamples": 300})
    except Exception as e:   # SHAP on string categories can fail in some shap versions — evaluate without it; step 2.3 logs SHAP itself
        print(f"(explainability skipped in mlflow.models.evaluate: {str(e).splitlines()[0][:120]})")
        evaluation = mlflow.models.evaluate(model=eval_copy.model_uri, data=eval_data, targets="converted", model_type="classifier",
                                            evaluator_config={"log_model_explainability": False})
    show_run(run, "MLflow model: lead conversion")
LEAD_RUN_ID = run.info.run_id
artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(LEAD_RUN_ID) if not a.is_dir]
print("Evaluation artifacts in the run:", ", ".join(sorted(artifacts)))
print({k: round(v, 3) for k, v in evaluation.metrics.items() if k in ("roc_auc", "precision_recall_auc", "f1_score", "log_loss", "accuracy_score")})
mlflow.sklearn.autolog(disable=True)
client = mlflow.MlflowClient(registry_uri="databricks-uc")
LEAD_MODEL_VERSION = max((v for v in client.search_model_versions(f"name='{LEAD_MODEL}'") if v.run_id == run.info.run_id),
                         key=lambda v: int(v.version)).version
set_alias(LEAD_MODEL, "champion", LEAD_MODEL_VERSION)      # scoring jobs and serving point at @champion, never a hard-coded version
print(f"Hold-out AUC {AUC_MLFLOW:.3f} · top-decile lift {LIFT_MLFLOW:.1f}x · registered {LEAD_MODEL} v{LEAD_MODEL_VERSION} as @champion")
displayHTML(button("Open the model in Unity Catalog", f"{HOST}/explore/data/models/{CATALOG}/{SCHEMA}/lead_conversion_model{ORG}", "#1B3139")
            + button("Open the feature table", uc_url(CATALOG, SCHEMA, "lead_features"), "#7A3E9D"))

# COMMAND ----------

import shap
# fe.score_batch looks the features up from the feature table by lead_id — the same path a nightly job (or serving) uses
open_ids = labels.filter("split = 'open'").select("lead_id")
scored = fe.score_batch(model_uri=f"models:/{LEAD_MODEL}@champion", df=open_ids, result_type="double").toPandas()
open_leads = scored.merge(spark.table(FEATURE_TABLE).select("lead_id", "created_date").toPandas(), on="lead_id")

def shap_by_feature(pipe, X):
    """SHAP values per feature for the gradient-boosted model (ordinal-encoded categories keep one column per feature)."""
    sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(pipe.named_steps["prep"].transform(X[X_cols]))
    return sv[1] if isinstance(sv, list) else sv
sv = shap_by_feature(lead_model.pipeline, open_leads)
NICE = {"company_size": "company size", "industry": "industry", "source": "lead source", "email_opens": "email opens", "web_visits": "web visits",
        "engagement_frequency": "engagement frequency", "days_since_last_activity": "days since last activity", "days_engaged": "days engaged in total",
        "demo_requested": "requested a demo", "event_attended": "attended an event", "icp_fit": "ideal-customer-profile fit"}
def reasons(i):
    row = open_leads.iloc[i]
    show = lambda c, v: ("yes" if v else "no") if c in ("demo_requested", "event_attended", "icp_fit") else v
    return [f"{'+' if sv[i][j] > 0 else '−'} {NICE[X_cols[j]]}: {show(X_cols[j], row[X_cols[j]])}" for j in np.argsort(-np.abs(sv[i]))[:3]]
scores = open_leads[["lead_id", "created_date", "company_size", "industry", "source", "demo_requested", "web_visits", "days_since_last_activity"]].copy()
scores["conversion_probability"] = open_leads.prediction.round(4)
scores["score_decile"] = (10 - pd.qcut(scores.conversion_probability.rank(method="first"), 10, labels=False)).astype(int)
scores["top_reasons"] = [reasons(i) for i in range(len(open_leads))]
scores["model_version"] = f"{LEAD_MODEL}@champion (v{LEAD_MODEL_VERSION})"
scores["talking_point"] = ""   # filled by ai_gen in step 2.3b
spark.createDataFrame(scores).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq("lead_scores"))
# Win rate by decile on the hold-out (what managers ask about)
bt = hold[["lead_id", "converted"]].merge(labels.filter("split = 'hold-out'").select("lead_id", "closed_won").toPandas(), on="lead_id")
bt["converted"] = bt.converted.astype(bool)
bt["conversion_probability"] = p_hold
bt["score_decile"] = (10 - pd.qcut(pd.Series(p_hold).rank(method="first"), 10, labels=False)).astype(int).values
spark.createDataFrame(bt).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq("lead_score_backtest"))
import matplotlib.pyplot as plt
importance = pd.DataFrame({"feature": [NICE[c] for c in X_cols], "mean_abs_shap": np.abs(sv).mean(axis=0).round(4)}).sort_values("mean_abs_shap")
with mlflow.start_run(run_id=LEAD_RUN_ID):   # attach the explanations to the @champion model's training run
    mlflow.log_table(importance, "shap/mean_abs_shap_open_leads.json")
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.barh(importance.feature, importance.mean_abs_shap, color="#FF3621")
    ax.set_title("What drives lead scores — mean |SHAP| on this week's open leads"); ax.set_xlabel("mean |SHAP value|"); fig.tight_layout()
    mlflow.log_figure(fig, "shap/feature_importance_open_leads.png"); plt.close(fig)
    shap.summary_plot(sv, open_leads[X_cols].apply(lambda c: c.astype("category").cat.codes if c.dtype == object else c),
                      feature_names=[NICE[c] for c in X_cols], show=False)
    mlflow.log_figure(plt.gcf(), "shap/beeswarm_open_leads.png"); plt.close("all")
print(f"Scored {len(scores)} open leads with {LEAD_MODEL}@champion (v{LEAD_MODEL_VERSION}) · SHAP importance and beeswarm logged to the model's run")

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO lead_scores s
# MAGIC USING (
# MAGIC   SELECT lead_id, ai_gen(concat('You are a sales coach. In ONE sentence (max 25 words), tell a rep how to open the call with this lead. ',
# MAGIC                                 'Company size ', company_size, ', ', industry, ', came from ', source, '. Why the model likes (+) or doubts (-) it: ',
# MAGIC                                 array_join(top_reasons, '; '), '. Do not mention models, scores or SHAP.')) AS talking_point
# MAGIC   FROM lead_scores WHERE score_decile <= 2
# MAGIC ) t ON s.lead_id = t.lead_id
# MAGIC WHEN MATCHED THEN UPDATE SET s.talking_point = t.talking_point
# MAGIC WHEN NOT MATCHED BY SOURCE THEN UPDATE SET s.talking_point = NULL;
# MAGIC
# MAGIC SELECT lead_id, company_size, industry, source, round(conversion_probability * 100, 1) AS conversion_pct, score_decile, top_reasons, talking_point
# MAGIC FROM lead_scores ORDER BY conversion_probability DESC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Win rate by score decile — does the score rank leads the way reps experience them?
# MAGIC SELECT score_decile, count(*) AS leads, round(avg(CAST(converted AS INT)) * 100, 1) AS conversion_rate_pct,
# MAGIC        round(avg(CAST(closed_won AS INT)) * 100, 1) AS win_rate_pct
# MAGIC FROM lead_score_backtest GROUP BY score_decile ORDER BY score_decile;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Win rate by score decile — does the score rank leads the way reps experience them?
# MAGIC SELECT score_decile, count(*) AS leads, round(avg(CAST(converted AS INT)) * 100, 1) AS conversion_rate_pct,
# MAGIC        round(avg(CAST(closed_won AS INT)) * 100, 1) AS win_rate_pct
# MAGIC FROM lead_score_backtest GROUP BY score_decile ORDER BY score_decile) SELECT `score_decile`,SUM(`win_rate_pct`) `column_9745038815` FROM q GROUP BY `score_decile`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4–2.5 Compare with the analyst path — `ai_predict_class`
# MAGIC The same features and the **same hold-out**, with the hold-out labels set to NULL so `ai_predict_class` scores them. `_confidence` is the probability of the **predicted** class, so to rank leads it's converted to the probability of **conversion**: `confidence` when the prediction is `true`, `1 − confidence` when it's `false`. Step 2.5 logs **AUC** and **top-decile lift** for both approaches — the `@champion` model and `ai_predict_class` — to the experiment.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE lead_predict_class COMMENT 'ai_predict_class conversion probability on the hold-out leads' AS
# MAGIC SELECT lead_id, actual_converted,
# MAGIC        CASE WHEN converted_prediction THEN converted_confidence ELSE 1 - converted_confidence END AS conversion_probability
# MAGIC FROM ai_predict_class(
# MAGIC   input => TABLE(
# MAGIC     SELECT f.lead_id, l.converted AS actual_converted, CASE WHEN l.split = 'training' THEN l.converted END AS converted,
# MAGIC            f.company_size, f.industry, f.source, f.email_opens, f.web_visits, f.engagement_frequency, f.days_since_last_activity, f.days_engaged,
# MAGIC            f.demo_requested, f.event_attended, f.icp_fit
# MAGIC     FROM lead_features f JOIN lead_labels l USING (lead_id) WHERE l.split IN ('training', 'hold-out')
# MAGIC   ),
# MAGIC   target_col   => 'converted',
# MAGIC   feature_cols => array('company_size', 'industry', 'source', 'email_opens', 'web_visits', 'engagement_frequency',
# MAGIC                         'days_since_last_activity', 'days_engaged', 'demo_requested', 'event_attended', 'icp_fit'),
# MAGIC   seed         => 42
# MAGIC )
# MAGIC WHERE actual_converted IS NOT NULL AND lead_id IN (SELECT lead_id FROM lead_labels WHERE split = 'hold-out');
# MAGIC SELECT count(*) AS holdout_leads_scored, round(avg(conversion_probability), 3) AS avg_conversion_probability FROM lead_predict_class;

# COMMAND ----------

pc = spark.table("lead_predict_class").toPandas()
AUC_PC, LIFT_PC = roc_auc_score(pc.actual_converted.astype(int), pc.conversion_probability), lift_top_decile(pc.actual_converted, pc.conversion_probability)
experiment()
with mlflow.start_run(run_name="2.5 lead conversion — ai_predict_class (analyst path)") as run:
    mlflow.log_params({"function": "ai_predict_class", "features": ", ".join(X_cols), "holdout_leads": len(pc)})
    mlflow.log_metrics({"holdout_auc": round(AUC_PC, 4), "holdout_top_decile_lift": round(LIFT_PC, 2)})
    show_run(run, "AI Function: lead conversion")
cmp = pd.DataFrame([("MLflow model @champion (feature table, UC registry, SHAP reasons)", round(AUC_MLFLOW, 3), round(LIFT_MLFLOW, 2)),
                    ("ai_predict_class (one SQL call, no saved model)", round(AUC_PC, 3), round(LIFT_PC, 2))],
                   columns=["approach", "holdout_auc", "top_decile_lift"])
display(cmp)
print(f"Hold-out AUC: @champion {AUC_MLFLOW:.3f} vs ai_predict_class {AUC_PC:.3f} — an analyst gets most of the way there in one SQL call. "
      "The MLflow model is the production path: a governed feature table, versioned in Unity Catalog with a @champion alias, explained with SHAP, servable for CRM write-back.")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Reverse-ETL stub: the rows a sync (e.g. a Lakeflow Job calling the Salesforce API, or a reverse-ETL partner) pushes to Lead.Score__c
# MAGIC CREATE OR REPLACE TABLE crm_lead_writeback AS
# MAGIC SELECT lead_id AS Id, round(conversion_probability * 100) AS Halcyon_Score__c, score_decile AS Score_Decile__c,
# MAGIC        array_join(top_reasons, ' | ') AS Score_Reasons__c, talking_point AS Talking_Point__c, model_version AS Model_Version__c,
# MAGIC        current_timestamp() AS synced_at
# MAGIC FROM lead_scores;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE lead_outreach AS
# MAGIC SELECT lead_id, company_size, industry, source, round(conversion_probability * 100, 1) AS conversion_pct, talking_point,
# MAGIC        ai_gen(concat('Draft a short first-touch sales email (subject line + 3-4 sentences, sign as "the Halcyon team") to a lead at a ', company_size,
# MAGIC                      '-employee ', industry, ' company that came from ', source, '. Angle: ', coalesce(talking_point, ''),
# MAGIC                      ' End with a specific ask for a 20-minute call. No placeholders in square brackets.')) AS email_draft
# MAGIC FROM lead_scores ORDER BY conversion_probability DESC LIMIT 20;
# MAGIC
# MAGIC SELECT lead_id, industry, source, conversion_pct, email_draft FROM lead_outreach ORDER BY conversion_pct DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.7 💬 Ask Genie — leads
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Who are the top 20 leads this week, and why?")

# COMMAND ----------

ask("What's the win rate by score decile?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.9 🛠 Try it yourself
# MAGIC In **step 2.2**, remove `"days_since_last_activity"` from `NUM` (and from the `feature_cols` list in **step 2.4**), then re-run **steps 2.2–2.5**. How much do AUC and top-decile lift drop? That's the value of engagement recency — and the experiment keeps both runs side by side.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Churn & upsell — which accounts do we save, which can we grow, before the quarter ends?
# MAGIC
# MAGIC | Step | What happens | Capability |
# MAGIC |---|---|---|
# MAGIC | 3.1–3.2 | Build the labels (churned, expanded) and features **as of 90 days before each renewal**, so nothing after the outcome leaks in | SQL → `renewal_features` |
# MAGIC | 3.3 | Predict churn and upsell — last year's renewals held out | `ai_predict_class` ×2 |
# MAGIC | 3.4 | Catch the warning sign: **usage falling while NPS still looks healthy** | `ai_detect_anomalies` on weekly active users |
# MAGIC | 3.5 | Explain **why ARR at risk is going up** | `ai_top_drivers` → `arr_drivers` |
# MAGIC | 3.6 | Sort accounts and act | deterministic SQL `CASE` buckets · `ai_gen` next-best action → `account_actions` |
# MAGIC | 3.7 | How good are the predictions? | **MLflow**: hold-out AUC, precision and recall, with ROC / precision-recall curves |

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW renewal_labels AS
# MAGIC WITH bm AS (   -- billing by master account and month (through account_xref)
# MAGIC   SELECT x.master_account_id AS account_id, b.month, sum(b.seats_purchased) AS seats, size(array_distinct(flatten(collect_list(b.products)))) AS n_products
# MAGIC   FROM billing b JOIN account_xref x ON x.source_system = 'billing' AND x.source_id = b.billing_account_id GROUP BY ALL
# MAGIC ), pre AS (    -- the month before the renewal
# MAGIC   SELECT r.account_id, r.renewal_date, b.seats AS seats_before, b.n_products AS products_before
# MAGIC   FROM contract_renewals r JOIN bm b ON b.account_id = r.account_id AND b.month = last_day(add_months(r.renewal_date, -1))
# MAGIC ), post AS (   -- the 6 months after it
# MAGIC   SELECT r.account_id, r.renewal_date, max(b.seats) AS seats_after, max(b.n_products) AS products_after
# MAGIC   FROM contract_renewals r JOIN bm b ON b.account_id = r.account_id AND b.month BETWEEN r.renewal_date AND add_months(r.renewal_date, 6)
# MAGIC   GROUP BY ALL
# MAGIC )
# MAGIC SELECT r.*,
# MAGIC   CASE WHEN r.renewed_arr IS NULL THEN NULL ELSE r.renewed_arr < r.arr END AS churned,
# MAGIC   CASE WHEN r.renewed_arr IS NULL OR r.renewal_date > add_months(current_date(), -6) THEN NULL
# MAGIC        ELSE coalesce(post.seats_after > pre.seats_before OR post.products_after > pre.products_before, false) END AS expanded
# MAGIC FROM contract_renewals r LEFT JOIN pre USING (account_id, renewal_date) LEFT JOIN post USING (account_id, renewal_date);
# MAGIC
# MAGIC SELECT CASE WHEN renewed_arr IS NULL THEN 'still to come' ELSE 'happened' END AS renewal, count(*) AS renewals,
# MAGIC        count_if(churned) AS churned, count_if(expanded) AS expanded, round(sum(arr)) AS arr
# MAGIC FROM renewal_labels GROUP BY ALL;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE renewal_features AS
# MAGIC WITH base AS (
# MAGIC   SELECT l.*, date_sub(l.renewal_date, 90) AS as_of, c.segment, c.industry,
# MAGIC          CASE WHEN l.renewal_date > current_date() THEN 'scoring'
# MAGIC               WHEN l.renewal_date > add_months(current_date(), -12) THEN 'hold-out' ELSE 'training' END AS split
# MAGIC   FROM renewal_labels l JOIN customer_360 c USING (account_id)
# MAGIC ), bm AS (
# MAGIC   SELECT x.master_account_id AS account_id, b.month, sum(b.seats_purchased) AS seats, avg(b.days_late) AS days_late,
# MAGIC          element_at(max_by(b.products, b.revenue), 1) AS primary_product
# MAGIC   FROM billing b JOIN account_xref x ON x.source_system = 'billing' AND x.source_id = b.billing_account_id GROUP BY ALL
# MAGIC ), u AS (
# MAGIC   SELECT b.account_id, b.renewal_date,
# MAGIC          avg(CASE WHEN u.week_start > date_sub(b.as_of, 56) THEN u.active_users END) AS au_8w,
# MAGIC          avg(CASE WHEN u.week_start <= date_sub(b.as_of, 56) THEN u.active_users END) AS au_prev_8w
# MAGIC   FROM base b JOIN account_weekly_usage u ON u.account_id = b.account_id AND u.week_start BETWEEN date_sub(b.as_of, 112) AND b.as_of
# MAGIC   GROUP BY ALL
# MAGIC ), bl AS (
# MAGIC   SELECT b.account_id, b.renewal_date, max_by(m.seats, m.month) AS seats, max_by(m.primary_product, m.month) AS primary_product,
# MAGIC          avg(CASE WHEN m.month >= add_months(b.as_of, -6) THEN m.days_late END) AS avg_days_late
# MAGIC   FROM base b JOIN bm m ON m.account_id = b.account_id AND m.month <= b.as_of GROUP BY ALL
# MAGIC ), tk AS (
# MAGIC   SELECT b.account_id, b.renewal_date, count(t.ticket_id) AS tickets_6m, count_if(t.severity = 'Sev 1') AS sev1_tickets_6m,
# MAGIC          count_if(t.theme = 'outage') AS outage_tickets_6m, count_if(t.theme = 'billing') AS billing_tickets_6m
# MAGIC   FROM base b LEFT JOIN tickets_resolved t ON t.master_account_id = b.account_id AND t.opened BETWEEN add_months(b.as_of, -6) AND b.as_of
# MAGIC   GROUP BY ALL
# MAGIC ), np AS (
# MAGIC   SELECT b.account_id, b.renewal_date, max_by(n.score, n.date) AS latest_nps,
# MAGIC          avg(CASE WHEN n.date >= add_months(b.as_of, -12) THEN CASE WHEN n.sentiment = 'negative' THEN 1.0 ELSE 0.0 END END) AS negative_sentiment_share
# MAGIC   FROM base b JOIN nps_sentiment n ON n.account_id = b.account_id AND n.date <= b.as_of GROUP BY ALL
# MAGIC )
# MAGIC SELECT b.account_id, b.renewal_date, b.arr, b.renewed_arr, b.churned, b.expanded, b.split, b.segment, b.industry, bl.primary_product,
# MAGIC        round(u.au_8w / nullif(u.au_prev_8w, 0) - 1, 3) AS usage_trend, round(u.au_8w / nullif(bl.seats, 0), 3) AS seat_utilization,
# MAGIC        tk.tickets_6m, tk.sev1_tickets_6m, tk.outage_tickets_6m, tk.billing_tickets_6m, np.latest_nps, round(np.negative_sentiment_share, 3) AS negative_sentiment_share,
# MAGIC        round(bl.avg_days_late, 1) AS avg_days_late
# MAGIC FROM base b JOIN u USING (account_id, renewal_date) LEFT JOIN bl USING (account_id, renewal_date)
# MAGIC LEFT JOIN tk USING (account_id, renewal_date) LEFT JOIN np USING (account_id, renewal_date)
# MAGIC WHERE u.au_prev_8w IS NOT NULL AND u.au_8w IS NOT NULL;
# MAGIC
# MAGIC SELECT split, count(*) AS renewals, round(avg(CAST(churned AS INT)) * 100, 1) AS churn_rate_pct, round(avg(CAST(expanded AS INT)) * 100, 1) AS expansion_rate_pct,
# MAGIC        round(sum(arr)) AS arr
# MAGIC FROM renewal_features GROUP BY ALL ORDER BY split;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Predict churn and upsell — `ai_predict_class`, twice
# MAGIC One call with target `churned`, one with target `expanded`. Both train on the older renewals and score two kinds of rows where the label is NULL: **last year's renewals (the hold-out — labels set to NULL on purpose, so step 3.7 can check them)** and **renewals still to come**.
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on **Predictive AI Functions** (Beta) on the **Previews** page.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE churn_predictions AS
# MAGIC SELECT account_id, renewal_date, split, actual AS actual_churned, churned_prediction,
# MAGIC        CASE WHEN churned_prediction THEN churned_confidence ELSE 1 - churned_confidence END AS churn_probability
# MAGIC FROM ai_predict_class(
# MAGIC   input => TABLE(
# MAGIC     SELECT account_id, renewal_date, split, churned AS actual, CASE WHEN split = 'training' THEN churned END AS churned,
# MAGIC            segment, industry, primary_product, usage_trend, seat_utilization, tickets_6m, sev1_tickets_6m, outage_tickets_6m,
# MAGIC            billing_tickets_6m, latest_nps, negative_sentiment_share, avg_days_late
# MAGIC     FROM renewal_features
# MAGIC   ),
# MAGIC   target_col   => 'churned',
# MAGIC   feature_cols => array('segment', 'industry', 'primary_product', 'usage_trend', 'seat_utilization', 'tickets_6m', 'sev1_tickets_6m',
# MAGIC                         'outage_tickets_6m', 'billing_tickets_6m', 'latest_nps', 'negative_sentiment_share', 'avg_days_late'),
# MAGIC   seed         => 42
# MAGIC )
# MAGIC WHERE split <> 'training';
# MAGIC
# MAGIC CREATE OR REPLACE TABLE expansion_predictions AS
# MAGIC SELECT account_id, renewal_date, split, actual AS actual_expanded, expanded_prediction,
# MAGIC        CASE WHEN expanded_prediction THEN expanded_confidence ELSE 1 - expanded_confidence END AS expansion_probability
# MAGIC FROM ai_predict_class(
# MAGIC   input => TABLE(
# MAGIC     SELECT account_id, renewal_date, CASE WHEN split = 'hold-out' AND expanded IS NULL THEN 'unobservable' ELSE split END AS split,
# MAGIC            expanded AS actual, CASE WHEN split = 'training' THEN expanded END AS expanded,
# MAGIC            segment, industry, primary_product, usage_trend, seat_utilization, tickets_6m, sev1_tickets_6m, outage_tickets_6m,
# MAGIC            billing_tickets_6m, latest_nps, negative_sentiment_share, avg_days_late
# MAGIC     FROM renewal_features
# MAGIC   ),
# MAGIC   target_col   => 'expanded',
# MAGIC   feature_cols => array('segment', 'industry', 'primary_product', 'usage_trend', 'seat_utilization', 'tickets_6m', 'sev1_tickets_6m',
# MAGIC                         'outage_tickets_6m', 'billing_tickets_6m', 'latest_nps', 'negative_sentiment_share', 'avg_days_late'),
# MAGIC   seed         => 42
# MAGIC )
# MAGIC WHERE split IN ('hold-out', 'scoring');
# MAGIC
# MAGIC SELECT 'churn' AS model, split, count(*) AS renewals, count_if(churned_prediction) AS predicted_true, round(avg(churn_probability), 3) AS avg_probability
# MAGIC FROM churn_predictions GROUP BY ALL
# MAGIC UNION ALL
# MAGIC SELECT 'expansion', split, count(*), count_if(expanded_prediction), round(avg(expansion_probability), 3) FROM expansion_predictions GROUP BY ALL
# MAGIC ORDER BY model, split;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Catch the warning sign — `ai_detect_anomalies`
# MAGIC The scariest accounts are the ones that **look** fine: NPS 9, no complaints — while their **weekly active users** quietly fall. `ai_detect_anomalies` scores every account's weekly usage (`group_col => 'account_id'`); keep accounts flagged for a **drop in the last 8 weeks** whose **latest NPS is 8 or higher** — the *"usage falling, NPS healthy"* accounts. Step 3.7 checks how many of the planted ones it caught.
# MAGIC
# MAGIC *Fallback: if the Predictive AI Functions preview isn't enabled, a rolling 12-week baseline runs instead (`method` column).*

# COMMAND ----------

run_on_sql_warehouse(r"""
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    -- Fallback: each week vs the same account's 12 weeks before (excluding the last 8), flag drops below mean - max(3 sd, 15%)
    CREATE OR REPLACE TABLE usage_anomalies AS
    WITH w AS (
      SELECT account_id, week_start, CAST(active_users AS DOUBLE) AS active_users,
             avg(active_users) OVER (PARTITION BY account_id ORDER BY week_start ROWS BETWEEN 20 PRECEDING AND 9 PRECEDING) AS mu,
             stddev(active_users) OVER (PARTITION BY account_id ORDER BY week_start ROWS BETWEEN 20 PRECEDING AND 9 PRECEDING) AS sd,
             row_number() OVER (PARTITION BY account_id ORDER BY week_start) AS n
      FROM account_weekly_usage WHERE week_start < CAST(date_trunc('WEEK', current_date()) AS DATE)
    )
    SELECT week_start, account_id, active_users, mu AS expected, mu - greatest(3 * sd, 0.15 * mu) AS lower_bound, mu + greatest(3 * sd, 0.15 * mu) AS upper_bound,
           least(1.0, abs(active_users - mu) / greatest(4 * sd, 0.2 * mu)) AS anomaly_score,
           n > 20 AND abs(active_users - mu) > greatest(3 * sd, 0.15 * mu) AS is_anomaly, 'rolling baseline (fallback)' AS method
    FROM w WHERE n > 20 AND week_start >= date_sub(current_date(), 182);
  END;

  CREATE OR REPLACE TABLE usage_anomalies AS
  SELECT a.*, 'ai_detect_anomalies' AS method
  FROM ai_detect_anomalies(
    TABLE(SELECT account_id, week_start, active_users FROM account_weekly_usage WHERE week_start < CAST(date_trunc('WEEK', current_date()) AS DATE)),
    time_col          => 'week_start',
    value_col         => 'active_users',
    group_col         => 'account_id',
    anomaly_threshold => anomaly_threshold,
    only_last_n       => 26
  ) a;
END;
""", anomaly_threshold=("DOUBLE", num("anomaly_threshold", 0.95)))
display(spark.sql("""
SELECT c.account_name, c.owner, c.segment, c.latest_nps, count(*) AS drop_weeks_last_8, round(min(a.active_users)) AS lowest_weekly_users,
       round(max(a.expected)) AS expected_weekly_users, c.next_renewal_date, round(c.revenue_12m) AS revenue_12m, any_value(a.method) AS method
FROM usage_anomalies a JOIN customer_360 c USING (account_id)
WHERE a.is_anomaly AND a.active_users < a.expected AND a.week_start >= date_sub(current_date(), 63) AND c.latest_nps >= 8
GROUP BY ALL HAVING count(*) >= 2
ORDER BY revenue_12m DESC"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Explain ARR at risk — `ai_top_drivers`
# MAGIC Is next quarter's churn risk the same problem as last quarter's, or something new? `ai_top_drivers` with `metric => 'arr'`, `aggregation => 'sum'` and `dimensions => array('segment', 'industry', 'primary_product')` compares:
# MAGIC * **Test group:** ARR on **next quarter's renewals at churn risk** (`ai_predict_class` churn probability ≥ 10% — see step 3.6 for why 10%)
# MAGIC * **Control group:** ARR that **actually churned last quarter**
# MAGIC
# MAGIC `primary_product` is derived as a string column first, since `products` is a list. The output shows which segments are driving the increase.

# COMMAND ----------

# ai_top_drivers runs on Databricks SQL — this cell sends the SQL below, exactly as written, to your SQL warehouse.
run_on_sql_warehouse(r"""
CREATE OR REPLACE TABLE arr_drivers_raw AS
SELECT * FROM ai_top_drivers(
  input => TABLE(
    SELECT arr, grp = 'next quarter, at churn risk' AS is_test, segment, industry, primary_product FROM (
      SELECT f.arr, 'next quarter, at churn risk' AS grp, f.segment, f.industry, f.primary_product
      FROM renewal_features f JOIN churn_predictions p USING (account_id, renewal_date)
      WHERE f.renewal_date BETWEEN next_q_start AND next_q_end AND p.churn_probability >= 0.10
      UNION ALL
      SELECT arr - renewed_arr, 'last quarter, churned', segment, industry, primary_product
      FROM renewal_features
      WHERE renewal_date >= last_q_start AND renewal_date < CAST(date_trunc('QUARTER', current_date()) AS DATE) AND churned)
  ),
  metric      => 'arr',
  is_test     => 'is_test',
  dimensions  => array('segment', 'industry', 'primary_product'),
  aggregation => 'sum'
);

CREATE OR REPLACE TABLE arr_drivers AS
SELECT array_join(contributor, ' & ') AS contributor, round(metric_control) AS churned_arr_last_quarter, round(metric_test) AS arr_at_risk_next_quarter,
       round(change) AS change_usd, round(relative_change * 100, 1) AS change_pct, round(support * 100, 1) AS share_pct
FROM arr_drivers_raw;
""", next_q_start=("DATE", "CAST(date_trunc('QUARTER', add_months(current_date(), 3)) AS DATE)"),
                     next_q_end=("DATE", "date_sub(add_months(CAST(date_trunc('QUARTER', add_months(current_date(), 3)) AS DATE), 3), 1)"),
                     last_q_start=("DATE", "add_months(CAST(date_trunc('QUARTER', current_date()) AS DATE), -3)"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM arr_drivers WHERE contributor <> 'all' ORDER BY change_usd DESC LIMIT 10;

# COMMAND ----------

# MAGIC %sql WITH q AS (SELECT * FROM arr_drivers WHERE contributor <> 'all' ORDER BY change_usd DESC LIMIT 10) SELECT `contributor`,SUM(`change_usd`) `column_5604153164` FROM q GROUP BY `contributor`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Sort accounts and act
# MAGIC A **deterministic SQL `CASE`** puts every upcoming renewal into one bucket, so the rules are easy to audit and change. Churn is rare (about 1 renewal in 9), so `ai_predict_class` seldom crosses 50% — the buckets act on the churn **probability** instead: on last year's renewals, a **10% risk threshold** flags churners at about three times the base rate (step 3.7 logs the exact precision and recall — `ai_predict_class` retrains on every call, so they move a little from run to run).
# MAGIC
# MAGIC | Bucket | Rule |
# MAGIC |---|---|
# MAGIC | **Save** | churn probability ≥ 10%, **or** the usage-drop flag from step 3.4 *(so Save ARR is larger than step 3.5's predicted-churn ARR, and it covers the next two quarters)* |
# MAGIC | **Expand** | expansion predicted **and** churn probability below 5% |
# MAGIC | **Nurture** | a borderline signal: churn probability 6–10%, or expansion probability 40–60% |
# MAGIC | **Healthy** | everything else |
# MAGIC
# MAGIC Then **`ai_gen`** writes a **next-best action** for each account from its `customer_360` brief. Everything lands in **`account_actions`**.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE account_actions AS
# MAGIC WITH drop_flag AS (
# MAGIC   SELECT DISTINCT a.account_id FROM usage_anomalies a JOIN customer_360 c USING (account_id)
# MAGIC   WHERE a.is_anomaly AND a.active_users < a.expected AND a.week_start >= date_sub(current_date(), 63) AND c.latest_nps >= 8
# MAGIC   GROUP BY a.account_id HAVING count(*) >= 2
# MAGIC ), j AS (
# MAGIC   SELECT c.account_id, c.account_name, c.owner, c.segment, c.industry, f.primary_product, f.renewal_date, f.arr,
# MAGIC          round(p.churn_probability, 3) AS churn_probability, round(e.expansion_probability, 3) AS expansion_probability,
# MAGIC          p.churned_prediction, e.expanded_prediction, d.account_id IS NOT NULL AS usage_drop_nps_healthy,
# MAGIC          c.products_not_owned, c.account_brief
# MAGIC   FROM renewal_features f
# MAGIC   JOIN customer_360 c USING (account_id)
# MAGIC   JOIN churn_predictions p USING (account_id, renewal_date)
# MAGIC   LEFT JOIN expansion_predictions e USING (account_id, renewal_date)
# MAGIC   LEFT JOIN drop_flag d USING (account_id)
# MAGIC   WHERE f.split = 'scoring'
# MAGIC ), b AS (
# MAGIC   SELECT *,
# MAGIC     CASE WHEN churn_probability >= 0.10 OR usage_drop_nps_healthy THEN 'Save'
# MAGIC          WHEN expanded_prediction AND churn_probability < 0.05 THEN 'Expand'
# MAGIC          WHEN churn_probability >= 0.06 OR expansion_probability BETWEEN 0.4 AND 0.6 THEN 'Nurture'
# MAGIC          ELSE 'Healthy' END AS bucket
# MAGIC   FROM j
# MAGIC )
# MAGIC SELECT account_id, account_name, owner, segment, industry, primary_product, renewal_date, arr, churn_probability, expansion_probability,
# MAGIC        usage_drop_nps_healthy, bucket, products_not_owned,
# MAGIC        ai_gen(concat('You are a sales manager. In 2 sentences, tell the account owner (address them as "you"; do not name them) the next-best action for this account before its renewal on ',
# MAGIC                      renewal_date, '. Bucket: ', bucket, CASE WHEN usage_drop_nps_healthy THEN ' (usage is falling even though NPS looks healthy)' ELSE '' END,
# MAGIC                      '. Churn probability ', churn_probability, ', expansion probability ', coalesce(expansion_probability, 0),
# MAGIC                      '. Products not yet owned: ', array_join(products_not_owned, ', '), '. Account brief: ', account_brief)) AS next_best_action
# MAGIC FROM b;
# MAGIC
# MAGIC SELECT bucket, count(*) AS accounts, round(sum(arr)) AS arr, count_if(usage_drop_nps_healthy) AS usage_drop_nps_healthy
# MAGIC FROM account_actions GROUP BY ALL ORDER BY arr DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (SELECT bucket, count(*) AS accounts, round(sum(arr)) AS arr, count_if(usage_drop_nps_healthy) AS usage_drop_nps_healthy
# MAGIC FROM account_actions GROUP BY ALL ORDER BY arr DESC) SELECT `bucket`,SUM(`arr`) `column_1893238791` FROM q GROUP BY `bucket`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT account_name, owner, renewal_date, round(arr) AS arr, churn_probability, usage_drop_nps_healthy, next_best_action
# MAGIC FROM account_actions WHERE bucket = 'Save' ORDER BY arr DESC LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7 Check accuracy in MLflow — hold-out AUC, precision and recall
# MAGIC Last year's renewals were held out (labels set to NULL) in step 3.3. Step 3.7 compares those predictions with what actually happened and logs **AUC, precision and recall** for both `ai_predict_class` runs — churn precision and recall at the 10% risk threshold the buckets use, expansion at the predicted label. It also checks how many of the planted *usage falling, NPS healthy* accounts step 3.4 caught. Each run's *Artifacts* tab has the ROC and precision-recall curves and the confusion matrix.
# MAGIC
# MAGIC *Going further — real-time scoring:* `ai_predict_class` doesn't save a model you could serve. If real-time churn scoring is part of the story, train the churn model in MLflow (as in section 2) and serve it with **Model Serving** — see *Build more*.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, precision_score
ch = spark.sql("SELECT actual_churned AS y, churn_probability >= 0.10 AS yhat, churn_probability AS p FROM churn_predictions WHERE split = 'hold-out' AND actual_churned IS NOT NULL").toPandas()   # at the 10% risk threshold (step 3.6)
ex = spark.sql("SELECT actual_expanded AS y, expanded_prediction AS yhat, expansion_probability AS p FROM expansion_predictions WHERE split = 'hold-out' AND actual_expanded IS NOT NULL").toPandas()
planted = {r.entity_id for r in spark.sql("SELECT entity_id FROM ground_truth WHERE anomaly_type = 'usage_falling_nps_healthy'").collect()}
caught = {r.account_id for r in spark.sql("SELECT account_id FROM account_actions WHERE usage_drop_nps_healthy").collect()}
experiment()
rows = []
for name, d in [("churn", ch), ("expansion", ex)]:
    auc = roc_auc_score(d.y.astype(int), d.p) if d.y.nunique() > 1 else float("nan")
    prec = precision_score(d.y.astype(int), d.yhat.astype(int), zero_division=0)
    with mlflow.start_run(run_name=f"3.7 ai_predict_class {name} — hold-out (last year's renewals)") as run:
        mlflow.log_params({"function": "ai_predict_class", "target": "churned" if name == "churn" else "expanded", "holdout": "renewals in the last 12 months"})
        mlflow.log_metrics({"holdout_auc": round(auc, 4), "holdout_precision": round(prec, 4), "holdout_renewals": len(d),
                            "holdout_recall": round(float((d.yhat.astype(int) & d.y.astype(int)).sum() / max(d.y.astype(int).sum(), 1)), 4),
                            "actual_rate_pct": round(100 * d.y.astype(int).mean(), 1)})
        if name == "churn":
            mlflow.log_metrics({"usage_drop_planted": len(planted), "usage_drop_caught": len(planted & caught)})
        if d.y.nunique() > 1:   # ROC and precision-recall curves + confusion matrix, as figures in the run's Artifacts tab
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            RocCurveDisplay.from_predictions(d.y.astype(int), d.p, ax=axes[0], name=f"{name} (ai_predict_class)")
            PrecisionRecallDisplay.from_predictions(d.y.astype(int), d.p, ax=axes[1], name=f"{name} (ai_predict_class)")
            ConfusionMatrixDisplay.from_predictions(d.y.astype(int), d.yhat.astype(int), ax=axes[2], display_labels=[f"no {name}", name], colorbar=False)
            axes[0].set_title("ROC curve"); axes[1].set_title("Precision-recall curve"); axes[2].set_title("Confusion matrix (hold-out)")
            fig.tight_layout(); mlflow.log_figure(fig, f"evaluation/{name}_holdout_curves.png"); plt.close(fig)
        show_run(run, f"AI Function: {name} hold-out")
    rec = float((d.yhat.astype(int) & d.y.astype(int)).sum() / max(d.y.astype(int).sum(), 1))
    rows.append((name, len(d), round(100 * d.y.astype(int).mean(), 1), round(auc, 3), round(prec, 3), round(rec, 3)))
display(pd.DataFrame(rows, columns=["prediction", "holdout_renewals", "actual_rate_pct", "holdout_auc", "holdout_precision", "holdout_recall"]))
print(f"Usage falling, NPS healthy: step 3.4 caught {len(planted & caught)} of the {len(planted)} planted accounts ({len(caught)} flagged in total)")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.8 💬 Ask Genie — churn & upsell
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("How much ARR is at risk next quarter, and which accounts are they?", before=("Section 3 (labels, features, predictions and buckets)", 131))

# COMMAND ----------

ask("Which accounts are best for upsell?")

# COMMAND ----------

ask("Why is ARR at risk going up?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.11 🛠 Try it yourself
# MAGIC In **step 3.6**, change the **Save** threshold from `churn_probability >= 0.10` to `>= 0.15` and re-run it (and **3.6b**). How much ARR leaves the *Save* list — and which accounts would you now miss? The buckets are plain SQL, so sales leadership can tune them without touching a model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wrap-up — put it in front of the business

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 The Sales Genie agent
# MAGIC The agent you've been asking since step 0.9 now covers the gold tables: **`customer_360`**, **`account_xref`**, **`lead_scores`**, **`account_actions`**, **`arr_drivers`** (plus `tickets_resolved`, `lead_score_backtest`, `lead_outreach` and `contract_renewals`). All AI results were written to tables **once**; Genie reads them and never re-runs a model or an LLM call per question.

# COMMAND ----------

ask("Which Enterprise accounts are in the Save bucket, who owns them, and what should each owner do this week?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 The Sales dashboard — one page
# MAGIC Step 4.2 builds (or updates) a one-page **AI/BI dashboard** with the Genie agent attached (*Ask Genie* button):
# MAGIC * **Account 360 view**
# MAGIC * **Lead leaderboard**
# MAGIC * **ARR at risk vs expansion pipeline**

# COMMAND ----------

DATASETS = {
    "accounts": f"""SELECT account_name, owner, segment, industry, revenue_12m, revenue_trend_pct, active_users, seat_utilization, usage_trend_pct,
                           tickets_90d, latest_nps, next_renewal_date FROM {fq('customer_360')}""",
    "leads": f"""SELECT lead_id, industry, company_size, source, round(conversion_probability * 100, 1) AS conversion_pct, score_decile, talking_point
                 FROM {fq('lead_scores')} ORDER BY conversion_probability DESC LIMIT 25""",
    "renewals": f"""SELECT date_trunc('MONTH', renewal_date) AS renewal_month, bucket, round(sum(arr)) AS arr FROM {fq('account_actions')}
                    WHERE bucket IN ('Save', 'Expand') GROUP BY ALL""",
    "kpis": f"""SELECT (SELECT round(sum(arr)) FROM {fq('account_actions')} WHERE bucket = 'Save') AS arr_at_risk,
                       (SELECT round(sum(arr)) FROM {fq('account_actions')} WHERE bucket = 'Expand') AS expansion_pipeline_arr,
                       (SELECT count(*) FROM {fq('lead_scores')} WHERE score_decile = 1) AS top_decile_leads,
                       (SELECT count(*) FROM {fq('customer_360')}) AS accounts""",
}
WIDGETS = [
    text_widget("# Sales — customer 360, leads & renewals\nOne row per customer (identity-resolved across CRM, billing and support), the leads worth calling this week, "
                "and next two quarters of renewals sorted into Save / Expand / Nurture / Healthy. Built on Databricks AI Functions and MLflow. Use **Ask Genie** for anything else.", 0, 0, 12, 2),
    counter_widget("kpis", "arr_at_risk", "ARR at risk — Save bucket ($)", 0, 2, 3),
    counter_widget("kpis", "expansion_pipeline_arr", "Expansion pipeline — Expand bucket ($)", 3, 2, 3),
    counter_widget("kpis", "top_decile_leads", "Top-decile open leads", 6, 2, 3),
    counter_widget("kpis", "accounts", "Master accounts", 9, 2, 3),
    bar_widget("renewals", "renewal_month", "arr", "bucket", "ARR at risk vs expansion pipeline, by renewal month", 0, 5, 6, 7),
    table_widget("leads", ["lead_id", "industry", "company_size", "source", "conversion_pct", "talking_point"], "Lead leaderboard (MLflow model)", 6, 5, 6, 7),
    table_widget("accounts", ["account_name", "owner", "segment", "industry", "revenue_12m", "revenue_trend_pct", "seat_utilization", "usage_trend_pct",
                              "tickets_90d", "latest_nps", "next_renewal_date"], "Account 360", 0, 12, 12, 8),
]
DASHBOARD_URL = publish_dashboard(DASHBOARD_NAME, DATASETS, WIDGETS)
displayHTML(button("Open the Sales dashboard", DASHBOARD_URL) + button("Open the Genie agent", GENIE_URL, "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Lineage and the MLflow trust layer
# MAGIC Every table-to-table hop is recorded in **Unity Catalog** — open a card → **Lineage** to follow, for example, *`billing` + `crm_accounts` → `match_candidates` (`ai_similarity`, `ai_classify`) → `account_xref` → `customer_360` → `account_actions`*. The **MLflow experiment** (traces stored in Unity Catalog) holds the proof: **matching precision and recall** on the `identity_resolution_eval` dataset, the lead model **`@champion` vs `ai_predict_class`**, and **churn and upsell hold-out accuracy**. The lead model is registered in Unity Catalog with its `@champion` alias, and the feature table `lead_features` shows which model reads it.

# COMMAND ----------

show_uc_cards(roles={"Feature table", "Silver", "Gold", "ai_similarity · ai_classify", "ai_enrich", "AI Function task", "ai_analyze_sentiment", "MLflow model",
                     "ai_gen", "ai_predict_class", "ai_detect_anomalies", "ai_top_drivers"}, title="Silver, gold, model and AI Function tables")

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=[experiment().experiment_id], order_by=["start_time DESC"])
cols = [c for c in runs.columns if c in ("metrics.precision_pct", "metrics.recall_pct", "metrics.ticket_theme_accuracy_pct", "metrics.holdout_auc",
                                         "metrics.holdout_top_decile_lift", "metrics.holdout_precision", "metrics.holdout_recall", "metrics.usage_drop_caught")]
display(runs[["tags.mlflow.runName"] + cols].rename(columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.runName", "run")))
displayHTML(button("Open the MLflow experiment", f"{HOST}/ml/experiments/{experiment().experiment_id}{ORG}")
            + button("Open the lead model in Unity Catalog", f"{HOST}/explore/data/models/{CATALOG}/{SCHEMA}/lead_conversion_model{ORG}", "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Executive summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'Customer 360' AS play, 'Master accounts (from CRM + billing + support records)' AS metric,
# MAGIC        concat(count(DISTINCT master_account_id), ' (from ', count(*), ' records)') AS value FROM account_xref
# MAGIC UNION ALL SELECT 'Customer 360', 'CRM duplicates merged', CAST(count_if(source_system = 'crm' AND match_method <> 'self') AS STRING) FROM account_xref
# MAGIC UNION ALL SELECT 'Customer 360', 'Accounts enriched by ai_enrich', CAST(count(*) AS STRING) FROM account_enrichment
# MAGIC UNION ALL SELECT 'Customer 360', 'Accounts that grew usage but not spend', CAST(count_if(usage_trend_pct > 10 AND revenue_trend_pct <= 0) AS STRING) FROM customer_360
# MAGIC UNION ALL SELECT 'Lead scoring', 'Open leads scored', CAST(count(*) AS STRING) FROM lead_scores
# MAGIC UNION ALL SELECT 'Lead scoring', 'Win rate, top decile vs overall (hold-out)',
# MAGIC        concat(round(avg(CASE WHEN score_decile = 1 THEN CAST(closed_won AS INT) END) * 100, 1), '% vs ', round(avg(CAST(closed_won AS INT)) * 100, 1), '%') FROM lead_score_backtest
# MAGIC UNION ALL SELECT 'Churn & upsell', 'ARR at risk, next two quarters (Save)', format_number(sum(CASE WHEN bucket = 'Save' THEN arr END), 0) FROM account_actions
# MAGIC UNION ALL SELECT 'Churn & upsell', 'Expansion pipeline (Expand)', format_number(sum(CASE WHEN bucket = 'Expand' THEN arr END), 0) FROM account_actions
# MAGIC UNION ALL SELECT 'Churn & upsell', 'Usage falling, NPS healthy', CAST(count_if(usage_drop_nps_healthy) AS STRING) FROM account_actions
# MAGIC UNION ALL SELECT 'Churn & upsell', 'Top ARR-at-risk driver (ai_top_drivers)', max_by(contributor, change_usd) FROM arr_drivers WHERE contributor <> 'all';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Your turn — ask anything, then build more
# MAGIC **Ask your own question:** pick one in the **`question`** widget — or type anything in **`custom_question`** — and re-run **step 6.1**. Click **Continue this conversation in Genie** to drill down with follow-ups and charts.

# COMMAND ----------

question = dbutils.widgets.get("custom_question").strip() or dbutils.widgets.get("question")
ask(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 🚀 Build more — ideas to take this further
# MAGIC | Idea | How | Effort |
# MAGIC |---|---|---|
# MAGIC | **Live Salesforce** instead of CSV | **Lakeflow Connect for Salesforce** → the same `crm_*` / `leads` tables, incrementally | 30 min |
# MAGIC | **Real-time churn scoring** | Train the churn model in MLflow like step 2.2 and serve it with **Model Serving**; call it from the CRM when a ticket is opened | 1 hr |
# MAGIC | **AutoML** for the lead model | Run the AutoML Python API on a Databricks Runtime ML cluster on `lead_features`; it logs every trial to the same experiment and you register the best one | 30 min |
# MAGIC | **Push scores to Salesforce** | A Lakeflow Job (or a reverse-ETL partner) that upserts `crm_lead_writeback` into `Lead` nightly | 1 hr |
# MAGIC | **Ground `ai_enrich` in your own data** | Add an AI Search index of your account plans / 10-Ks as a `vector_search` knowledge source (step 1.3) | 30 min |
# MAGIC | **Route tickets** | Add a `route` label set or `ai_decide` to the ticket job — send outages for Enterprise accounts straight to the CSM | 15 min |
# MAGIC | **Unpause the ticket job** | Job page → Schedule → *Unpause* — new tickets are classified every hour | 1 min |
# MAGIC | **Share Genie** with the business | Open the Genie agent → **Share** → add your sales, CS and RevOps groups | 2 min |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. What you just saw — and where to take it next
# MAGIC
# MAGIC * **One view of the customer.** CRM, billing, support, product and NPS — joined into one governed `customer_360` in Unity Catalog, even though no two systems shared an account ID. Rules where they're exact, `ai_similarity` where they're not, `ai_classify` for the borderline — and MLflow proves the precision.
# MAGIC * **AI Functions turned raw text into signals.** `ai_enrich` filled in firmographics with its reasoning, an `ai_classify` **AI Function task** sorts every ticket as it arrives, `ai_analyze_sentiment` reads NPS, `ai_summarize` writes the account brief and `ai_gen` drafts the next move.
# MAGIC * **MLflow handles the model that goes back into the CRM.** A governed feature table, the model registered in Unity Catalog and scored by its `@champion` alias with SHAP reasons — and `ai_predict_class` beside it, getting most of the way there in one SQL call.
# MAGIC * **Churn and upsell, before the quarter ends.** `ai_predict_class` ×2, `ai_detect_anomalies` for the silent usage drops and `ai_top_drivers` for *why* — sorted into auditable buckets with a next-best action each.
# MAGIC * **Genie and a dashboard put it in front of the business.** Reps and managers ask in plain English; Genie reads the precomputed tables.
# MAGIC
# MAGIC **Production path**
# MAGIC 1. **Connect the real systems** — Lakeflow Connect for Salesforce, your billing and support desk feeds; swap the views in *Bring your own data* (step 0.11).
# MAGIC 2. **Schedule it** — unpause the ticket job, run sections 1–3 nightly as a Lakeflow Job, and sync `crm_lead_writeback` to Salesforce.
# MAGIC 3. **Share the Genie agent and the dashboard** with sales, customer success and RevOps.
# MAGIC 4. **Graduate where needed** — serve the churn model in real time with Model Serving.
# MAGIC
# MAGIC **Docs:** [ai_similarity](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_similarity) · [ai_classify](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_classify) · [ai_enrich](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_enrich) · [ai_predict_class](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_predict_class) · [ai_detect_anomalies](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_detect_anomalies) · [ai_top_drivers](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_top_drivers) · [AI Function tasks](https://docs.databricks.com/aws/en/jobs/ai-function-task) · [MLflow](https://docs.databricks.com/aws/en/mlflow/) · [Lakeflow Connect for Salesforce](https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/salesforce) · [Genie](https://docs.databricks.com/aws/en/genie/)
