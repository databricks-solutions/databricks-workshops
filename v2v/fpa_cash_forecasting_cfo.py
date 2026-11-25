# Databricks notebook source

# MAGIC %md
# MAGIC # FP&A Variance, Cash Forecasting & Revenue Assurance for the Office of the CFO
# MAGIC **For finance leaders:** one governed finance foundation that answers the three questions every CFO asks — *why did we miss plan, what will our cash position be, and where are we leaving revenue on the table?* AI Functions do the heavy lifting in SQL, MLflow shows you can trust the numbers, and Genie and an AI/BI dashboard put the answers in front of the business.
# MAGIC
# MAGIC **Industry:** Cross-industry — any company that plans, bills and collects
# MAGIC
# MAGIC Every month-end close ends with the same scramble: *Which cost centers blew the budget, and why? Will we have enough cash in week 8? Are we billing what our contracts say?* The answers live in the ledger, the bank feed, the billing system and a folder of signed contract PDFs nobody reads twice.
# MAGIC This notebook connects all of them in **Unity Catalog** and answers each question **in SQL, with no ML pipelines to build** — then proves the answers with **MLflow** and hands them to the business through **Genie**.
# MAGIC
# MAGIC | Section | Question | Databricks capability |
# MAGIC |---|---|---|
# MAGIC | **1 · FP&A** | Why did we miss plan last quarter, and where are we heading? | `ai_top_drivers` (×2), `ai_detect_anomalies`, `ai_forecast` v2 with a future covariate, `ai_query` commentary · **MLflow** backtest |
# MAGIC | **2 · Cash** | What will our cash position be over the next 13 weeks, and who is putting it at risk? | `ai_predict_value`, `ai_detect_anomalies`, `ai_forecast` (weekly), a Unity Catalog **SQL table function** for what-ifs, `ai_gen` · **MLflow** pay-date accuracy |
# MAGIC | **3 · Revenue assurance** | Where are we leaving revenue on the table? | **Lakeflow Job** with native **AI Function tasks** (`ai_parse_document` → `ai_extract`), SQL rules, an **MLflow Isolation Forest** in the UC registry, `ai_detect_anomalies`, `ai_decide`, `ai_summarize` · **MLflow** extraction accuracy |
# MAGIC | **4 · Wrap-up** | How does the business use this every day? | **Genie** agent, one-page **AI/BI dashboard**, **lineage**, the **MLflow** trust layer |
# MAGIC | 💬 | After sections 0–3: **ask Genie** the same question in plain English — no SQL | **Genie** agent on Unity Catalog |
# MAGIC | 🛠 | After sections 0–3: a **Try it yourself** exercise, and a *Build more* list at the end | |
# MAGIC
# MAGIC ---
# MAGIC ### How to run
# MAGIC 1. Keep the default **Serverless** compute. Steps that use `ai_forecast`, `ai_detect_anomalies` or `ai_top_drivers` (which run on Databricks SQL) automatically send their SQL to a **Serverless or Pro SQL warehouse** in your workspace (the notebook finds one for you).
# MAGIC 2. A **workspace admin** enables these on the **Previews** page (profile menu → *Previews*): **Predictive AI Functions** (Beta — `ai_forecast` v2, `ai_detect_anomalies`, `ai_top_drivers`, `ai_predict_value`), **Lakeflow AI Function Task** (Beta — the contract-reading job), and the **AI Functions** previews for `ai_decide` and `ai_gen`. **If any AI Function cell fails, this is the first thing to check.**
# MAGIC 3. Run **S.0–S.3** and set the widgets (`catalog`, `schema`, …) to a location where you can create tables. Then either **Run all** (about 45 minutes — mostly the contract-reading job, which starts in the background in step 0.8), or — better for learning — run section by section: each section ends with a **💬 Ask Genie** cell and a **🛠 Try it yourself** exercise.
# MAGIC 4. **Every section and cell is numbered** (S.1, 0.1, 1.1 …) — look for the number in each cell's title.
# MAGIC 5. **New to some of the code?** Ask **Genie Code** to explain any cell: click into it, open Genie Code and type `/explain` (or ask *"what does this cell do?"*).
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews listed in step 2 above. Then re-run the failed cell. The `ai_forecast` and `ai_detect_anomalies` steps also fall back automatically (to `ai_forecast` v1 and a rolling-baseline detector) so the rest of the notebook keeps running — check the `method` / `model_version` column.
# MAGIC
# MAGIC > **No data? No problem.** Section 0 generates three years of a fictitious company's ledger, plan, headcount, billing, bank and AP data, plus **200 signed contract PDFs** — with problems planted for the AI to find and an answer key (`ground_truth`) so MLflow can score the results. **Have your own data?** See *"Bring your own data"* (step 0.12).
# MAGIC
# MAGIC **Terms used in this notebook:** **FP&A** = financial planning & analysis · **DSO / DPO** = days sales / payables outstanding · **true-up** = billing usage above a contract's minimum commitment · **leakage** = revenue you were entitled to but didn't bill · **MAPE / MAE** = mean absolute (percentage) error · a **volume** is a Unity Catalog folder for files (here, the contract PDFs) · **VARIANT** = semi-structured JSON column type (what the AI Function tasks return).
# MAGIC
# MAGIC *All companies, people, contracts and figures in this notebook are fictitious.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## S. Setup — choose where to write the demo tables
# MAGIC Four short setup steps. The **widgets** (the notebook's parameters) drive everything below.
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | **S.0** | Installs the Python libraries for the MLflow steps (MLflow 3, Feature Engineering client, scikit-learn, Faker) in one go and restarts Python |
# MAGIC | **S.1** | Creates the widgets |
# MAGIC | **S.2** | Creates / selects the catalog, the `finance` schema and the `contracts` volume, and turns the widget values into SQL session variables |
# MAGIC | **S.3** | Loads the Python helper functions used later (Unity Catalog links, SQL warehouse runner, Genie, MLflow, dashboard) |
# MAGIC
# MAGIC | Widget | Meaning | Default |
# MAGIC |---|---|---|
# MAGIC | `catalog` / `schema` | Where tables are created in **Unity Catalog**. The catalog is shared with the *Sales* notebook (schema `sales`); both create it with `IF NOT EXISTS`, so either can run first. If you can't create catalogs, the current catalog is used | `demo_cfo_sales` / `finance` |
# MAGIC | `anomaly_threshold` | 0–1. Lower = more sensitive anomaly flagging | `0.95` |
# MAGIC | `discount_band_pct` | Reps may bill up to this % below the signed price without deal-desk approval; more than that is a rule hit | `7` |
# MAGIC | `dso_shift_days` | What-if for the 13-week cash forecast: collect every receipt this many days earlier (negative) or later | `-1` |
# MAGIC | `llm_endpoint` | Pay-per-token Foundation Model endpoint `ai_query` uses for the variance commentary | `databricks-claude-sonnet-4-5` |
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

# MAGIC %pip install -q --upgrade databricks-feature-engineering "mlflow[databricks]>=3.9" "scikit-learn>=1.5" shap matplotlib faker "databricks-sdk>=0.70"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Widgets — they appear in the bar at the top of the notebook, or in the left-hand Parameters panel
GENIE_QUESTIONS = [
    "Why did opex miss plan last quarter?",
    "Which cost centers are over budget?",
    "What's the revenue forecast for next quarter?",
    "What's our projected cash in week 8?",
    "Which customers are pushing DSO up?",
    "What happens to cash if DSO improves by 1 day?",
    "How much revenue leakage did we find?",
    "Which customers have the biggest billing gaps?",
    "Which contracts don't match what we're billing?",
]
dbutils.widgets.text("catalog", "demo_cfo_sales", "1. Catalog")
dbutils.widgets.text("schema", "finance", "2. Schema")
dbutils.widgets.dropdown("anomaly_threshold", "0.95", ["0.90", "0.95", "0.99"], "3. Anomaly sensitivity (lower = more alerts)")
dbutils.widgets.dropdown("discount_band_pct", "7", ["3", "5", "7", "10"], "4. Allowed discount band (%)")
dbutils.widgets.dropdown("dso_shift_days", "-1", ["-5", "-3", "-1", "0", "1", "3", "5", "10"], "5. What-if: collect N days earlier (-) / later (+)")
dbutils.widgets.dropdown("llm_endpoint", "databricks-claude-sonnet-4-5",
                         ["databricks-claude-sonnet-4-5", "databricks-claude-sonnet-4-6", "databricks-meta-llama-3-3-70b-instruct", "databricks-gpt-oss-120b"],
                         "6. LLM for ai_query")
dbutils.widgets.dropdown("question", GENIE_QUESTIONS[4], GENIE_QUESTIONS, "7. Ask Genie (any time)")
dbutils.widgets.text("custom_question", "", "8. ...or type your own Genie question")

# COMMAND ----------

# Changed a widget? Re-run this cell (S.2) and the helpers (S.3), then the cells you want to refresh.
cat = dbutils.widgets.get("catalog").strip() or "demo_cfo_sales"
sch = dbutils.widgets.get("schema").strip() or "finance"
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS `{cat}` COMMENT 'Office of the CFO & Sales AI Functions demos (synthetic data)'")
except Exception as e:
    fallback = spark.sql("SELECT current_catalog()").first()[0]
    print(f"⚠️ You don't have permission to create catalog '{cat}' — using the current catalog '{fallback}' instead.")
    cat = fallback
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{cat}`.`{sch}` COMMENT 'Office of the CFO: FP&A, cash and revenue assurance (synthetic data)'")
spark.sql(f"USE `{cat}`.`{sch}`")
spark.sql("CREATE VOLUME IF NOT EXISTS contracts COMMENT 'Signed customer order forms (PDF), one per contract'")

num = lambda name, default: float(dbutils.widgets.get(name) or default)
spark.sql(f"DECLARE OR REPLACE VARIABLE anomaly_threshold DOUBLE DEFAULT {num('anomaly_threshold', 0.95)}")
spark.sql(f"DECLARE OR REPLACE VARIABLE discount_band_pct DOUBLE DEFAULT {num('discount_band_pct', 7)}")
spark.sql(f"DECLARE OR REPLACE VARIABLE dso_shift_days INT DEFAULT {int(num('dso_shift_days', -1))}")
spark.sql(f"DECLARE OR REPLACE VARIABLE llm_endpoint STRING DEFAULT '{dbutils.widgets.get('llm_endpoint')}'")
# The calendar the whole notebook reports on: last closed month, last full quarter, and the quarter before it
spark.sql("DECLARE OR REPLACE VARIABLE lme DATE DEFAULT last_day(add_months(current_date(), -1))")
spark.sql("DECLARE OR REPLACE VARIABLE lfq_end DATE DEFAULT date_sub(date_trunc('QUARTER', current_date()), 1)")
spark.sql("DECLARE OR REPLACE VARIABLE lfq_start DATE DEFAULT CAST(date_trunc('QUARTER', lfq_end) AS DATE)")
spark.sql("DECLARE OR REPLACE VARIABLE pq_start DATE DEFAULT add_months(lfq_start, -3)")
display(spark.sql("""SELECT current_catalog() AS catalog, current_schema() AS schema, lme AS last_closed_month,
       concat('Q', quarter(lfq_end), ' ', year(lfq_end)) AS last_full_quarter, lfq_start, lfq_end,
       anomaly_threshold, discount_band_pct, dso_shift_days, llm_endpoint"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### S.3 Load the helper functions
# MAGIC Step S.3 **doesn't create any tables or change any data** — it only defines Python helpers the rest of the notebook calls, so later cells stay short and readable:
# MAGIC
# MAGIC | Helper | Used in | What it does |
# MAGIC |---|---|---|
# MAGIC | `WorkspaceClient()`, `CATALOG`, `SCHEMA`, `CONTRACTS` | everywhere | Connects to this workspace with **your** identity and remembers where the demo tables and the `contracts` volume live (from S.2) |
# MAGIC | `pick_warehouse()` / `run_on_sql_warehouse()` | steps 1.3, 1.4, 1.6, 1.9, 2.4, 2.5, 3.7 | Finds a **Serverless or Pro SQL warehouse** and sends `ai_top_drivers` / `ai_forecast` / `ai_detect_anomalies` SQL to it *exactly as written* (these functions run on Databricks SQL), passing in your widget values |
# MAGIC | `uc_url()`, `button()`, `show_uc_cards()` | step 0.7 and later | Builds clickable **Unity Catalog** links and the table cards (layer · rows · sample · lineage) |
# MAGIC | `ask_genie()`, `show_genie_answer()` — and `ask()`, defined with the agent in step 0.10 | every 💬 Ask Genie step | Asks the **Genie agent** a question through the Genie API and renders the answer, the SQL Genie wrote and a *Continue in Genie* link |
# MAGIC | `experiment()`, `show_run()` | steps 1.10, 2.8, 3.6, 3.10 | Opens the notebook's **MLflow experiment** — created with **Unity Catalog trace storage** (traces land in UC tables), reused only if it's active — and shows a link to each run |
# MAGIC | `eval_dataset()`, `set_alias()` | steps 3.6, 3.10 | Registers a versioned **evaluation dataset in Unity Catalog** and points the model's **`@champion`** alias at a version |
# MAGIC | `create_or_update_job()`, `show_task()`, `wait_for_run()`, `run_job()`, `publish_dashboard()` | steps 0.8, 3.1, 3.11, 4.2 | Creates the Lakeflow Job, renders an AI Function task definition, waits for / runs the job and builds the AI/BI dashboard |
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
CONTRACTS = f"/Volumes/{CATALOG}/{SCHEMA}/contracts"
JOB_NAME = f"Contract reading pipeline ({CATALOG}.{SCHEMA})"
DASHBOARD_NAME = f"Office of the CFO — plan, cash & leakage ({SCHEMA})"
EXPERIMENT_NAME = f"{NB_FOLDER}/Office of the CFO — AI Functions trust layer ({SCHEMA})"

# table -> (layer shown on the card, description, include in the Genie agent?)
TABLE_INFO = {
    "gl_actuals": ("Bronze · source", "General ledger: monthly actuals by entity, cost center, account and vendor", False),
    "budget_plan": ("Bronze · source", "Monthly plan by cost center and account (incl. next quarter)", True),
    "headcount": ("Bronze · source", "Employees with cost center, start / end date and salary", False),
    "headcount_plan": ("Bronze · source", "Planned headcount by cost center and month (incl. next quarter)", False),
    "customers": ("Bronze · source", "Customers with segment, industry and payment terms", True),
    "vendors": ("Bronze · source", "Vendor master", True),
    "bank_transactions": ("Bronze · source", "Operating bank account: every receipt and payment", False),
    "ar_invoices": ("Bronze · source", "Customer invoices with billed quantity and price, due and paid dates", False),
    "ap_bills": ("Bronze · source", "Vendor and payroll bills, incl. scheduled future bills", False),
    "contracts": ("Bronze · source", "Contracts as keyed into the billing system", False),
    "usage": ("Bronze · source", "Daily product usage per customer and SKU", False),
    "price_list": ("Bronze · source", "List price per SKU", False),
    "ground_truth": ("Answer key", "Every planted anomaly and the true contract terms — what MLflow scores against", False),
    "parsed_contracts": ("AI Function task", "parse_contracts task output: ai_parse_document per PDF", False),
    "contract_terms_extracted": ("AI Function task", "extract_contract_terms task output: ai_extract per contract", False),
    "contract_terms": ("Silver", "Extracted contract terms, one row per contract, with confidence", False),
    "invoice_payment_features": ("Silver", "Invoice features for pay-date prediction (no leakage)", False),
    "actuals_vs_plan": ("Gold", "Actuals vs plan with variance by month, cost center and account", True),
    "variance_drivers": ("ai_top_drivers", "What explains last quarter's opex miss", True),
    "opex_anomalies": ("ai_detect_anomalies", "Monthly opex per cost center: expected, bounds, anomaly flag", True),
    "forecast_next_qtr": ("ai_forecast", "Next quarter's revenue and opex with 90% intervals", True),
    "forecast_backtest": ("ai_forecast", "Backtest: forecast with history cut 3 months early vs actuals", False),
    "variance_commentary": ("ai_query", "3-sentence variance note per over-budget cost center", True),
    "working_capital": ("Gold", "DSO, DPO and AR aging", True),
    "payment_predictions": ("ai_predict_value", "Predicted days-to-pay for open, scheduled and hold-out invoices", False),
    "expected_receipts": ("ai_predict_value", "Every open / scheduled invoice with its expected pay date", True),
    "late_payer_anomalies": ("ai_detect_anomalies", "Monthly days-to-pay per customer with anomaly flags", True),
    "cash_forecast_13wk": ("Gold", "13-week cash: bottom-up vs top-down ai_forecast, with intervals", True),
    "collections_outreach": ("ai_gen", "Drafted collections email for the top 10 late payers", True),
    "contract_to_cash": ("Gold", "Every invoice vs signed contract terms and usage, with rule flags and $ leakage", True),
    "invoice_outlier_features": ("Feature table", "Governed feature table (PK invoice_id): leakage features for invoices that passed the rules", False),
    "invoice_outlier_scores": ("MLflow model", "Isolation Forest (@champion) score for every invoice that passed the rules", False),
    "price_drift_anomalies": ("ai_detect_anomalies", "Monthly billed-to-contract price ratio per customer", False),
    "leakage_exceptions": ("ai_decide · ai_summarize", "Every leakage case with $ at risk, verdict and case note", True),
}
ROLE_COLOR.update({"Bronze · source": "#5A6F77", "Silver": "#2272B4", "Gold": "#00A972", "Answer key": "#98102A", "AI Function task": "#1B3139", "MLflow model": "#0B2026", "Feature table": "#2272B4"})
print(f"Writing to {CATALOG}.{SCHEMA} · contracts volume {CONTRACTS} · SQL warehouse for ai_forecast / ai_detect_anomalies / Genie: '{WAREHOUSE.name}'")
displayHTML(button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139") + button("Open the contracts volume", volume_url("contracts"), "#5A6F77"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Generate synthetic data *(skip 0.1–0.5 if you bring your own)*
# MAGIC
# MAGIC **Halcyon Group** (fictitious) sells a data platform — compute units, API calls, analytics seats and storage — to 200 customers across financial services, healthcare, life sciences and manufacturing. Step 0.1 writes **36 months** of its finance data (36, not 24, because `ai_detect_anomalies` uses the first 8 points of each series as context only; every `period` is a **month-end date**, matching `frequency => 'ME'` in `ai_forecast`):
# MAGIC
# MAGIC | Table | What's in it |
# MAGIC |---|---|
# MAGIC | `gl_actuals`, `budget_plan` | Monthly ledger and plan by entity, cost center, account and vendor |
# MAGIC | `headcount`, `headcount_plan` | ~400 employees; planned headcount covers the forecast quarter too |
# MAGIC | `customers`, `contracts`, `price_list`, `usage` | Customers, the contract terms *as keyed into billing*, daily usage |
# MAGIC | `ar_invoices`, `ap_bills`, `bank_transactions` | Monthly invoices with billed qty / price, vendor & payroll bills (incl. scheduled ones), the operating bank account |
# MAGIC
# MAGIC Step 0.2 writes **200 signed contract PDFs** (three different layouts) to the `contracts` volume — each states the SKU, price, minimum commitment, true-up clause and end date. **About 5% say something different from the `contracts` table.** Step 0.3 has `ai_gen` write the free-text AP memos.
# MAGIC
# MAGIC …and **planted problems** for the AI Functions to find, every one recorded in **`ground_truth`**:
# MAGIC
# MAGIC | Section | Planted problem |
# MAGIC |---|---|
# MAGIC | 1 · FP&A | A **hiring spike** in one cost center and a **vendor overspend** in another, both starting in the first month of last quarter |
# MAGIC | 2 · Cash | A few **large customers who pay later and later** |
# MAGIC | 3 · Revenue assurance | ~3% of invoices **under-billed**, one customer **over commit whose true-up was never billed**, one customer whose **billed price slips ~1% a month** (too small for any one invoice to trip a rule), contracts **still in use after they expired**, a few **deep discounts** far outside policy, and the **PDF-vs-table term mismatches** |
# MAGIC
# MAGIC > ⚠️ **Step 0.1 starts fresh:** it replaces the demo tables in the schema from S.2, empties the `contracts` volume and deletes this notebook's job (if it exists), so the backfill re-reads every contract. **Don't run it against a schema that holds your own data** — skip to step 0.12 instead.
# MAGIC
# MAGIC All randomness is seeded, so results are reproducible, and dates are relative to today, so the data always looks current.

# COMMAND ----------

# Start fresh: delete this notebook's job (so the backfill re-reads every contract) and empty the contracts volume
for j in _api("GET", "/api/2.2/jobs/list", query={"name": JOB_NAME}).get("jobs", []):
    _api("POST", "/api/2.2/jobs/delete", body={"job_id": j["job_id"]})
for t in ("parsed_contracts", "contract_terms_extracted"):
    spark.sql(f"DROP TABLE IF EXISTS {fq(t)}")
for f in dbutils.fs.ls(CONTRACTS):
    dbutils.fs.rm(f.path, True)

# Synthetic finance data for "Halcyon Group" (fictitious). Seeded, and dated relative to today, so it always looks current.
import calendar, datetime as dt
import numpy as np, pandas as pd
from faker import Faker

rng = np.random.default_rng(7); fake = Faker("en_US"); Faker.seed(7)
TODAY = dt.date.today()

def month_end(y, m):
    return dt.date(y, m, calendar.monthrange(y, m)[1])
def add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    return month_end(d.year + y, m + 1)

LME = TODAY.replace(day=1) - dt.timedelta(days=1)                 # last closed month-end
MONTHS = [add_months(LME, -i) for i in range(35, -1, -1)]          # 36 month-ends
FUTURE = [add_months(LME, i) for i in (1, 2, 3)]                    # the forecast quarter
q_end_m = LME.month - LME.month % 3
LFQ_END = month_end(LME.year, q_end_m) if q_end_m else month_end(LME.year - 1, 12)
LFQ = [add_months(LFQ_END, -2), add_months(LFQ_END, -1), LFQ_END]   # last full quarter ("Q3")
SPIKE = LFQ[0]                                                      # where the planted FP&A problems start
t_of = {m: i for i, m in enumerate(MONTHS + FUTURE)}
US, UK = "Halcyon Inc. (US)", "Halcyon Ltd. (UK)"

# ---------------------------------------------------------------- headcount
CC = {  # start headcount, annual growth, average salary
    "Sales": (58, 0.07, 132000), "Marketing": (22, 0.06, 118000), "Customer Success": (30, 0.05, 92000),
    "R&D": (84, 0.07, 162000), "IT": (18, 0.03, 138000), "G&A": (25, 0.03, 121000), "Operations": (20, 0.04, 98000)}
emps, eid, hc_actual, hc_clean = [], 1000, {}, {}
for cc, (n0, g, sal) in CC.items():
    staff = []
    for _ in range(n0):
        staff.append(dict(start=MONTHS[0] - dt.timedelta(days=int(rng.integers(30, 1800)))))
    for e in staff:
        e["end"] = None
    for t, m in enumerate(MONTHS):
        active = [e for e in staff if e["end"] is None]
        for e in active:                                               # ~11% annual attrition
            if rng.random() < 0.11 / 12:
                e["end"] = m - dt.timedelta(days=int(rng.integers(0, 25)))
        n_act = len([e for e in staff if e["end"] is None])
        hires = rng.poisson(n_act * (g + 0.11) / 12)
        spike = 9 if (cc == "Customer Success" and m == SPIKE) else 0  # planted: unplanned CS hiring spike
        for k in range(hires + spike):
            staff.append(dict(start=m.replace(day=1) + dt.timedelta(days=int(rng.integers(0, 20))), end=None, spike=k >= hires))
        hc_actual[(cc, m)] = len([e for e in staff if e["start"] <= m and (e["end"] is None or e["end"] > m)])
        hc_clean[(cc, m)] = hc_actual[(cc, m)] - (9 if cc == "Customer Success" and m >= SPIKE else 0)
    for e in staff:
        eid += 1
        emps.append(dict(employee_id=f"E{eid}", cost_center=cc, start_date=e["start"], end_date=e["end"],
                         salary=float(round(sal * rng.lognormal(0, 0.18) / 500) * 500),
                         status="Active" if e["end"] is None else "Terminated"))
headcount = pd.DataFrame(emps)

def monthly_payroll(cc, m, clean=False):
    act = headcount[(headcount.cost_center == cc) & (headcount.start_date <= m) &
                    (headcount.end_date.isna() | (headcount.end_date > m))]
    s = act.salary.sum() / 12
    if clean and cc == "Customer Success" and m >= SPIKE:
        s -= 9 * CC[cc][2] / 12
    return s

hp = []
for cc, (n0, g, sal) in CC.items():
    for m in MONTHS:
        hp.append(dict(period=m, cost_center=cc, planned_headcount=int(hc_clean[(cc, m)])))
    for k, m in enumerate(FUTURE, 1):   # forward plan: today's headcount + planned hires
        hp.append(dict(period=m, cost_center=cc, planned_headcount=int(hc_actual[(cc, LME)] + round(k * hc_actual[(cc, LME)] * g / 12 + 0.4))))
headcount_plan = pd.DataFrame(hp)

# ---------------------------------------------------------------- vendors & opex
VENDORS = [  # vendor_id, name, cost_center, account, base monthly spend, annual growth, seasonality
    ("V-1001", "Nimbus Cloud Services", "IT", "Software & Cloud", 82000, 0.18, None),
    ("V-1001", "Nimbus Cloud Services", "R&D", "Software & Cloud", 64000, 0.15, None),
    ("V-1001", "Nimbus Cloud Services", "Operations", "Software & Cloud", 21000, 0.08, None),
    ("V-1002", "Brightline Software", "R&D", "Software & Cloud", 41000, 0.15, None),
    ("V-1003", "Corvid Security", "IT", "Software & Cloud", 17500, 0.05, None),
    ("V-1004", "Atlas Workplace Partners", "G&A", "Facilities", 96000, 0.03, None),
    ("V-1005", "Keystone Legal LLP", "G&A", "Professional Services", 21000, 0.02, "lumpy"),
    ("V-1006", "Ledgerline Audit Partners", "G&A", "Professional Services", 14000, 0.02, "audit"),
    ("V-1007", "Peakview Media", "Marketing", "Marketing Programs", 68000, 0.08, "campaign"),
    ("V-1008", "Summit Events Co.", "Marketing", "Marketing Programs", 29000, 0.05, "lumpy"),
    ("V-1009", "Talentbridge Staffing", "R&D", "Contractors", 46000, 0.04, None),
    ("V-1010", "Northstar Advisory", "Customer Success", "Contractors", 12500, 0.02, None),
    ("V-1011", "Wayfarer Travel", "Sales", "Travel & Entertainment", 39000, 0.08, "travel"),
    ("V-1011", "Wayfarer Travel", "Customer Success", "Travel & Entertainment", 9000, 0.05, "travel"),
    ("V-1012", "Relay Logistics", "Operations", "Freight & Fulfillment", 31000, 0.06, None),
    ("V-1013", "Fieldwise CRM", "Sales", "Software & Cloud", 24000, 0.07, None),
]
vendors = pd.DataFrame([dict(vendor_id=v[0], vendor_name=v[1], category=v[3]) for v in VENDORS]).drop_duplicates("vendor_id")
vendors = pd.concat([vendors, pd.DataFrame([dict(vendor_id="V-9000", vendor_name="Paycentral Payroll Services", category="Payroll")])])

def season(kind, m):
    mo = m.month
    if kind == "travel":   return {8: 0.7, 12: 0.6, 1: 0.85}.get(mo, 1.05)
    if kind == "campaign": return {3: 1.35, 9: 1.4, 12: 0.75}.get(mo, 1.0)
    if kind == "audit":    return {2: 2.6, 3: 2.2}.get(mo, 0.55)
    if kind == "lumpy":    return 1.0
    return 1.0

gl, plan, opex_lines = [], {}, []
def add_plan(m, cc, acct, amt):
    plan[(m, cc, acct)] = plan.get((m, cc, acct), 0) + amt

for (vid, vname, cc, acct, base, g, kind) in VENDORS:
    for m in MONTHS + FUTURE:
        t = t_of[m]
        clean = base * (1 + g) ** (t / 12) * season(kind, m)
        add_plan(m, cc, acct, clean)
        if m in FUTURE:
            continue
        amt = clean * rng.lognormal(0, 0.06)
        if kind == "lumpy":
            amt *= rng.choice([0.4, 0.8, 1.0, 1.5, 2.2], p=[0.2, 0.25, 0.25, 0.2, 0.1])
        if vid == "V-1001" and cc == "IT" and m >= SPIKE:   # planted: cloud overspend in IT
            amt *= 2.3
        if vid == "V-1003" and m >= SPIKE:           # ...partly hidden: security tooling consolidated into the cloud contract
            amt *= 0.25
        opex_lines.append(dict(period=m, cost_center=cc, account=acct, vendor_id=vid, amount=round(amt, 2)))

for cc in CC:
    for m in MONTHS + FUTURE:
        if m in FUTURE:
            pay_clean = monthly_payroll(cc, LME) * (headcount_plan.set_index(["cost_center", "period"]).loc[(cc, m), "planned_headcount"]
                                                   / max(hc_actual[(cc, LME)], 1))
        else:
            pay_clean = monthly_payroll(cc, m, clean=True)
        add_plan(m, cc, "Salaries & Wages", pay_clean)
        add_plan(m, cc, "Benefits & Payroll Tax", pay_clean * 0.23)
        if m in FUTURE:
            continue
        pay = monthly_payroll(cc, m)
        opex_lines.append(dict(period=m, cost_center=cc, account="Salaries & Wages", vendor_id=None, amount=round(pay, 2)))
        opex_lines.append(dict(period=m, cost_center=cc, account="Benefits & Payroll Tax", vendor_id=None, amount=round(pay * 0.23 * rng.lognormal(0, 0.01), 2)))

for r in opex_lines:   # Sales and Marketing are booked across the US and UK entities
    if r["cost_center"] in ("Sales", "Marketing"):
        gl.append(dict(r, entity=US, amount=round(r["amount"] * 0.74, 2)))
        gl.append(dict(r, entity=UK, amount=round(r["amount"] * 0.26, 2)))
    else:
        gl.append(dict(r, entity=US))

# ---------------------------------------------------------------- customers, contracts, usage, invoices
SKUS = {"PLAT-UNIT": ("Halcyon Platform — compute units", 12.00, "platform units"),
        "API-1K": ("Halcyon Data API — per block of 1,000 calls", 0.85, "1K-call blocks"),
        "SEAT-PRO": ("Halcyon Analytics Pro — named seats", 95.00, "named seats"),
        "STORAGE-TB": ("Halcyon Managed Storage — per TB-month", 23.00, "TB-months")}
INDUSTRIES = ["Financial Services", "Insurance", "Healthcare", "Life Sciences", "Manufacturing", "Retail & CPG", "Technology", "Energy & Utilities"]
names, custs = set(), []
while len(names) < 200:
    names.add(fake.company().replace(",", ""))
for i, nm in enumerate(sorted(names)):
    seg = rng.choice(["Enterprise", "Mid-Market", "SMB"], p=[0.2, 0.4, 0.4])
    terms = {"Enterprise": rng.choice([60, 45]), "Mid-Market": rng.choice([45, 30]), "SMB": 30}[seg]
    custs.append(dict(customer_id=f"C{1001 + i}", name=nm, segment=seg, industry=str(rng.choice(INDUSTRIES)), payment_terms_days=int(terms)))

customers = pd.DataFrame(custs)

ent = customers[customers.segment == "Enterprise"].customer_id.tolist()
mm = customers[customers.segment == "Mid-Market"].customer_id.tolist()
LATE_PAYERS = ent[:4]                    # planted: large customers who pay later and later
TRUEUP_CUST = mm[0]                      # planted: over commit, true-up never billed
SLIP_CUST = ent[5]                       # planted: billed price slips ~1% a month
pool = [c for c in customers.customer_id if c not in LATE_PAYERS + [TRUEUP_CUST, SLIP_CUST]]
pick = list(rng.choice(pool, 24, replace=False))
EXPIRED = pick[:7]                       # planted: contract expired, still using (and billed at the old price)
DEEP = pick[7:10]                        # planted: unapproved deep discount (billing matches the contract)
MISMATCH = {c: kind for c, kind in zip(pick[10:20], ["contracted_price"] * 4 + ["min_commit"] * 3 + ["end_date"] * 3)}  # 5% PDF-vs-table

contracts, pdf_terms, usage_rows, invoices, gt = [], [], [], [], []
target = {"Enterprise": (70000, 140000), "Mid-Market": (15000, 36000), "SMB": (2500, 7000)}
inv_no = 500000
TRUEUP_TEXTS = [
    "Usage in excess of the Minimum Monthly Commitment will be invoiced monthly in arrears at the Contracted Unit Price (true-up).",
    "True-up: any consumption above the Minimum Monthly Commitment is billed in arrears each month at the Contracted Unit Price.",
    "If monthly usage exceeds the committed quantity, Customer will be charged for the overage at the Contracted Unit Price on the next invoice.",
]
for k, c in customers.iterrows():
    cid, seg = c.customer_id, c.segment
    sku = str(rng.choice(list(SKUS), p=[0.4, 0.2, 0.25, 0.15]))
    lst = SKUS[sku][1]
    disc = {"Enterprise": rng.uniform(0.10, 0.25), "Mid-Market": rng.uniform(0.05, 0.15), "SMB": rng.uniform(0.0, 0.08)}[seg]
    if cid in DEEP:
        disc = rng.uniform(0.42, 0.48)
    price = round(lst * (1 - disc), 4 if sku == "API-1K" else 2)
    since = add_months(MONTHS[0], -int(rng.integers(-12, 30)))           # customer since (may predate the window)
    since = min(since, add_months(LME, -4))
    term = int(rng.choice([12, 24, 36]))
    start = (add_months(since, -1) + dt.timedelta(days=1))
    end = add_months(start, term) - dt.timedelta(days=0)
    while end < TODAY + dt.timedelta(days=45):                          # auto-renewed: the contract row is the current term
        start, end = end + dt.timedelta(days=1), add_months(end, term)
    if cid in EXPIRED:                                                  # never renewed: ended 2-7 months ago
        end = add_months(LME, -int(rng.integers(2, 8)))
        start = add_months(end, -term) + dt.timedelta(days=1)
    m0_qty = rng.uniform(*target[seg]) / price
    commit = round(m0_qty * rng.uniform(0.6, 0.85), -1 if m0_qty > 100 else 0)
    growth = rng.uniform(-0.05, 0.25)
    row = dict(contract_id=f"CTR-{20001 + k}", customer_id=cid, sku=sku, contracted_price=price,
               min_commit=float(commit), start_date=start, end_date=end, list_price=lst)
    truth = dict(row, true_up_terms=str(rng.choice(TRUEUP_TEXTS)), layout=int(k % 3))
    kind = MISMATCH.get(cid)
    if kind == "contracted_price":  # the table was keyed lower than the signed price
        row["contracted_price"] = round(price * rng.uniform(0.85, 0.92), 4 if sku == "API-1K" else 2)
    elif kind == "min_commit":  # the table holds a smaller minimum commitment than the signed contract
        row["min_commit"] = float(round(commit * rng.uniform(0.55, 0.7), -1 if commit > 100 else 0))
    elif kind == "end_date":    # the table shows a renewal the customer never signed
        truth["end_date"] = add_months(LME, -int(rng.integers(2, 6)))
    contracts.append(row); pdf_terms.append(truth)
    if kind:
        gt.append(dict(use_case="Revenue assurance", anomaly_type="pdf_table_mismatch", entity_type="contract", entity_id=row["contract_id"],
                       field=kind, true_value=str(truth[kind]), recorded_value=str(row[kind]), detail=f"Signed PDF and contracts table disagree on {kind}"))
    for f in ["customer_name", "sku", "contracted_price", "min_commit", "start_date", "end_date", "true_up_terms"]:
        tv = c["name"] if f == "customer_name" else truth[f]
        gt.append(dict(use_case="Contract extraction", anomaly_type="contract_terms", entity_type="contract", entity_id=row["contract_id"],
                       field=f, true_value=str(tv), recorded_value=None, detail="What the signed contract PDF says"))

    # monthly usage -> daily usage -> monthly invoice
    base_lateness, months_billed = max(0.0, rng.normal(3, 5)), 0
    for m in MONTHS:
        if m < since:
            continue
        t = t_of[m]
        qty = m0_qty * (1 + growth) ** (months_billed / 12) * rng.lognormal(0, 0.07)
        if cid == TRUEUP_CUST and m > add_months(LME, -9):
            qty = commit * rng.uniform(1.3, 1.6)
        months_billed += 1
        qty = float(round(qty))
        days = calendar.monthrange(m.year, m.month)[1]
        wts = np.array([0.6 if dt.date(m.year, m.month, d).weekday() >= 5 else 1.0 for d in range(1, days + 1)]) * rng.lognormal(0, 0.15, days)
        daily = np.floor(wts / wts.sum() * qty)
        daily[-1] += qty - daily.sum()
        for d in range(days):
            usage_rows.append((cid, sku, dt.date(m.year, m.month, d + 1), float(daily[d])))
        c_commit, c_price = row["min_commit"], row["contracted_price"]     # billing runs off the contracts TABLE
        billed_qty, billed_price = max(qty, c_commit), c_price
        if cid == TRUEUP_CUST and m > add_months(LME, -9):
            billed_qty = c_commit                                           # true-up never charged
        if cid == SLIP_CUST and m > add_months(LME, -6):   # 1% lower every month for the last 6 months
            months_back = (LME.year - m.year) * 12 + (LME.month - m.month)
            billed_price = round(c_price * (1 - 0.01 * (6 - months_back)), 4)
        inv_no += 1
        iid = f"INV-{inv_no}"
        if (cid not in LATE_PAYERS + [TRUEUP_CUST, SLIP_CUST] + EXPIRED + list(MISMATCH) and qty > c_commit * 1.15
                and rng.random() < 0.03):                                  # planted: ~3% of invoices under-billed
            billed_qty = float(max(c_commit + 1, round(qty * rng.uniform(0.65, 0.9))))
            gt.append(dict(use_case="Revenue assurance", anomaly_type="underbilled_invoice", entity_type="invoice", entity_id=iid,
                           field="billed_qty", true_value=str(qty), recorded_value=str(billed_qty), detail=f"{cid}: billed fewer units than used"))
        amount = round(billed_qty * billed_price, 2)
        due = m + dt.timedelta(days=int(c.payment_terms_days))
        late = base_lateness + rng.normal(0, 4)
        if cid in LATE_PAYERS:
            late += max(0, 12 - (LME.year - m.year) * 12 - (LME.month - m.month)) * 3.2   # +3 days a month over the last year
        paid = due + dt.timedelta(days=int(round(max(-8, late))))
        paid = max(paid, m + dt.timedelta(days=5))
        invoices.append(dict(invoice_id=iid, customer_id=cid, invoice_date=m, due_date=due,
                             paid_date=paid if paid < TODAY else None, amount=amount, sku=sku,
                             billed_qty=billed_qty, billed_price=billed_price))

for cid in LATE_PAYERS:
    gt.append(dict(use_case="Cash", anomaly_type="late_payer", entity_type="customer", entity_id=cid, field="days_to_pay", true_value=None,
                   recorded_value=None, detail="Pays ~3 days later every month over the last 12 months"))
ctr = {r["customer_id"]: r["contract_id"] for r in contracts}
gt.append(dict(use_case="Revenue assurance", anomaly_type="missed_true_up", entity_type="contract", entity_id=ctr[TRUEUP_CUST], field="billed_qty",
               true_value=None, recorded_value=None, detail=f"{TRUEUP_CUST}: usage 30-60% over commit for 9 months, only the commit billed"))
gt.append(dict(use_case="Revenue assurance", anomaly_type="price_slip", entity_type="contract", entity_id=ctr[SLIP_CUST], field="billed_price",
               true_value=None, recorded_value=None, detail=f"{SLIP_CUST}: billed price slips 1% a month for 6 months"))
for cid in EXPIRED:
    gt.append(dict(use_case="Revenue assurance", anomaly_type="expired_in_use", entity_type="contract", entity_id=ctr[cid], field="end_date",
                   true_value=None, recorded_value=None, detail=f"{cid}: contract ended, still using and billed at the old discount"))
for cid in DEEP:
    gt.append(dict(use_case="Revenue assurance", anomaly_type="deep_discount", entity_type="contract", entity_id=ctr[cid], field="contracted_price",
                   true_value=None, recorded_value=None, detail=f"{cid}: 42-48% discount, far outside policy"))
gt.append(dict(use_case="FP&A", anomaly_type="hiring_spike", entity_type="cost_center", entity_id="Customer Success", field="Salaries & Wages",
               true_value=str(SPIKE), recorded_value=None, detail="9 unplanned hires in the first month of last quarter"))
gt.append(dict(use_case="FP&A", anomaly_type="vendor_overspend", entity_type="vendor", entity_id="V-1001", field="IT / Software & Cloud",
               true_value=str(SPIKE), recorded_value=None, detail="Nimbus Cloud Services spend x2.3 from the first month of last quarter"))

inv = pd.DataFrame(invoices)
rev = inv.assign(account=inv.sku.map({"PLAT-UNIT": "Subscription Revenue", "SEAT-PRO": "Subscription Revenue",
                                      "API-1K": "Usage Revenue", "STORAGE-TB": "Usage Revenue"})).groupby(["invoice_date", "account"]).amount.sum()
for (m, acct), amt in rev.items():
    gl.append(dict(period=m, entity=US, cost_center="Revenue", account=acct, vendor_id=None, amount=round(amt, 2)))
rev_by_m = rev.groupby(level=0).sum()
for m in MONTHS + FUTURE:   # revenue plan: smooth trend, a stretch target from last quarter on
    t = t_of[m]
    base = np.polyval(np.polyfit(range(30), [rev_by_m.get(x, 0) for x in MONTHS[:30]], 1), t)
    uplift = 1.0 if m >= SPIKE else 0.985
    for acct, share in (("Subscription Revenue", 0.66), ("Usage Revenue", 0.34)):
        add_plan(m, "Revenue", acct, base * share * uplift)

gl_actuals = pd.DataFrame(gl)[["period", "entity", "cost_center", "account", "vendor_id", "amount"]]
budget_plan = pd.DataFrame([dict(period=m, cost_center=cc, account=a, plan_amount=round(v * rng.lognormal(0, 0.008), 2))
                            for (m, cc, a), v in plan.items()])

# ---------------------------------------------------------------- AP bills (vendors + payroll), incl. scheduled future bills
bills, bno = [], 70000
for r in opex_lines:
    if r["vendor_id"] is None:
        continue
    bno += 1
    due = r["period"] + dt.timedelta(days=30)
    paid = due + dt.timedelta(days=int(rng.integers(-3, 4)))
    bills.append(dict(bill_id=f"BILL-{bno}", vendor_id=r["vendor_id"], cost_center=r["cost_center"], bill_date=r["period"],
                      due_date=due, paid_date=paid if paid < TODAY else None, amount=r["amount"]))
last_run = {(r["vendor_id"], r["cost_center"]): r["amount"] for r in opex_lines if r["vendor_id"] and r["period"] == LME}
for m in FUTURE:
    for (vid, cc), amt in last_run.items():
        bno += 1
        bills.append(dict(bill_id=f"BILL-{bno}", vendor_id=vid, cost_center=cc, bill_date=m, due_date=m + dt.timedelta(days=30),
                          paid_date=None, amount=round(amt, 2)))
payroll_m = {m: sum(monthly_payroll(cc, m) for cc in CC) * 1.23 for m in MONTHS}
run_rate = payroll_m[LME]
d = MONTHS[0].replace(day=1)
while d <= TODAY + dt.timedelta(days=100):
    for day in (15, calendar.monthrange(d.year, d.month)[1]):
        pdate = dt.date(d.year, d.month, day)
        me = month_end(d.year, d.month)
        amt = (payroll_m.get(me, run_rate)) / 2
        bno += 1
        bills.append(dict(bill_id=f"BILL-{bno}", vendor_id="V-9000", cost_center="All", bill_date=pdate - dt.timedelta(days=3),
                          due_date=pdate, paid_date=pdate if pdate < TODAY else None, amount=round(amt, 2)))
    d = month_end(d.year, d.month) + dt.timedelta(days=1)
ap_bills = pd.DataFrame(bills)
ap_bills["memo"] = ""

# ---------------------------------------------------------------- bank transactions
cname = customers.set_index("customer_id").name.to_dict()
vname = vendors.set_index("vendor_id").vendor_name.to_dict()
tx = [dict(date=MONTHS[0].replace(day=1), account="Operating ••4410", amount=12_000_000.0, counterparty="Opening balance", type="Opening balance",
           memo="Opening balance carried forward")]
for r in invoices:
    if r["paid_date"]:
        tx.append(dict(date=r["paid_date"], account="Operating ••4410", amount=r["amount"], counterparty=cname[r["customer_id"]],
                       type="AR receipt", memo=f"ACH credit — remittance {r['invoice_id']}"))
for b in bills:
    if b["paid_date"]:
        tx.append(dict(date=b["paid_date"], account="Operating ••4410", amount=-b["amount"], counterparty=vname[b["vendor_id"]],
                       type="Payroll" if b["vendor_id"] == "V-9000" else "AP payment", memo=f"Payment {b['bill_id']}"))
d = MONTHS[0]
while d < TODAY:
    if d.month in (1, 4, 6, 9):
        tx.append(dict(date=dt.date(d.year, d.month, 15), account="Operating ••4410", amount=-round(420000 * rng.lognormal(0, 0.05), 2),
                       counterparty="US Treasury / HMRC", type="Tax", memo="Estimated income tax payment"))
    tx.append(dict(date=d, account="Operating ••4410", amount=round(9000 * rng.lognormal(0, 0.1), 2), counterparty="First Meridian Bank",
                   type="Interest", memo="Interest on operating balance"))
    d = add_months(d, 1)
bank_transactions = pd.DataFrame([x for x in tx if x["date"] < TODAY])

usage = pd.DataFrame(usage_rows, columns=["customer_id", "sku", "date", "quantity"])
contracts_df = pd.DataFrame(contracts).drop(columns=["list_price"])
price_list = pd.DataFrame([dict(sku=k, description=v[0], list_price=v[1], unit=v[2]) for k, v in SKUS.items()])
ground_truth = pd.DataFrame(gt)
CONTRACT_PDF_TERMS = pdf_terms          # used by step 0.2 to write the signed PDFs
print(f"Last closed month {LME} · last full quarter {LFQ[0]:%b}–{LFQ[-1]:%b %Y} · {len(customers)} customers · {len(inv):,} invoices · "
      f"{len(usage):,} usage rows · {len(gl_actuals):,} GL lines · {len(ap_bills):,} AP bills · {len(bank_transactions):,} bank transactions")

def save(pdf, name, comment):
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq(name))
    spark.sql(f"COMMENT ON TABLE {fq(name)} IS '{_q(comment)}'")
for pdf, name in [(gl_actuals, "gl_actuals"), (budget_plan, "budget_plan"), (headcount, "headcount"), (headcount_plan, "headcount_plan"),
                  (customers, "customers"), (vendors, "vendors"), (bank_transactions, "bank_transactions"), (inv, "ar_invoices"),
                  (ap_bills, "ap_bills"), (contracts_df, "contracts"), (usage, "usage"), (price_list, "price_list"), (ground_truth, "ground_truth")]:
    save(pdf, name, TABLE_INFO[name][1])
print("✓ 13 tables written to", f"{CATALOG}.{SCHEMA}")

# COMMAND ----------

class PDFDoc:
    """Tiny dependency-free PDF writer (text, lines, boxes) for realistic-looking business documents."""
    W, H = 612, 792

    def __init__(self):
        self.pages, self.ops = [], []

    @staticmethod
    def _esc(s):
        s = str(s).encode("cp1252", "replace").decode("cp1252")
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    @staticmethod
    def _width(s, size):
        """Approximate Helvetica-Bold advance widths (slightly generous so text never overflows)."""
        w = 0
        for ch in str(s):
            w += 0.74 if ch.isupper() else 0.58 if ch.isdigit() else 0.3 if ch in " .,:;'|!il1" else 0.56
        return w * size

    def text(self, x, y, s, size=9, bold=False, color=(0, 0, 0), align="left"):
        while align == "right" and size > 12 and self._width(s, size) > 250:
            size -= 1
        if align == "right":
            x -= self._width(s, size)
        elif align == "center":
            x -= self._width(s, size) / 2
        r, g, b = color
        self.ops.append(f"BT {r} {g} {b} rg /{'F2' if bold else 'F1'} {size} Tf {x:.1f} {self.H - y:.1f} Td ({self._esc(s)}) Tj ET")

    def line(self, x1, y1, x2, y2, width=0.6, gray=0.55):
        self.ops.append(f"{gray} G {width} w {x1:.1f} {self.H - y1:.1f} m {x2:.1f} {self.H - y2:.1f} l S")

    def box(self, x, y, w, h, fill=None, stroke=0.55):
        if fill is not None:
            r, g, b = fill
            self.ops.append(f"{r} {g} {b} rg {x:.1f} {self.H - y - h:.1f} {w:.1f} {h:.1f} re f")
        if stroke is not None:
            self.ops.append(f"{stroke} G 0.6 w {x:.1f} {self.H - y - h:.1f} {w:.1f} {h:.1f} re S")

    def _wrap(self, s, width, size):
        words, lines, cur = str(s).split(" "), [], ""
        for wd in words:
            cand = (cur + " " + wd).strip()
            if cur and self._width(cand, size) > width:
                lines.append(cur); cur = wd
            else:
                cur = cand
        return lines + [cur]

    def table(self, x, y, cols, rows, size=8.5, row_h=16, header_fill=(0.92, 0.93, 0.95)):
        """cols = [(header, width, align)], rows = [[cell, ...]]. Long cells wrap. Returns y after the table."""
        total, top = sum(c[1] for c in cols), y
        self.box(x, y, total, row_h, fill=header_fill)
        cx = x
        for head, wdt, al in cols:
            self.text(cx + (wdt - 4 if al == "right" else 4), y + row_h - 5, head, size, bold=True, align=al)
            cx += wdt
        y += row_h
        for r in rows:
            wrapped = [self._wrap(cell, wdt - 8, size) for (head, wdt, al), cell in zip(cols, r)]
            h = row_h + (max(len(wl) for wl in wrapped) - 1) * (size + 2)
            cx = x
            for (head, wdt, al), wl in zip(cols, wrapped):
                for k, ln in enumerate(wl):
                    self.text(cx + (wdt - 4 if al == "right" else 4), y + row_h - 5 + k * (size + 2), ln, size, align=al)
                cx += wdt
            y += h
            self.line(x, y, x + total, y, 0.3, 0.75)
        self.box(x, top, total, y - top, stroke=0.55)
        return y

    def new_page(self):
        self.pages.append(self.ops)
        self.ops = []

    def bytes(self):
        if self.ops:
            self.new_page()
        objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"]
        kids = []
        for ops in self.pages:
            stream = "\n".join(ops).encode("cp1252", "replace")
            objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
            objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.W} {self.H}] "
                        f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {len(objs)} 0 R >>")
            kids.append(f"{len(objs)} 0 R")
        objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
        out, offs = bytearray(b"%PDF-1.4\n"), []
        for i, o in enumerate(objs, 1):
            offs.append(len(out))
            out += f"{i} 0 obj\n".encode() + (o if isinstance(o, bytes) else o.encode()) + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        out += "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
        out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
        return bytes(out)
# Signed order forms, one PDF per contract, in three layouts (no two customers' paper looks the same).
import os

NAVY, GRAY, RED = (0.11, 0.19, 0.22), (0.35, 0.43, 0.47), (0.85, 0.2, 0.13)
fmt_money = lambda v, sku: (f"${v:,.4f}" if sku == "API-1K" else f"${v:,.2f}")
fmt_qty = lambda v: f"{v:,.0f}"
fmt_date = lambda d: d.strftime("%B %-d, %Y")

def contract_pdf(t, cust, skus, signer):
    desc, lst, unit = skus[t["sku"]]
    p = PDFDoc()
    if t["layout"] == 0:                                    # classic order form
        p.box(0, 0, 612, 70, fill=NAVY, stroke=None)
        p.text(40, 32, "HALCYON GROUP", 18, True, (1, 1, 1))
        p.text(40, 52, "Master Subscription Agreement — Order Form", 11, False, (1, 1, 1))
        p.text(572, 32, f"Order Form {t['contract_id']}", 11, True, (1, 1, 1), "right")
        p.text(40, 100, "Customer", 9, True, GRAY); p.text(40, 114, cust["name"], 11, True)
        p.text(40, 128, f"{cust['industry']} · {cust['segment']}", 9)
        p.text(330, 100, "Provider", 9, True, GRAY); p.text(330, 114, "Halcyon Inc.", 11, True)
        p.text(330, 128, "1200 Harbor Point Drive, Boston, MA 02210", 9)
        y = p.table(40, 160, [("Product (SKU)", 230, "left"), ("Contracted unit price", 130, "right"), ("Min. monthly commitment", 172, "right")],
                    [[f"{desc} ({t['sku']})", f"{fmt_money(t['contracted_price'], t['sku'])} per unit", f"{fmt_qty(t['min_commit'])} {unit}"]])
        p.text(40, y + 26, "Subscription term", 10, True)
        p.text(40, y + 42, f"Start date: {fmt_date(t['start_date'])}        End date: {fmt_date(t['end_date'])}", 10)
        p.text(40, y + 70, "Usage, minimums and true-up", 10, True)
        for i, ln in enumerate(p._wrap(t["true_up_terms"] + " Usage below the Minimum Monthly Commitment is billed at the commitment.", 520, 9.5)):
            p.text(40, y + 86 + i * 13, ln, 9.5)
        p.text(40, y + 140, "Payment terms", 10, True)
        p.text(40, y + 156, f"Invoices are issued monthly in arrears and are due net {cust['payment_terms_days']} days from the invoice date.", 9.5)
        sy = y + 220
    elif t["layout"] == 1:                                  # compact schedule
        p.text(40, 50, f"SCHEDULE A to the Halcyon Master Services Agreement", 13, True, NAVY)
        p.text(40, 68, f"Schedule No. {t['contract_id']}  ·  Customer: {cust['name']}", 10)
        p.line(40, 80, 572, 80, 1.2, 0.2)
        rows = [["Service", f"{desc}"], ["SKU", t["sku"]], ["Price", f"{fmt_money(t['contracted_price'], t['sku'])} per {unit[:-1] if unit.endswith('s') else unit}"],
                ["Committed volume", f"{fmt_qty(t['min_commit'])} {unit} per calendar month (minimum)"],
                ["Effective date", t["start_date"].isoformat()], ["Expiration date", t["end_date"].isoformat()],
                ["Overage / true-up", t["true_up_terms"]], ["Payment", f"Net {cust['payment_terms_days']}, monthly in arrears, USD"]]
        y = p.table(40, 100, [("Term", 150, "left"), ("Agreed", 382, "left")], rows, size=9, row_h=18)
        p.text(40, y + 24, "This Schedule is governed by the Master Services Agreement between the parties. In case of conflict, this Schedule prevails.", 8.5, color=GRAY)
        sy = y + 80
    else:                                                   # legal-style memorandum
        p.text(306, 50, "SUBSCRIPTION ORDER", 15, True, NAVY, "center")
        p.text(306, 68, f"Reference {t['contract_id']}", 10, False, GRAY, "center")
        paras = [
            f"This Subscription Order is entered into by Halcyon Inc. (\"Halcyon\") and {cust['name']} (\"Customer\") and is effective as of "
            f"{fmt_date(t['start_date'])} (the \"Effective Date\").",
            f"1. Services. Halcyon will provide {desc} (SKU {t['sku']}).",
            f"2. Fees. Customer will pay {fmt_money(t['contracted_price'], t['sku'])} per unit for the {unit} consumed (the \"Contracted Unit Price\"). "
            f"Customer commits to a minimum of {fmt_qty(t['min_commit'])} {unit} in each calendar month (the \"Minimum Monthly Commitment\").",
            f"3. True-up. {t['true_up_terms']}",
            f"4. Term. This Subscription Order remains in effect until {fmt_date(t['end_date'])}, unless renewed in writing by both parties.",
            f"5. Invoicing. Halcyon invoices monthly in arrears; invoices are payable within {cust['payment_terms_days']} days.",
        ]
        y = 100
        for para in paras:
            for ln in p._wrap(para, 520, 10):
                p.text(46, y, ln, 10); y += 14
            y += 8
        sy = y + 40
    p.line(40, sy, 270, sy); p.line(330, sy, 560, sy)
    p.text(40, sy + 14, f"For {cust['name']}: {signer}", 8.5); p.text(330, sy + 14, "For Halcyon Inc.: Dana Whitfield, VP Revenue", 8.5)
    p.text(40, sy + 28, f"Signed {fmt_date(t['start_date'])}", 8.5, color=GRAY)
    p.text(306, 770, "Confidential — Halcyon Group (fictitious company, synthetic document)", 7.5, color=GRAY, align="center")
    return p.bytes()

def write_contract_pdfs(terms, customers, skus, folder, seed_names):
    os.makedirs(folder, exist_ok=True)
    cust = customers.set_index("customer_id").to_dict("index")
    for i, t in enumerate(terms):
        c = dict(cust[t["customer_id"]], name=cust[t["customer_id"]]["name"])
        with open(f"{folder}/{t['contract_id']}.pdf", "wb") as f:
            f.write(contract_pdf(t, c, skus, seed_names[i % len(seed_names)]))
    return len(terms)
SIGNERS = [f"{fake.name()}, {t}" for t in ["CFO", "VP Finance", "COO", "Head of Procurement", "General Counsel", "CIO"] for _ in range(4)]
n = write_contract_pdfs(CONTRACT_PDF_TERMS, customers, SKUS, CONTRACTS, SIGNERS)
displayHTML(f'<div style="{FONT};font-size:14px">✓ {n} contract PDFs in <b>{CONTRACTS}</b> — three layouts, ~5% of them disagree with the <code>contracts</code> table. '
            f'{button("Browse & preview the PDFs", volume_url("contracts"), "#5A6F77")}</div>')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- One memo per vendor, cost center and quarter, written by ai_gen from the actual spend change, then stamped on every bill
# MAGIC CREATE OR REPLACE TABLE ap_memo_lookup AS
# MAGIC WITH q AS (
# MAGIC   SELECT b.vendor_id, v.vendor_name, v.category, b.cost_center, CAST(date_trunc('QUARTER', b.bill_date) AS DATE) AS quarter, sum(b.amount) AS spend
# MAGIC   FROM ap_bills b JOIN vendors v USING (vendor_id)
# MAGIC   WHERE b.vendor_id <> 'V-9000' AND b.bill_date <= lme
# MAGIC   GROUP BY ALL
# MAGIC ), c AS (SELECT *, lag(spend) OVER (PARTITION BY vendor_id, cost_center ORDER BY quarter) AS prev_spend FROM q)
# MAGIC SELECT vendor_id, cost_center, quarter,
# MAGIC        ai_gen(concat('Write a single accounts-payable invoice memo line (max 22 words, no quotes, no greeting) for a bill from ', vendor_name,
# MAGIC                      ' (', category, ') charged to the ', cost_center, ' cost center. Quarter-over-quarter spend change: ',
# MAGIC                      coalesce(CAST(round((spend / prev_spend - 1) * 100) AS STRING), 'n/a'), '%. ',
# MAGIC                      CASE WHEN spend > prev_spend * 1.5 THEN 'The jump was not in the budget - say what drove it (e.g. extra capacity, overage, rush work).'
# MAGIC                           ELSE 'Describe the routine service billed.' END)) AS memo
# MAGIC FROM c;
# MAGIC
# MAGIC MERGE INTO ap_bills b
# MAGIC USING ap_memo_lookup m
# MAGIC ON b.vendor_id = m.vendor_id AND b.cost_center = m.cost_center AND CAST(date_trunc('QUARTER', b.bill_date) AS DATE) = m.quarter
# MAGIC WHEN MATCHED THEN UPDATE SET memo = m.memo;
# MAGIC
# MAGIC UPDATE ap_bills SET memo = CASE WHEN vendor_id = 'V-9000' THEN 'Semi-monthly payroll run: gross wages, employer taxes and benefits'
# MAGIC                                 ELSE 'Scheduled recurring bill (purchase order on file)' END
# MAGIC WHERE memo IS NULL OR memo = '';
# MAGIC
# MAGIC SELECT v.vendor_name, b.cost_center, b.bill_date, b.amount, b.memo
# MAGIC FROM ap_bills b JOIN vendors v USING (vendor_id)
# MAGIC WHERE b.bill_date BETWEEN lfq_start AND lfq_end AND b.vendor_id IN ('V-1001', 'V-1007', 'V-1011')
# MAGIC ORDER BY b.bill_date LIMIT 9;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'gl_actuals' AS table_name, count(*) AS row_count FROM gl_actuals UNION ALL
# MAGIC SELECT 'budget_plan', count(*) FROM budget_plan UNION ALL
# MAGIC SELECT 'headcount', count(*) FROM headcount UNION ALL
# MAGIC SELECT 'headcount_plan', count(*) FROM headcount_plan UNION ALL
# MAGIC SELECT 'customers', count(*) FROM customers UNION ALL
# MAGIC SELECT 'contracts', count(*) FROM contracts UNION ALL
# MAGIC SELECT 'usage', count(*) FROM usage UNION ALL
# MAGIC SELECT 'ar_invoices', count(*) FROM ar_invoices UNION ALL
# MAGIC SELECT 'ap_bills', count(*) FROM ap_bills UNION ALL
# MAGIC SELECT 'bank_transactions', count(*) FROM bank_transactions UNION ALL
# MAGIC SELECT 'ground_truth', count(*) FROM ground_truth;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ground_truth records every planted problem. You'll see each one surface in sections 1-3, and MLflow scores against it.
# MAGIC SELECT use_case, anomaly_type, count(*) AS planted_cases, any_value(detail) AS example
# MAGIC FROM ground_truth
# MAGIC WHERE anomaly_type <> 'contract_terms'
# MAGIC GROUP BY ALL
# MAGIC ORDER BY use_case, anomaly_type;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.6 Column comments for lineage and Genie
# MAGIC Every table — and its key columns — gets a business description. Genie relies on these to pick the right table and column, and they show up next to the lineage graph in Unity Catalog.

# COMMAND ----------

COLUMN_COMMENTS = {
    ("gl_actuals", "period"): "Month-end date of the accounting period", ("gl_actuals", "amount"): "Actual amount in USD (revenue and expense both positive)",
    ("gl_actuals", "cost_center"): "Cost center; 'Revenue' holds the revenue accounts", ("gl_actuals", "vendor_id"): "Vendor for non-payroll spend; NULL for payroll and revenue",
    ("budget_plan", "plan_amount"): "Planned amount in USD", ("headcount_plan", "planned_headcount"): "Planned employees at month-end",
    ("customers", "payment_terms_days"): "Contractual payment terms (net days)", ("ar_invoices", "paid_date"): "NULL while the invoice is open",
    ("ar_invoices", "billed_qty"): "Quantity billed on the invoice", ("ar_invoices", "billed_price"): "Unit price billed on the invoice",
    ("ap_bills", "paid_date"): "NULL while the bill is open or scheduled", ("contracts", "contracted_price"): "Unit price as keyed into billing (the signed PDF wins if they differ)",
    ("contracts", "min_commit"): "Minimum monthly commitment (quantity) as keyed into billing", ("usage", "quantity"): "Units consumed that day",
    ("bank_transactions", "amount"): "Positive = cash in, negative = cash out (USD)",
    ("actuals_vs_plan", "variance"): "actual_amount - plan_amount (USD); positive opex variance = over budget",
    ("actuals_vs_plan", "variance_pct"): "Variance as % of plan", ("forecast_next_qtr", "amount_forecast"): "ai_forecast point forecast (USD)",
    ("expected_receipts", "expected_pay_date"): "invoice_date + days_to_pay predicted by ai_predict_value",
    ("cash_forecast_13wk", "bottom_up_balance"): "Bank balance + expected AR receipts - scheduled AP payments, cumulative by week",
    ("cash_forecast_13wk", "top_down_balance"): "Bank balance + cumulative ai_forecast of weekly net cash flow",
    ("contract_to_cash", "leakage_usd"): "Revenue we were entitled to under the signed contract but didn't bill (USD)",
    ("leakage_exceptions", "amount_at_risk_usd"): "Estimated revenue at risk for this exception (USD)",
    ("leakage_exceptions", "verdict"): "ai_decide: billing_error or valid_discount, judged against the signed contract terms extracted from the PDF",
    ("working_capital", "dso_days"): "Days sales outstanding = open AR / last 90 days revenue x 90",
}
describe_tables()
print("✓ Comments applied to", len([t for t in TABLE_INFO if t in existing_tables()]), "tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.7 Your source data in Unity Catalog
# MAGIC Every table — and the folder of PDFs — is governed by **Unity Catalog**: permissions, lineage and audit come for free. Click a card to see the schema, sample data or lineage.

# COMMAND ----------

show_uc_cards(roles={"Bronze · source", "Answer key"}, volume="contracts")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.8 Start reading the contracts now — in the background
# MAGIC Reading 200 PDFs is a **recurring, multi-step pipeline** — new contracts arrive every week — so it runs as a **Lakeflow Job of native AI Function tasks** rather than notebook code:
# MAGIC
# MAGIC ```
# MAGIC parse_contracts ───────────────> extract_contract_terms
# MAGIC (Document Parsing · ai_parse_document)   (Information Extraction · ai_extract)
# MAGIC  reads every new PDF in the volume        reads the parsed output, writes contract_terms_extracted
# MAGIC ```
# MAGIC
# MAGIC These task types are **incremental by design**: the first run processes every PDF (the **backfill**), later runs only new or changed files (the **daily** run in step 3.11). A brand-new parsing task has a **one-time start-up of about 15–20 minutes** on its first run, so step 0.8 creates the job and **starts it now, without waiting** — it should be done, or nearly done, by the time you reach section 3. You'll look at both task definitions in step 3.1.

# COMMAND ----------

CONTRACT_SCHEMA = {
    "contract_number": {"type": "string", "description": "Order form / schedule / reference number, format CTR-#####"},
    "customer_name": {"type": "string", "description": "The customer company's name only (not Halcyon, and without the industry or segment line printed under it)"},
    "sku": {"type": "enum", "labels": ["PLAT-UNIT", "API-1K", "SEAT-PRO", "STORAGE-TB"]},
    "contracted_price": {"type": "number", "description": "Contracted unit price in USD"},
    "min_commit": {"type": "number", "description": "Minimum monthly commitment as the number written on the contract, in the contract's own unit (platform units, named seats, 1K-call blocks or TB-months). Do not convert units or multiply."},
    "true_up_terms": {"type": "string", "description": "One sentence: how usage above the minimum commitment is billed"},
    "start_date": {"type": "string", "description": "Effective / start date as YYYY-MM-DD"},
    "end_date": {"type": "string", "description": "End / expiration date as YYYY-MM-DD"},
    "payment_terms_days": {"type": "integer", "description": "Net payment terms in days"},
}
parse_task = {
    "task_key": "parse_contracts",
    "ai_function_task": {
        "volume": {"volume_path": CONTRACTS},
        "output": {"catalog": CATALOG, "schema": SCHEMA, "table": "parsed_contracts"},
        "ai_parse_document": {"version": "2.0"},
    },
}
extract_task = {
    "task_key": "extract_contract_terms",
    "depends_on": [{"task_key": "parse_contracts"}],
    "ai_function_task": {
        "delta_table": {"table_name": "{{tasks.parse_contracts.output.table_name}}", "input_column": "response", "merge_key_columns": ["id"]},
        "output": {"catalog": CATALOG, "schema": SCHEMA, "table": "contract_terms_extracted"},
        "ai_extract": {"version": "2.1", "schema": json.dumps(CONTRACT_SCHEMA), "enable_confidence_scores": True,
                       "instructions": "Signed order forms between Halcyon Inc. (the provider) and a customer. Return dates as YYYY-MM-DD."},
    },
}
TASKS = [parse_task, extract_task]
for t in TASKS:
    t.update(max_retries=1, min_retry_interval_millis=60_000, timeout_seconds=7200)
job_settings = {"name": JOB_NAME, "max_concurrent_runs": 1, "queue": {"enabled": True}, "tags": {"industry_play": "office-of-the-cfo"},
                "schedule": {"quartz_cron_expression": "0 0 6 * * ?", "timezone_id": "America/New_York", "pause_status": "PAUSED"},
                "tasks": TASKS}
JOB_ID = create_or_update_job(job_settings)
BACKFILL = start_job_run(JOB_ID)
displayHTML(f"""<div style="{FONT};font-size:14px">Backfill started in the background (2 AI Function tasks, serverless, daily 6:00 AM schedule <b>paused</b>) — continue with section 1.
  {button("Watch the backfill run", BACKFILL["run_page_url"])}{button("Open the job", JOB_URL, "#1B3139")}</div>""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.9 Meet your Genie agent
# MAGIC Sections 1–3 each end with **💬 Ask Genie**: the same insight, asked in plain English instead of SQL. That's how controllers, FP&A analysts, treasury and the revenue team will use these results — no SQL, no waiting on the data team, and every answer respects **Unity Catalog** permissions.
# MAGIC
# MAGIC Step 0.10 creates (or reuses) the **Office of the CFO Genie** agent over this schema. **Why are the AI results written to tables?** The AI Functions run **once**, in the notebook or the job; Genie then reads those tables and never re-runs a forecast or an LLM call per question — answers are fast, cheap and consistent. As later steps build tables (and the `cash_what_if` function in step 2.6), the agent picks them up automatically before each question.

# COMMAND ----------

GENIE_TITLE = f"Office of the CFO Genie ({SCHEMA})"
GENIE_INSTRUCTIONS = f"""This agent answers FP&A, cash and revenue-assurance questions for Halcyon Group's Office of the CFO.
- "Last quarter" is the last full calendar quarter; periods are month-end dates. Opex = line_type 'Opex' in {fq('actuals_vs_plan')}; over budget = variance > 0 for opex.
- fiscal_quarter values look like 'Q3 2026' (with a space). Last quarter = period BETWEEN date_trunc('QUARTER', add_months(current_date(), -3)) AND date_sub(date_trunc('QUARTER', current_date()), 1).
- Why we missed plan: {fq('variance_commentary')} (one row per over-budget cost center for LAST QUARTER ONLY - no date filter needed; read its commentary column) and {fq('variance_drivers')} (ai_top_drivers; contributor lists the segment, change_usd is the USD change; vendor_id=V-1001 is Nimbus Cloud Services, see {fq('vendors')}).
- Forecast next quarter: {fq('forecast_next_qtr')} (monthly amount_forecast with 90% interval amount_lower / amount_upper for the next 3 months; sum the 3 months for the quarter); revenue rows have cost_center = 'Revenue'. Compare with plan from {fq('budget_plan')} for the same periods and cost_center.
- Cash: {fq('cash_forecast_13wk')} (week_number 1-13; bottom_up_balance and top_down_balance). For what-ifs on collections call the table function {fq('cash_what_if')}(dso_shift_days) — negative = collect earlier, e.g. "DSO improves by 1 day" = cash_what_if(-1).
- DSO: {fq('working_capital')}; customers pushing DSO up = {fq('late_payer_anomalies')} WHERE is_anomaly and {fq('expected_receipts')} (overdue amounts).
- Revenue leakage: {fq('leakage_exceptions')} (amount_at_risk_usd, verdict); invoice-level detail and contract-vs-billing mismatches in {fq('contract_to_cash')}.
- Revenue leakage = amount_at_risk_usd WHERE verdict = 'billing_error'. Cases with verdict 'valid_discount' are allowed by the signed contract — report them separately as discounts beyond policy to review, never as leakage.
- Start with the answer (no restating the question). Always state the period and filters used and round money to whole dollars."""
GENIE_EXAMPLES = [
    ("Which cost centers are over budget last quarter?",
     f"SELECT cost_center, round(sum(actual_amount)) AS actual, round(sum(plan_amount)) AS plan, round(sum(variance)) AS variance\nFROM {fq('actuals_vs_plan')}\n"
     f"WHERE line_type = 'Opex' AND period BETWEEN date_trunc('QUARTER', add_months(current_date(), -3)) AND date_sub(date_trunc('QUARTER', current_date()), 1)\n"
     f"GROUP BY cost_center HAVING sum(variance) > 0 ORDER BY variance DESC"),
    ("Why did opex miss plan last quarter?",
     f"SELECT cost_center, fiscal_quarter, variance_usd, variance_pct, commentary\nFROM {fq('variance_commentary')}\nORDER BY variance_usd DESC"),
    ("What's the revenue forecast for next quarter vs plan?",
     f"SELECT round(sum(f.amount_forecast)) AS forecast, round(sum(f.amount_lower)) AS low_90, round(sum(f.amount_upper)) AS high_90, round(sum(p.plan_amount)) AS plan\n"
     f"FROM {fq('forecast_next_qtr')} f JOIN {fq('budget_plan')} p USING (period, cost_center, account)\nWHERE f.cost_center = 'Revenue'"),
    ("How much revenue leakage did we find?",
     f"SELECT exception_type, count(*) AS cases, round(sum(amount_at_risk_usd)) AS leakage_usd\nFROM {fq('leakage_exceptions')}\nWHERE verdict = 'billing_error'\nGROUP BY exception_type ORDER BY leakage_usd DESC"),
]
GENIE_FUNCTIONS = ["cash_what_if"]   # added to the agent once step 2.6 creates it
create_or_sync_genie()
displayHTML(button("Open the Genie agent", GENIE_URL) + button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.11 💬 Ask Genie — explore the source data
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Which 5 customers had the highest invoiced revenue over the last 12 months, and what are their payment terms?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.11b 🛠 Try it yourself
# MAGIC Open the **`contracts`** volume from **step 0.7** and preview two or three PDFs. Notice the three different layouts — an order form, a schedule with a terms table, and a legal-style memorandum. Template-based OCR would need one template per layout; `ai_extract` in section 3 needs none.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.12 Bring your own data
# MAGIC Every downstream cell reads only these tables and the `contracts` volume. To run on real data, skip steps 0.1–0.5, land your contracts in the volume and point the names at your own tables with views. Minimum columns:
# MAGIC
# MAGIC | Table | Required columns | Typical source |
# MAGIC |---|---|---|
# MAGIC | `gl_actuals` / `budget_plan` | `period` (month-end), `entity, cost_center, account, vendor_id, amount` / `period, cost_center, account, plan_amount` | ERP GL (SAP, Oracle, NetSuite, Workday), planning tool (Anaplan, Adaptive, Pigment) |
# MAGIC | `headcount` / `headcount_plan` | `employee_id, cost_center, start_date, end_date, salary, status` / `period, cost_center, planned_headcount` | HRIS (Workday, SuccessFactors) |
# MAGIC | `customers`, `ar_invoices` | `customer_id, name, segment, industry, payment_terms_days` / `invoice_id, customer_id, invoice_date, due_date, paid_date, amount, sku, billed_qty, billed_price` | Billing / ERP AR (Zuora, NetSuite, SAP) |
# MAGIC | `ap_bills`, `bank_transactions` | `bill_id, vendor_id, due_date, paid_date, amount, memo` / `date, account, amount, counterparty, type, memo` | ERP AP; bank feeds (BAI2 / camt.053 via your TMS) |
# MAGIC | `contracts`, `usage`, `price_list` | `contract_id, customer_id, sku, contracted_price, min_commit, start_date, end_date` / `customer_id, sku, date, quantity` / `sku, list_price` | CPQ / CLM (Salesforce CPQ, Ironclad), product telemetry |
# MAGIC | `contracts` volume | One signed PDF per contract, named `<contract_id>.pdf` (scans and Word files work too) | CLM export, DocuSign / SharePoint sync |
# MAGIC
# MAGIC ```sql
# MAGIC -- Example: point the notebook at your ERP's AR invoices
# MAGIC CREATE OR REPLACE VIEW ar_invoices AS
# MAGIC SELECT doc_number AS invoice_id, bill_to_id AS customer_id, CAST(doc_date AS DATE) AS invoice_date, CAST(due_date AS DATE) AS due_date,
# MAGIC        CAST(cleared_date AS DATE) AS paid_date, amount_usd AS amount, item_sku AS sku, quantity AS billed_qty, unit_price AS billed_price
# MAGIC FROM my_catalog.erp.ar_open_and_cleared_items;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. FP&A — why did we miss plan, and where are we heading?
# MAGIC
# MAGIC Four AI Functions, each answering one FP&A question in a single SQL call:
# MAGIC
# MAGIC | Step | Question | AI Function |
# MAGIC |---|---|---|
# MAGIC | 1.1–1.2 | Where are we over / under plan? | plain SQL → gold `actuals_vs_plan` |
# MAGIC | 1.3 | **What caused the variance** — which cost centers, accounts and vendors? | `ai_top_drivers` ×2 → `variance_drivers` |
# MAGIC | 1.4–1.5 | **When did it start?** | `ai_detect_anomalies` → `opex_anomalies` |
# MAGIC | 1.6–1.7 | **Where are we heading** next quarter — with planned headcount as a future covariate? | `ai_forecast` v2 → `forecast_next_qtr` |
# MAGIC | 1.8 | Write the **variance commentary** for the CFO pack | `ai_query` → `variance_commentary` |
# MAGIC | 1.9–1.10 | **Can we trust the forecast?** Backtest it and log MAPE and interval coverage | `ai_forecast` + **MLflow** |
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on **Predictive AI Functions** (Beta) on the **Previews** page. Then re-run the failed cell.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Actuals vs plan with variance, by month, cost center and account
# MAGIC CREATE OR REPLACE TABLE actuals_vs_plan AS
# MAGIC WITH a AS (SELECT period, cost_center, account, sum(amount) AS actual_amount FROM gl_actuals GROUP BY ALL)
# MAGIC SELECT p.period, concat('Q', quarter(p.period), ' ', year(p.period)) AS fiscal_quarter,
# MAGIC        p.cost_center, p.account,
# MAGIC        CASE WHEN p.cost_center = 'Revenue' THEN 'Revenue' ELSE 'Opex' END AS line_type,
# MAGIC        round(a.actual_amount, 2) AS actual_amount, round(p.plan_amount, 2) AS plan_amount,
# MAGIC        round(a.actual_amount - p.plan_amount, 2) AS variance,
# MAGIC        round((a.actual_amount / p.plan_amount - 1) * 100, 1) AS variance_pct
# MAGIC FROM budget_plan p
# MAGIC JOIN a USING (period, cost_center, account);
# MAGIC
# MAGIC -- Last quarter, by cost center
# MAGIC SELECT cost_center, line_type,
# MAGIC        round(sum(actual_amount)) AS actual_usd, round(sum(plan_amount)) AS plan_usd, round(sum(variance)) AS variance_usd,
# MAGIC        round((sum(actual_amount) / sum(plan_amount) - 1) * 100, 1) AS variance_pct,
# MAGIC        CASE WHEN line_type = 'Opex' AND sum(variance) > 0.03 * sum(plan_amount) THEN 'OVER BUDGET'
# MAGIC             WHEN line_type = 'Revenue' AND sum(variance) < -0.01 * sum(plan_amount) THEN 'BELOW PLAN' ELSE 'on plan' END AS status
# MAGIC FROM actuals_vs_plan
# MAGIC WHERE period BETWEEN lfq_start AND lfq_end
# MAGIC GROUP BY ALL
# MAGIC ORDER BY line_type DESC, variance_usd DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Last quarter, by cost center
# MAGIC SELECT cost_center, line_type,
# MAGIC        round(sum(actual_amount)) AS actual_usd, round(sum(plan_amount)) AS plan_usd, round(sum(variance)) AS variance_usd,
# MAGIC        round((sum(actual_amount) / sum(plan_amount) - 1) * 100, 1) AS variance_pct,
# MAGIC        CASE WHEN line_type = 'Opex' AND sum(variance) > 0.03 * sum(plan_amount) THEN 'OVER BUDGET'
# MAGIC             WHEN line_type = 'Revenue' AND sum(variance) < -0.01 * sum(plan_amount) THEN 'BELOW PLAN' ELSE 'on plan' END AS status
# MAGIC FROM actuals_vs_plan
# MAGIC WHERE period BETWEEN CAST(date_trunc('QUARTER', date_sub(date_trunc('QUARTER', current_date()), 1)) AS DATE) AND date_sub(date_trunc('QUARTER', current_date()), 1)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY line_type DESC, variance_usd DESC) SELECT `cost_center`,SUM(`variance_pct`) `column_7455990829` FROM q GROUP BY `cost_center`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Opex vs plan by month — chart this as a line: x = period, y = usd, color = series
# MAGIC SELECT period, 'actual' AS series, round(sum(actual_amount)) AS usd FROM actuals_vs_plan WHERE line_type = 'Opex' AND period >= add_months(lme, -17) GROUP BY ALL
# MAGIC UNION ALL
# MAGIC SELECT period, 'plan', round(sum(plan_amount)) FROM actuals_vs_plan WHERE line_type = 'Opex' AND period >= add_months(lme, -17) GROUP BY ALL
# MAGIC ORDER BY period, series;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Opex vs plan by month — chart this as a line: x = period, y = usd, color = series
# MAGIC SELECT period, 'actual' AS series, round(sum(actual_amount)) AS usd FROM actuals_vs_plan WHERE line_type = 'Opex' AND period >= add_months(last_day(add_months(current_date(), -1)), -17) GROUP BY ALL
# MAGIC UNION ALL
# MAGIC SELECT period, 'plan', round(sum(plan_amount)) FROM actuals_vs_plan WHERE line_type = 'Opex' AND period >= add_months(last_day(add_months(current_date(), -1)), -17) GROUP BY ALL
# MAGIC ORDER BY period, series) SELECT `period`,SUM(`usd`) `column_1307414451`,`series` FROM q GROUP BY `period`,`series`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Find what caused the variance — `ai_top_drivers`, twice
# MAGIC `ai_top_drivers` compares a **test** group with a **control** group and ranks the segments — and **pairs** of segments — that explain the difference. Two runs, two questions (both `metric => 'amount'`, `aggregation => 'sum'`):
# MAGIC
# MAGIC 1. **Where did we miss plan?** Last quarter's actuals (`is_test = true`) stacked on last quarter's plan (`is_test = false`), `dimensions => array('cost_center', 'account')`. The plan has no vendor column, so vendor can't be a dimension here. *This surfaces the hiring spike.*
# MAGIC 2. **Which vendors drove it?** Last quarter's actuals (`is_test = true`) vs the quarter before (`is_test = false`), `dimensions => array('cost_center', 'vendor_id')`. *This surfaces the vendor overspend.*

# COMMAND ----------

# ai_top_drivers runs on Databricks SQL — this cell sends the SQL below, exactly as written, to your SQL warehouse.
run_on_sql_warehouse(r"""
-- 1) Where did we miss plan? Last quarter's actuals (is_test = true) stacked on last quarter's plan (is_test = false)
CREATE OR REPLACE TABLE variance_drivers_vs_plan AS
SELECT * FROM ai_top_drivers(
  input => TABLE(
    SELECT a.cost_center, a.account, k.is_test, CASE WHEN k.is_test THEN a.actual_amount ELSE a.plan_amount END AS amount
    FROM actuals_vs_plan a CROSS JOIN (SELECT explode(array(true, false)) AS is_test) k
    WHERE a.line_type = 'Opex' AND a.period BETWEEN lfq_start AND lfq_end
  ),
  metric      => 'amount',
  is_test     => 'is_test',
  dimensions  => array('cost_center', 'account'),
  aggregation => 'sum'
);

-- 2) Which vendors drove it? Last quarter's actuals (is_test = true) vs the quarter before (is_test = false)
CREATE OR REPLACE TABLE variance_drivers_vs_prior_quarter AS
SELECT * FROM ai_top_drivers(
  input => TABLE(
    SELECT cost_center, coalesce(vendor_id, 'payroll') AS vendor_id, amount, period >= lfq_start AS is_test
    FROM gl_actuals
    WHERE cost_center <> 'Revenue' AND period BETWEEN pq_start AND lfq_end
  ),
  metric      => 'amount',
  is_test     => 'is_test',
  dimensions  => array('cost_center', 'vendor_id'),
  aggregation => 'sum'
);

-- Both results in one table for Genie and the dashboard
CREATE OR REPLACE TABLE variance_drivers AS
SELECT 'Actuals vs plan (last quarter)' AS comparison, array_join(contributor, ' & ') AS contributor,
       round(metric_control) AS control_usd, round(metric_test) AS test_usd, round(change) AS change_usd,
       round(relative_change * 100, 1) AS change_pct, round(support * 100, 1) AS share_of_spend_pct
FROM variance_drivers_vs_plan
UNION ALL
SELECT 'Last quarter vs the quarter before', array_join(contributor, ' & '), round(metric_control), round(metric_test),
       round(change), round(relative_change * 100, 1), round(support * 100, 1)
FROM variance_drivers_vs_prior_quarter;
""", lfq_start=("DATE", "CAST(date_trunc('QUARTER', date_sub(date_trunc('QUARTER', current_date()), 1)) AS DATE)"),
                     lfq_end=("DATE", "date_sub(date_trunc('QUARTER', current_date()), 1)"), pq_start=("DATE", "add_months(CAST(date_trunc('QUARTER', date_sub(date_trunc('QUARTER', current_date()), 1)) AS DATE), -3)"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The top drivers of each comparison (vendor names added for readability)
# MAGIC SELECT d.comparison, d.contributor, coalesce(v.vendor_name, '') AS vendor, d.control_usd, d.test_usd, d.change_usd, d.change_pct
# MAGIC FROM variance_drivers d
# MAGIC LEFT JOIN vendors v ON d.contributor LIKE concat('%vendor_id=', v.vendor_id, '%')
# MAGIC WHERE d.contributor <> 'all'
# MAGIC QUALIFY row_number() OVER (PARTITION BY d.comparison ORDER BY abs(d.change_usd) DESC) <= 6
# MAGIC ORDER BY d.comparison, abs(d.change_usd) DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Flag the anomalies — `ai_detect_anomalies`
# MAGIC `ai_detect_anomalies` scores every month of every cost center's opex against what its own history predicted (`group_col => 'cost_center'`). It answers *when* the problem started — it should flag the **first month of last quarter** for both planted problems. Adjust the `anomaly_threshold` widget to tune sensitivity.
# MAGIC
# MAGIC *Fallback: if the Predictive AI Functions preview isn't enabled, a rolling 12-month baseline runs instead (`method` column).*

# COMMAND ----------

# ai_detect_anomalies runs on Databricks SQL — this cell sends the SQL below, exactly as written, to your SQL warehouse.
run_on_sql_warehouse(r"""
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    -- Fallback: each month vs the previous 12 months of the same cost center (mean ± 3 sd) — illustrative only
    CREATE OR REPLACE TABLE opex_anomalies AS
    WITH m AS (SELECT period, cost_center, sum(amount) AS opex FROM gl_actuals WHERE cost_center <> 'Revenue' GROUP BY ALL),
    w AS (SELECT *, avg(opex) OVER (PARTITION BY cost_center ORDER BY period ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS mu,
                    stddev(opex) OVER (PARTITION BY cost_center ORDER BY period ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS sd,
                    row_number() OVER (PARTITION BY cost_center ORDER BY period) AS n
          FROM m)
    SELECT period, cost_center, opex, mu AS expected, mu - 3 * sd AS lower_bound, mu + 3 * sd AS upper_bound,
           least(1.0, abs(opex - mu) / nullif(4 * sd, 0)) AS anomaly_score,
           n > 8 AND abs(opex - mu) > 3 * sd AS is_anomaly, 'rolling 12-month baseline (fallback)' AS method
    FROM w WHERE n > 8;
  END;

  CREATE OR REPLACE TABLE opex_anomalies AS
  SELECT a.*, 'ai_detect_anomalies' AS method
  FROM ai_detect_anomalies(
    TABLE(SELECT period, cost_center, sum(amount) AS opex FROM gl_actuals WHERE cost_center <> 'Revenue' GROUP BY ALL),
    time_col          => 'period',
    value_col         => 'opex',
    group_col         => 'cost_center',
    anomaly_threshold => anomaly_threshold
  ) a;
END;
""", anomaly_threshold=("DOUBLE", num("anomaly_threshold", 0.95)))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The hiring spike (Customer Success) and the vendor overspend (IT) should both start in the first month of last quarter
# MAGIC SELECT period, cost_center, round(opex) AS opex_usd, round(expected) AS expected_usd,
# MAGIC        round((opex / expected - 1) * 100, 1) AS pct_above_expected, round(anomaly_score, 2) AS anomaly_score, method
# MAGIC FROM opex_anomalies
# MAGIC WHERE is_anomaly
# MAGIC ORDER BY period DESC, anomaly_score DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- The hiring spike (Customer Success) and the vendor overspend (IT) should both start in the first month of last quarter
# MAGIC SELECT period, cost_center, round(opex) AS opex_usd, round(expected) AS expected_usd,
# MAGIC        round((opex / expected - 1) * 100, 1) AS pct_above_expected, round(anomaly_score, 2) AS anomaly_score, method
# MAGIC FROM opex_anomalies
# MAGIC WHERE is_anomaly
# MAGIC ORDER BY period DESC, anomaly_score DESC) SELECT `cost_center`,MAX(`pct_above_expected`) `column_4017352950` FROM q GROUP BY `cost_center`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.6 Forecast next quarter — `ai_forecast` v2 with planned headcount as a future covariate
# MAGIC One SQL call forecasts **every cost center × account** series (`group_col => array('cost_center', 'account')`), revenue and opex alike, with `frequency => 'ME'`, a 90% `prediction_interval_width`, `positive_only => true` and `holiday_region => 'US'`.
# MAGIC
# MAGIC **Planned headcount as a future covariate:** the input joins `headcount_plan` by cost center and month, and adds **three rows per series for the forecast months** with `planned_headcount` filled in and `amount` left `NULL`. `ai_forecast` then conditions next quarter's salaries on the hires you've already planned (revenue rows use company-wide headcount).
# MAGIC
# MAGIC *If AI Forecast V2 isn't enabled in this workspace yet, the cell falls back to version 1 (no covariates) so the rest of the notebook still runs — check the `model_version` column.*

# COMMAND ----------

# ai_forecast runs on Databricks SQL — this cell sends the SQL below, exactly as written, to your SQL warehouse.
FORECAST_SQL = r"""
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    -- Fallback: ai_forecast version 1 (Public Preview) — no covariates
    CREATE OR REPLACE TABLE {target} AS
    SELECT f.period, f.cost_center, f.account, f.amount_forecast, f.amount_upper, f.amount_lower, 'ai_forecast v1 (fallback)' AS model_version
    FROM ai_forecast(
      observed                  => TABLE(SELECT period, cost_center, account, sum(amount) AS amount FROM gl_actuals
                                         WHERE period <= cutoff GROUP BY ALL),
      horizon                   => add_months(cutoff, 4),
      time_col                  => 'period',
      value_col                 => 'amount',
      group_col                 => array('cost_center', 'account'),
      frequency                 => 'ME',
      prediction_interval_width => 0.9
    ) f
    WHERE f.period > cutoff AND f.period <= add_months(cutoff, 3);
  END;

  CREATE OR REPLACE TABLE {target} AS
  SELECT f.period, f.cost_center, f.account, f.amount_forecast, f.amount_upper, f.amount_lower, 'ai_forecast v2 (planned headcount covariate)' AS model_version
  FROM ai_forecast(
    observed => TABLE(
      WITH hc AS (SELECT period, cost_center, planned_headcount FROM headcount_plan
                  UNION ALL SELECT period, 'Revenue', sum(planned_headcount) FROM headcount_plan GROUP BY ALL),
      hist AS (SELECT period, cost_center, account, sum(amount) AS amount FROM gl_actuals WHERE period <= cutoff GROUP BY ALL),
      future AS (SELECT DISTINCT add_months(cutoff, k) AS period, cost_center, account, CAST(NULL AS DOUBLE) AS amount
                 FROM hist LATERAL VIEW explode(array(1, 2, 3)) e AS k)
      SELECT x.period, x.cost_center, x.account, x.amount, hc.planned_headcount
      FROM (SELECT * FROM hist UNION ALL SELECT * FROM future) x
      JOIN hc ON hc.period = x.period AND hc.cost_center = x.cost_center
    ),
    horizon                   => add_months(cutoff, 4),
    time_col                  => 'period',
    value_col                 => 'amount',
    group_col                 => array('cost_center', 'account'),
    covariate_col             => 'planned_headcount',
    prediction_interval_width => 0.9,
    frequency                 => 'ME',
    positive_only             => true,
    holiday_region            => 'US',
    version                   => '2'
  ) f
  WHERE f.period > cutoff AND f.period <= add_months(cutoff, 3);
END;
"""
run_on_sql_warehouse(FORECAST_SQL.replace("{target}", "forecast_next_qtr"), cutoff=("DATE", "last_day(add_months(current_date(), -1))"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Next quarter's forecast with its 90% interval, against the plan
# MAGIC WITH f AS (
# MAGIC   SELECT CASE WHEN cost_center = 'Revenue' THEN 'Revenue' ELSE 'Opex' END AS line_type, cost_center,
# MAGIC          sum(amount_forecast) AS forecast, sum(amount_lower) AS low_90, sum(amount_upper) AS high_90, any_value(model_version) AS model_version
# MAGIC   FROM forecast_next_qtr GROUP BY ALL
# MAGIC ), p AS (
# MAGIC   SELECT cost_center, sum(plan_amount) AS plan FROM budget_plan WHERE period > lme AND period <= add_months(lme, 3) GROUP BY ALL
# MAGIC )
# MAGIC SELECT f.line_type, f.cost_center, round(f.forecast) AS forecast_usd, round(f.low_90) AS low_90_usd, round(f.high_90) AS high_90_usd,
# MAGIC        round(p.plan) AS plan_usd, round((f.forecast / p.plan - 1) * 100, 1) AS forecast_vs_plan_pct, f.model_version
# MAGIC FROM f JOIN p USING (cost_center)
# MAGIC ORDER BY f.line_type DESC, forecast_vs_plan_pct DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Next quarter's forecast with its 90% interval, against the plan
# MAGIC WITH f AS (
# MAGIC   SELECT CASE WHEN cost_center = 'Revenue' THEN 'Revenue' ELSE 'Opex' END AS line_type, cost_center,
# MAGIC          sum(amount_forecast) AS forecast, sum(amount_lower) AS low_90, sum(amount_upper) AS high_90, any_value(model_version) AS model_version
# MAGIC   FROM forecast_next_qtr GROUP BY ALL
# MAGIC ), p AS (
# MAGIC   SELECT cost_center, sum(plan_amount) AS plan FROM budget_plan WHERE period > last_day(add_months(current_date(), -1)) AND period <= add_months(last_day(add_months(current_date(), -1)), 3) GROUP BY ALL
# MAGIC )
# MAGIC SELECT f.line_type, f.cost_center, round(f.forecast) AS forecast_usd, round(f.low_90) AS low_90_usd, round(f.high_90) AS high_90_usd,
# MAGIC        round(p.plan) AS plan_usd, round((f.forecast / p.plan - 1) * 100, 1) AS forecast_vs_plan_pct, f.model_version
# MAGIC FROM f JOIN p USING (cost_center)
# MAGIC ORDER BY f.line_type DESC, forecast_vs_plan_pct DESC) SELECT `cost_center`,SUM(`forecast_vs_plan_pct`) `column_9756581761` FROM q GROUP BY `cost_center`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.8 Write the commentary — `ai_query`
# MAGIC Every month someone writes the variance commentary for the CFO pack. `ai_query` calls a **pay-per-token Foundation Model API** endpoint (the `llm_endpoint` widget) to write a **3-sentence note for each over-budget cost center** — grounded in that cost center's drivers from step 1.3 and the related AP memos from step 0.3, so the note cites real numbers instead of guessing.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE variance_commentary AS
# MAGIC WITH over AS (
# MAGIC   SELECT cost_center, sum(actual_amount) AS actual, sum(plan_amount) AS plan, sum(variance) AS variance
# MAGIC   FROM actuals_vs_plan WHERE line_type = 'Opex' AND period BETWEEN lfq_start AND lfq_end
# MAGIC   GROUP BY ALL HAVING sum(variance) > 0.03 * sum(plan_amount)
# MAGIC ), drv AS (
# MAGIC   SELECT o.cost_center, concat_ws('; ', collect_list(concat(d.comparison, ': ', d.contributor, ' changed by $', format_number(d.change_usd, 0)))) AS drivers
# MAGIC   FROM over o JOIN variance_drivers d ON d.contributor LIKE concat('%cost_center=', o.cost_center, '%')
# MAGIC   GROUP BY ALL
# MAGIC ), memo AS (
# MAGIC   SELECT b.cost_center, concat_ws(' | ', slice(collect_set(concat(v.vendor_name, ': ', b.memo)), 1, 5)) AS memos
# MAGIC   FROM ap_bills b JOIN vendors v USING (vendor_id)
# MAGIC   WHERE b.bill_date BETWEEN lfq_start AND lfq_end AND b.vendor_id <> 'V-9000'
# MAGIC   GROUP BY ALL
# MAGIC ), hc AS (
# MAGIC   SELECT cost_center,
# MAGIC          max_by(planned_headcount, period) AS planned_headcount_end_of_quarter
# MAGIC   FROM headcount_plan WHERE period BETWEEN lfq_start AND lfq_end GROUP BY ALL
# MAGIC )
# MAGIC SELECT o.cost_center, concat('Q', quarter(lfq_end), ' ', year(lfq_end)) AS fiscal_quarter,
# MAGIC        round(o.actual) AS actual_usd, round(o.plan) AS plan_usd, round(o.variance) AS variance_usd,
# MAGIC        round(o.variance / o.plan * 100, 1) AS variance_pct, d.drivers, m.memos,
# MAGIC        ai_query(llm_endpoint, concat(
# MAGIC          'You are an FP&A analyst writing the variance commentary for the CFO pack. Write exactly 3 sentences, plain text, no bullets. ',
# MAGIC          'Sentence 1: the variance in dollars and percent. Sentence 2: the main driver, citing the numbers below. Sentence 3: the recommended action. ',
# MAGIC          'Cost center: ', o.cost_center, '. Quarter: Q', quarter(lfq_end), ' ', year(lfq_end), '. Actual $', format_number(o.actual, 0),
# MAGIC          ' vs plan $', format_number(o.plan, 0), ' (variance $', format_number(o.variance, 0), ', ', round(o.variance / o.plan * 100, 1), '%). ',
# MAGIC          'Drivers from ai_top_drivers: ', coalesce(d.drivers, 'none'), '. Related AP memos: ', coalesce(m.memos, 'none'),
# MAGIC          '. Planned headcount at quarter end: ', coalesce(CAST(h.planned_headcount_end_of_quarter AS STRING), 'n/a'), '.')) AS commentary
# MAGIC FROM over o
# MAGIC LEFT JOIN drv d USING (cost_center)
# MAGIC LEFT JOIN memo m USING (cost_center)
# MAGIC LEFT JOIN hc h USING (cost_center);
# MAGIC
# MAGIC SELECT cost_center, variance_usd, variance_pct, commentary FROM variance_commentary ORDER BY variance_usd DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.9–1.10 Check accuracy in MLflow — can we trust the forecast?
# MAGIC **1.9** re-runs the exact same `ai_forecast` SQL with history **cut off 3 months early**, and writes `forecast_backtest`. **1.10** compares it with what actually happened and logs **MAPE** and **90% interval coverage** per cost center to the notebook's **MLflow experiment** — along with a simple **seasonal-naive challenger** (same month last year × trailing growth), so you can see the AI forecast earn its place. *Going further: log a many-model Prophet challenger in the same experiment (see Build more).*
# MAGIC
# MAGIC *Expect the biggest misses where the planted problems start (IT and Customer Success): the backtest's training window ends right as they begin, so no forecast — AI or naive — sees them coming. A 90% interval should cover about 90% of actuals; where it doesn't, look harder — that's why you run anomaly detection next to forecasting.*

# COMMAND ----------

# The forecast SQL from step 1.6, with the cutoff moved back 3 months, sent to your SQL warehouse
run_on_sql_warehouse(FORECAST_SQL.replace("{target}", "forecast_backtest"), cutoff=("DATE", "last_day(add_months(current_date(), -4))"))
display(spark.sql("""
SELECT b.period, b.cost_center, b.account, round(b.amount_forecast) AS forecast_usd, round(a.actual) AS actual_usd,
       round(b.amount_lower) AS low_90_usd, round(b.amount_upper) AS high_90_usd, b.model_version
FROM forecast_backtest b
JOIN (SELECT period, cost_center, account, sum(amount) AS actual FROM gl_actuals GROUP BY ALL) a USING (period, cost_center, account)
ORDER BY cost_center, account, period LIMIT 30"""))

# COMMAND ----------

import mlflow, pandas as pd
bt = spark.sql("""
WITH a AS (SELECT period, cost_center, account, sum(amount) AS actual FROM gl_actuals GROUP BY ALL),
ly AS (SELECT period, cost_center, account, actual,
              lag(actual, 12) OVER (PARTITION BY cost_center, account ORDER BY period) AS same_month_ly FROM a),
g AS (SELECT cost_center, account, sum(CASE WHEN period > add_months(lme, -15) AND period <= add_months(lme, -3) THEN actual END) /
                                   sum(CASE WHEN period > add_months(lme, -27) AND period <= add_months(lme, -15) THEN actual END) AS growth
      FROM a GROUP BY ALL)
SELECT b.period, b.cost_center, b.account, b.amount_forecast AS forecast, b.amount_lower AS lower, b.amount_upper AS upper,
       ly.actual, ly.same_month_ly * g.growth AS naive, b.model_version
FROM forecast_backtest b JOIN ly USING (period, cost_center, account) JOIN g USING (cost_center, account)
""").toPandas()
bt["ape"] = (bt.forecast - bt.actual).abs() / bt.actual.abs()
bt["ape_naive"] = (bt.naive - bt.actual).abs() / bt.actual.abs()
bt["covered"] = (bt.actual >= bt.lower) & (bt.actual <= bt.upper)
by_cc = bt.groupby("cost_center").agg(mape_pct=("ape", lambda x: 100 * x.mean()), naive_mape_pct=("ape_naive", lambda x: 100 * x.mean()),
                                      coverage_pct=("covered", lambda x: 100 * x.mean())).round(1).reset_index()
experiment()
with mlflow.start_run(run_name="1.10 ai_forecast backtest (3 months)") as run:
    mlflow.log_params({"function": "ai_forecast", "model_version": bt.model_version.iloc[0], "horizon_months": 3,
                       "frequency": "ME", "prediction_interval_width": 0.9, "group_col": "cost_center, account", "series": bt.groupby(["cost_center", "account"]).ngroups})
    mlflow.log_metrics({"mape_pct": round(100 * bt.ape.mean(), 2), "interval_coverage_pct": round(100 * bt.covered.mean(), 1)})
    for r in by_cc.itertuples():
        key = re.sub(r"[^A-Za-z0-9]+", "_", r.cost_center).strip("_").lower()
        mlflow.log_metrics({f"mape_pct_{key}": r.mape_pct, f"coverage_pct_{key}": r.coverage_pct})
    mlflow.log_table(by_cc, "backtest_by_cost_center.json")
    show_run(run, "AI Function: forecast backtest")
with mlflow.start_run(run_name="1.10 challenger: seasonal naive") as run:
    mlflow.log_params({"function": "seasonal naive (same month last year x trailing growth)", "horizon_months": 3})
    mlflow.log_metric("mape_pct", round(100 * bt.ape_naive.mean(), 2))
    show_run(run, "Challenger")
display(by_cc.sort_values("mape_pct", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.11 💬 Ask Genie — FP&A
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Why did opex miss plan last quarter?", before=("Section 1 (actuals vs plan, drivers and commentary)", 89))

# COMMAND ----------

ask("What's the revenue forecast for next quarter, and how does it compare with plan?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.13 🛠 Try it yourself
# MAGIC In **step 1.3**, add `'entity'` to the dimensions of the second `ai_top_drivers` call (and to its input `SELECT`), then re-run it. Does the overspend sit in the US or the UK entity? Then open `variance_drivers` in Unity Catalog → **Lineage** to follow it back through the two raw `ai_top_drivers` tables to `actuals_vs_plan` and `gl_actuals`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cash — what will our cash position be, and who's putting it at risk?
# MAGIC
# MAGIC | Step | Question | Capability |
# MAGIC |---|---|---|
# MAGIC | 2.1 | How healthy is working capital — DSO, DPO, AR aging? | SQL → `working_capital` |
# MAGIC | 2.2–2.3 | **When will each open invoice actually get paid?** | `ai_predict_value` (target `days_to_pay`) → `expected_receipts` |
# MAGIC | 2.4 | **Which customers pay later and later?** | `ai_detect_anomalies` → `late_payer_anomalies` |
# MAGIC | 2.5 | **What's our cash position for the next 13 weeks** — bottom-up and top-down? | SQL + `ai_forecast` (weekly) → `cash_forecast_13wk` |
# MAGIC | 2.6 | **What if** we collect faster or slower? | a Unity Catalog **SQL table function**, `cash_what_if(dso_shift_days)` — the same one Genie calls |
# MAGIC | 2.7 | **Act:** draft the collections emails | `ai_gen` → `collections_outreach` |
# MAGIC | 2.8 | **Can we trust the pay dates?** | **MLflow**: MAE in days on a 20% hold-out |

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE working_capital AS
# MAGIC WITH ar AS (SELECT sum(CASE WHEN paid_date IS NULL THEN amount END) AS open_ar,
# MAGIC                    sum(CASE WHEN invoice_date > date_sub(current_date(), 90) THEN amount END) AS revenue_90d FROM ar_invoices),
# MAGIC ap AS (SELECT sum(CASE WHEN paid_date IS NULL AND bill_date <= current_date() THEN amount END) AS open_ap,   -- vendor bills (payroll excluded)
# MAGIC               sum(CASE WHEN bill_date > date_sub(current_date(), 90) AND bill_date <= current_date() THEN amount END) AS spend_90d
# MAGIC        FROM ap_bills WHERE vendor_id <> 'V-9000'),
# MAGIC aging AS (SELECT sum(CASE WHEN datediff(current_date(), due_date) <= 0 THEN amount END) AS current_usd,
# MAGIC                  sum(CASE WHEN datediff(current_date(), due_date) BETWEEN 1 AND 30 THEN amount END) AS past_due_1_30_usd,
# MAGIC                  sum(CASE WHEN datediff(current_date(), due_date) BETWEEN 31 AND 60 THEN amount END) AS past_due_31_60_usd,
# MAGIC                  sum(CASE WHEN datediff(current_date(), due_date) > 60 THEN amount END) AS past_due_60_plus_usd
# MAGIC           FROM ar_invoices WHERE paid_date IS NULL)
# MAGIC SELECT current_date() AS as_of, round(ar.open_ar) AS open_ar_usd, round(ar.open_ar / ar.revenue_90d * 90, 1) AS dso_days,
# MAGIC        round(ap.open_ap) AS open_ap_usd, round(ap.open_ap / ap.spend_90d * 90, 1) AS dpo_days,
# MAGIC        round(coalesce(current_usd, 0)) AS current_usd, round(coalesce(past_due_1_30_usd, 0)) AS past_due_1_30_usd,
# MAGIC        round(coalesce(past_due_31_60_usd, 0)) AS past_due_31_60_usd, round(coalesce(past_due_60_plus_usd, 0)) AS past_due_60_plus_usd
# MAGIC FROM ar, ap, aging;
# MAGIC
# MAGIC SELECT * FROM working_capital;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Predict when each open invoice gets paid — `ai_predict_value`
# MAGIC `ai_predict_value` **trains a regression model on the rows where `days_to_pay` is known and scores the rows where it's NULL — in one SQL call.** Three kinds of rows get a prediction:
# MAGIC
# MAGIC * **open invoices** (not paid yet),
# MAGIC * **scheduled billings** — next two months' invoices at each contract's run rate, so weeks 6–13 of the cash forecast aren't empty,
# MAGIC * a **20% hold-out** of last year's paid invoices, with `days_to_pay` set to NULL in a copy of the input — so step 2.8 can measure the error against what really happened.
# MAGIC
# MAGIC Features: the customer's **past average days late** (paid invoices only, **before each invoice date**, so the feature doesn't leak the answer), amount, `payment_terms_days`, invoice month, segment and industry. **Expected pay date = invoice date + predicted days to pay** (never earlier than two days from today, for invoices already overdue).

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Silver: one row per invoice — actual, open or scheduled — with features known at invoice time
# MAGIC CREATE OR REPLACE TABLE invoice_payment_features AS
# MAGIC WITH sched AS (   -- next two month-ends at each active contract's run rate
# MAGIC   SELECT concat('SCHED-', i.customer_id, '-', date_format(add_months(lme, k), 'yyyyMM')) AS invoice_id, i.customer_id,
# MAGIC          add_months(lme, k) AS invoice_date, date_add(add_months(lme, k), CAST(c.payment_terms_days AS INT)) AS due_date,
# MAGIC          CAST(NULL AS DATE) AS paid_date, i.amount, 'scheduled billing' AS kind
# MAGIC   FROM ar_invoices i JOIN customers c USING (customer_id)
# MAGIC   LATERAL VIEW explode(array(1, 2)) e AS k
# MAGIC   WHERE i.invoice_date = lme
# MAGIC ), inv AS (
# MAGIC   SELECT invoice_id, customer_id, invoice_date, due_date, paid_date, amount,
# MAGIC          CASE WHEN paid_date IS NULL THEN 'open invoice'
# MAGIC               WHEN invoice_date > add_months(lme, -12) AND pmod(xxhash64(invoice_id), 5) = 0 THEN 'hold-out'
# MAGIC               ELSE 'training' END AS kind
# MAGIC   FROM ar_invoices
# MAGIC   UNION ALL SELECT * FROM sched
# MAGIC ), hist AS (      -- customer's payment history strictly before each invoice date
# MAGIC   SELECT i.invoice_id, avg(datediff(p.paid_date, p.due_date)) AS past_avg_days_late, count(p.invoice_id) AS past_paid_invoices
# MAGIC   FROM inv i LEFT JOIN ar_invoices p ON p.customer_id = i.customer_id AND p.paid_date < i.invoice_date
# MAGIC   GROUP BY i.invoice_id
# MAGIC )
# MAGIC SELECT i.invoice_id, i.kind, i.customer_id, c.name AS customer_name, c.segment, c.industry, c.payment_terms_days,
# MAGIC        i.invoice_date, i.due_date, i.paid_date, i.amount, month(i.invoice_date) AS invoice_month,
# MAGIC        round(coalesce(h.past_avg_days_late, 0), 1) AS past_avg_days_late, h.past_paid_invoices,
# MAGIC        datediff(i.paid_date, i.invoice_date) AS actual_days_to_pay
# MAGIC FROM inv i JOIN customers c USING (customer_id) LEFT JOIN hist h USING (invoice_id);
# MAGIC
# MAGIC -- ai_predict_value: trains on the 'training' rows, scores open, scheduled and hold-out rows
# MAGIC CREATE OR REPLACE TABLE payment_predictions AS
# MAGIC SELECT *
# MAGIC FROM ai_predict_value(
# MAGIC   input => TABLE(
# MAGIC     SELECT invoice_id, kind, customer_id, customer_name, invoice_date, due_date, actual_days_to_pay,
# MAGIC            CASE WHEN kind = 'training' THEN actual_days_to_pay END AS days_to_pay,
# MAGIC            past_avg_days_late, past_paid_invoices, amount, payment_terms_days, invoice_month, segment, industry
# MAGIC     FROM invoice_payment_features
# MAGIC     WHERE invoice_date > add_months(lme, -24)
# MAGIC   ),
# MAGIC   target_col   => 'days_to_pay',
# MAGIC   feature_cols => array('past_avg_days_late', 'past_paid_invoices', 'amount', 'payment_terms_days', 'invoice_month', 'segment', 'industry')
# MAGIC );
# MAGIC
# MAGIC SELECT kind, count(*) AS invoices, round(avg(days_to_pay_prediction), 1) AS avg_predicted_days_to_pay
# MAGIC FROM payment_predictions WHERE days_to_pay IS NULL GROUP BY ALL;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE expected_receipts AS
# MAGIC SELECT p.invoice_id, p.kind, p.customer_id, p.customer_name, f.segment, p.invoice_date, p.due_date, p.amount,
# MAGIC        round(p.days_to_pay_prediction, 1) AS predicted_days_to_pay,
# MAGIC        greatest(date_add(p.invoice_date, CAST(round(p.days_to_pay_prediction) AS INT)), date_add(current_date(), 2)) AS expected_pay_date,
# MAGIC        greatest(datediff(current_date(), p.due_date), 0) AS days_past_due
# MAGIC FROM payment_predictions p JOIN invoice_payment_features f USING (invoice_id)
# MAGIC WHERE p.kind IN ('open invoice', 'scheduled billing');
# MAGIC
# MAGIC SELECT customer_name, kind, invoice_id, invoice_date, due_date, round(amount) AS amount_usd, predicted_days_to_pay, expected_pay_date, days_past_due
# MAGIC FROM expected_receipts ORDER BY amount DESC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Catch customers who pay later and later — `ai_detect_anomalies`
# MAGIC A customer drifting from 45 to 75 days doesn't trip any single-invoice alert, but it moves DSO. `ai_detect_anomalies` runs on each customer's **monthly average days-to-pay** (`group_col => 'customer_id'`), limited to customers with **at least 12 months of paid invoices** — the first 8 points of each series are context only.

# COMMAND ----------

run_on_sql_warehouse(r"""
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    -- Fallback: each month vs the same customer's 12 months before last quarter (mean + 3 sd, at least 3 days) — illustrative only
    CREATE OR REPLACE TABLE late_payer_anomalies AS
    WITH m AS (SELECT customer_id, invoice_date AS period, avg(datediff(paid_date, invoice_date)) AS days_to_pay
               FROM ar_invoices WHERE paid_date IS NOT NULL GROUP BY ALL),
    ok AS (SELECT customer_id FROM m GROUP BY customer_id HAVING count(*) >= 12),
    w AS (SELECT m.*, avg(days_to_pay) OVER (PARTITION BY customer_id ORDER BY period ROWS BETWEEN 15 PRECEDING AND 4 PRECEDING) AS mu,
                      stddev(days_to_pay) OVER (PARTITION BY customer_id ORDER BY period ROWS BETWEEN 15 PRECEDING AND 4 PRECEDING) AS sd,
                      row_number() OVER (PARTITION BY customer_id ORDER BY period) AS n
          FROM m JOIN ok USING (customer_id))
    SELECT period, customer_id, days_to_pay, mu AS expected, mu - 3 * greatest(sd, 1) AS lower_bound, mu + 3 * greatest(sd, 1) AS upper_bound,
           least(1.0, (days_to_pay - mu) / (4 * greatest(sd, 1))) AS anomaly_score,
           n > 8 AND days_to_pay > mu + greatest(3 * sd, 3) AS is_anomaly, 'rolling baseline (fallback)' AS method
    FROM w WHERE n > 8;
  END;

  CREATE OR REPLACE TABLE late_payer_anomalies AS
  SELECT a.*, 'ai_detect_anomalies' AS method
  FROM ai_detect_anomalies(
    TABLE(
      WITH m AS (SELECT customer_id, invoice_date AS period, avg(datediff(paid_date, invoice_date)) AS days_to_pay
                 FROM ar_invoices WHERE paid_date IS NOT NULL GROUP BY ALL)
      SELECT * FROM m WHERE customer_id IN (SELECT customer_id FROM m GROUP BY customer_id HAVING count(*) >= 12)
    ),
    time_col          => 'period',
    value_col         => 'days_to_pay',
    group_col         => 'customer_id',
    anomaly_threshold => anomaly_threshold
  ) a;
END;
""", anomaly_threshold=("DOUBLE", num("anomaly_threshold", 0.95)))
display(spark.sql("""
SELECT c.name AS customer, c.segment, count_if(a.is_anomaly AND a.period > add_months(lme, -6)) AS anomalous_months_last_6,
       round(avg(CASE WHEN a.period > add_months(lme, -3) THEN a.days_to_pay END), 1) AS days_to_pay_last_3m,
       round(avg(CASE WHEN a.period BETWEEN add_months(lme, -15) AND add_months(lme, -12) THEN a.days_to_pay END), 1) AS days_to_pay_a_year_ago,
       c.payment_terms_days, any_value(a.method) AS method
FROM late_payer_anomalies a JOIN customers c USING (customer_id)
GROUP BY ALL HAVING anomalous_months_last_6 >= 2
ORDER BY days_to_pay_last_3m - days_to_pay_a_year_ago DESC"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Build the 13-week cash forecast — two ways
# MAGIC Treasury trusts a forecast when two independent methods agree:
# MAGIC
# MAGIC * **Bottom-up:** today's bank balance **+ expected AR receipts** (step 2.3, by expected pay date) **− scheduled AP payments** (open and scheduled bills by due date, payroll included).
# MAGIC * **Top-down check:** `ai_forecast` on **weekly net cash flow** from `bank_transactions`, with `frequency => 'W'` and 13 weeks of horizon.
# MAGIC
# MAGIC Both land in `cash_forecast_13wk`, side by side, with the top-down interval.

# COMMAND ----------

run_on_sql_warehouse(r"""
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    CREATE OR REPLACE TABLE cash_topdown_forecast AS
    SELECT f.*, 'ai_forecast v1 (fallback)' AS model_version
    FROM ai_forecast(
      observed                  => TABLE(SELECT date_add(CAST(date_trunc('WEEK', date) AS DATE), 6) AS week_end, sum(amount) AS net_cash_flow
                                         FROM bank_transactions WHERE type <> 'Opening balance'
                                           AND date < CAST(date_trunc('WEEK', current_date()) AS DATE) GROUP BY ALL),
      horizon                   => date_add(CAST(date_trunc('WEEK', current_date()) AS DATE), 7 * 13 + 7),
      time_col                  => 'week_end',
      value_col                 => 'net_cash_flow',
      frequency                 => 'W',
      prediction_interval_width => 0.8
    ) f;
  END;

  CREATE OR REPLACE TABLE cash_topdown_forecast AS
  SELECT f.*, 'ai_forecast v2' AS model_version
  FROM ai_forecast(
    observed                  => TABLE(SELECT date_add(CAST(date_trunc('WEEK', date) AS DATE), 6) AS week_end, sum(amount) AS net_cash_flow
                                       FROM bank_transactions WHERE type <> 'Opening balance'
                                         AND date < CAST(date_trunc('WEEK', current_date()) AS DATE) GROUP BY ALL),
    horizon                   => date_add(CAST(date_trunc('WEEK', current_date()) AS DATE), 7 * 13 + 7),
    time_col                  => 'week_end',
    value_col                 => 'net_cash_flow',
    frequency                 => 'W',
    prediction_interval_width => 0.8,
    holiday_region            => 'US',
    version                   => '2'
  ) f;
END;
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE cash_forecast_13wk AS
# MAGIC WITH weeks AS (
# MAGIC   SELECT k AS week_number, date_add(CAST(date_trunc('WEEK', current_date()) AS DATE), 7 * k - 1) AS week_end
# MAGIC   FROM (SELECT explode(sequence(1, 13)) AS k)
# MAGIC ), opening AS (SELECT sum(amount) AS balance FROM bank_transactions),
# MAGIC inflow AS (
# MAGIC   SELECT w.week_number, sum(r.amount) AS inflows FROM weeks w
# MAGIC   JOIN expected_receipts r ON r.expected_pay_date BETWEEN date_sub(w.week_end, 6) AND w.week_end OR (w.week_number = 1 AND r.expected_pay_date < date_sub(w.week_end, 6))
# MAGIC   GROUP BY ALL
# MAGIC ), outflow AS (
# MAGIC   SELECT w.week_number, sum(b.amount) AS outflows FROM weeks w
# MAGIC   JOIN ap_bills b ON b.paid_date IS NULL AND (b.due_date BETWEEN date_sub(w.week_end, 6) AND w.week_end OR (w.week_number = 1 AND b.due_date < date_sub(w.week_end, 6)))
# MAGIC   GROUP BY ALL
# MAGIC ), td AS (
# MAGIC   SELECT w.week_number, f.net_cash_flow_forecast AS td_net, f.net_cash_flow_lower AS td_low, f.net_cash_flow_upper AS td_high, f.model_version
# MAGIC   FROM weeks w JOIN cash_topdown_forecast f ON f.week_end = w.week_end
# MAGIC ), j AS (
# MAGIC   SELECT w.*, coalesce(i.inflows, 0) AS inflows, coalesce(o.outflows, 0) AS outflows, td.td_net, td.td_low, td.td_high, td.model_version
# MAGIC   FROM weeks w LEFT JOIN inflow i USING (week_number) LEFT JOIN outflow o USING (week_number) LEFT JOIN td USING (week_number)
# MAGIC )
# MAGIC SELECT week_number, week_end,
# MAGIC        round(inflows) AS bottom_up_inflows, round(outflows) AS bottom_up_outflows, round(inflows - outflows) AS bottom_up_net,
# MAGIC        round((SELECT balance FROM opening) + sum(inflows - outflows) OVER (ORDER BY week_number)) AS bottom_up_balance,
# MAGIC        round(td_net) AS top_down_net,
# MAGIC        round((SELECT balance FROM opening) + sum(td_net) OVER (ORDER BY week_number)) AS top_down_balance,
# MAGIC        round((SELECT balance FROM opening) + sum(td_low) OVER (ORDER BY week_number)) AS top_down_balance_low,
# MAGIC        round((SELECT balance FROM opening) + sum(td_high) OVER (ORDER BY week_number)) AS top_down_balance_high,
# MAGIC        model_version AS top_down_model
# MAGIC FROM j;
# MAGIC
# MAGIC SELECT week_number, week_end, bottom_up_inflows, bottom_up_outflows, bottom_up_balance, top_down_balance,
# MAGIC        top_down_balance_low, top_down_balance_high,
# MAGIC        round((bottom_up_balance / top_down_balance - 1) * 100, 1) AS gap_pct
# MAGIC FROM cash_forecast_13wk ORDER BY week_number;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 13-week cash — chart this as a line: x = week_end, y = balance_usd, color = method
# MAGIC SELECT week_end, 'bottom-up' AS method, bottom_up_balance AS balance_usd FROM cash_forecast_13wk
# MAGIC UNION ALL SELECT week_end, 'top-down (ai_forecast)', top_down_balance FROM cash_forecast_13wk
# MAGIC UNION ALL SELECT week_end, 'top-down low (80%)', top_down_balance_low FROM cash_forecast_13wk
# MAGIC UNION ALL SELECT week_end, 'top-down high (80%)', top_down_balance_high FROM cash_forecast_13wk
# MAGIC ORDER BY week_end, method;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- 13-week cash — chart this as a line: x = week_end, y = balance_usd, color = method
# MAGIC SELECT week_end, 'bottom-up' AS method, bottom_up_balance AS balance_usd FROM cash_forecast_13wk
# MAGIC UNION ALL SELECT week_end, 'top-down (ai_forecast)', top_down_balance FROM cash_forecast_13wk
# MAGIC UNION ALL SELECT week_end, 'top-down low (80%)', top_down_balance_low FROM cash_forecast_13wk
# MAGIC UNION ALL SELECT week_end, 'top-down high (80%)', top_down_balance_high FROM cash_forecast_13wk
# MAGIC ORDER BY week_end, method) SELECT `week_end`,SUM(`balance_usd`) `column_1134558141`,`method` FROM q GROUP BY `week_end`,`method`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 What-if — a Unity Catalog SQL table function
# MAGIC `cash_what_if(dso_shift_days)` shifts **every expected receipt date** by N days and re-runs the bottom-up roll-up. It's a governed function in Unity Catalog, so the logic exists in **one place**: the notebook widget calls it here, the dashboard can call it, and **Genie calls the same function** when someone asks *"what happens to cash if DSO improves by 1 day?"*. Change the **`dso_shift_days`** widget, re-run S.2, then this step.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {fq('cash_what_if')}(dso_shift_days INT COMMENT 'Days to shift every expected receipt: negative = collect earlier (DSO improves), positive = later')
RETURNS TABLE (week_number INT, week_end DATE, inflows DOUBLE, outflows DOUBLE, projected_balance DOUBLE, baseline_balance DOUBLE, change_vs_baseline DOUBLE)
COMMENT 'What-if on the 13-week bottom-up cash forecast: shift every expected AR receipt by dso_shift_days and re-run the roll-up. Example: cash_what_if(-1) = DSO improves by 1 day.'
RETURN
  WITH weeks AS (SELECT week_number, week_end, bottom_up_balance FROM {fq('cash_forecast_13wk')}),
  opening AS (SELECT sum(amount) AS balance FROM {fq('bank_transactions')}),
  r AS (SELECT greatest(date_add(expected_pay_date, dso_shift_days), current_date()) AS d, amount FROM {fq('expected_receipts')}),
  inflow AS (SELECT w.week_number, sum(r.amount) AS inflows FROM weeks w
             JOIN r ON r.d BETWEEN date_sub(w.week_end, 6) AND w.week_end OR (w.week_number = 1 AND r.d < date_sub(w.week_end, 6)) GROUP BY ALL),
  outflow AS (SELECT w.week_number, sum(b.amount) AS outflows FROM weeks w
              JOIN {fq('ap_bills')} b ON b.paid_date IS NULL AND (b.due_date BETWEEN date_sub(w.week_end, 6) AND w.week_end
                                         OR (w.week_number = 1 AND b.due_date < date_sub(w.week_end, 6))) GROUP BY ALL),
  j AS (SELECT w.week_number, w.week_end, w.bottom_up_balance, coalesce(i.inflows, 0) AS inflows, coalesce(o.outflows, 0) AS outflows
        FROM weeks w LEFT JOIN inflow i USING (week_number) LEFT JOIN outflow o USING (week_number))
  SELECT week_number, week_end, round(inflows), round(outflows),
         round((SELECT balance FROM opening) + sum(inflows - outflows) OVER (ORDER BY week_number)),
         bottom_up_balance,
         round((SELECT balance FROM opening) + sum(inflows - outflows) OVER (ORDER BY week_number) - bottom_up_balance)
  FROM j
""")
display(spark.sql("SELECT dso_shift_days AS what_if_dso_shift_days, * FROM cash_what_if(dso_shift_days) ORDER BY week_number"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 Act — `ai_gen` drafts the collections emails
# MAGIC For the **10 customers putting the most cash at risk** (customers flagged as paying later and later in step 2.4 first, then the largest overdue balances), `ai_gen` drafts a firm, friendly collections email that cites the invoices, the amount and the trend. The collections team reviews and sends; nobody starts from a blank page.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE collections_outreach AS
# MAGIC WITH flagged AS (
# MAGIC   SELECT customer_id, count_if(is_anomaly AND period > add_months(lme, -6)) AS anomalous_months,
# MAGIC          avg(CASE WHEN period > add_months(lme, -3) THEN days_to_pay END) AS dtp_recent,
# MAGIC          avg(CASE WHEN period BETWEEN add_months(lme, -15) AND add_months(lme, -12) THEN days_to_pay END) AS dtp_year_ago
# MAGIC   FROM late_payer_anomalies GROUP BY ALL
# MAGIC ), overdue AS (
# MAGIC   SELECT customer_id, count(*) AS overdue_invoices, sum(amount) AS overdue_usd, max(days_past_due) AS max_days_past_due,
# MAGIC          concat_ws(', ', collect_list(invoice_id)) AS invoice_ids
# MAGIC   FROM expected_receipts WHERE kind = 'open invoice' AND days_past_due > 0 GROUP BY ALL
# MAGIC ), ranked AS (
# MAGIC   SELECT c.customer_id, c.name AS customer_name, c.segment, c.payment_terms_days, o.overdue_invoices, round(o.overdue_usd) AS overdue_usd,
# MAGIC          o.max_days_past_due, o.invoice_ids, round(f.dtp_recent, 1) AS days_to_pay_recent, round(f.dtp_year_ago, 1) AS days_to_pay_year_ago,
# MAGIC          coalesce(f.anomalous_months, 0) >= 2 AS paying_later_and_later
# MAGIC   FROM overdue o JOIN customers c USING (customer_id) LEFT JOIN flagged f USING (customer_id)
# MAGIC   ORDER BY paying_later_and_later DESC, overdue_usd DESC
# MAGIC   LIMIT 10
# MAGIC )
# MAGIC SELECT *, ai_gen(concat(
# MAGIC   'Draft a short, firm but friendly collections email (subject line + 4-6 sentences, sign as "Halcyon Accounts Receivable") to the accounts payable team at ',
# MAGIC   customer_name, '. Facts: ', overdue_invoices, ' overdue invoice(s) (', invoice_ids, ') totalling $', format_number(overdue_usd, 0),
# MAGIC   ', oldest ', max_days_past_due, ' days past due; payment terms net ', payment_terms_days, '. ',
# MAGIC   CASE WHEN paying_later_and_later THEN concat('Their average time to pay has grown from ', days_to_pay_year_ago, ' to ', days_to_pay_recent,
# MAGIC        ' days over the past year - mention it politely and ask whether anything changed in their approval process. ') ELSE '' END,
# MAGIC   'Ask for a payment date and offer a call.')) AS email_draft
# MAGIC FROM ranked;
# MAGIC
# MAGIC SELECT customer_name, overdue_usd, max_days_past_due, paying_later_and_later, email_draft FROM collections_outreach ORDER BY overdue_usd DESC;

# COMMAND ----------

import mlflow
ho = spark.sql("""SELECT p.invoice_id, p.customer_id, p.actual_days_to_pay, p.days_to_pay_prediction,
                         f.payment_terms_days AS naive_prediction
                  FROM payment_predictions p JOIN invoice_payment_features f USING (invoice_id)
                  WHERE p.kind = 'hold-out'""").toPandas()
ho["err"] = (ho.days_to_pay_prediction - ho.actual_days_to_pay).abs()
ho["err_naive"] = (ho.naive_prediction - ho.actual_days_to_pay).abs()
experiment()
with mlflow.start_run(run_name="2.8 ai_predict_value pay-date hold-out") as run:
    mlflow.log_params({"function": "ai_predict_value", "target": "days_to_pay", "holdout": "20% of paid invoices from the last 12 months",
                       "features": "past_avg_days_late, past_paid_invoices, amount, payment_terms_days, invoice_month, segment, industry"})
    mlflow.log_metrics({"mae_days": round(ho.err.mean(), 2), "within_7_days_pct": round(100 * (ho.err <= 7).mean(), 1),
                        "baseline_mae_days_payment_terms": round(ho.err_naive.mean(), 2), "holdout_invoices": len(ho)})
    mlflow.log_table(ho[["invoice_id", "customer_id", "actual_days_to_pay", "days_to_pay_prediction", "err", "naive_prediction", "err_naive"]],
                     "holdout_pay_date_errors.json")
    show_run(run, "AI Function: pay-date accuracy")
print(f"MAE {ho.err.mean():.1f} days on {len(ho)} hold-out invoices ({100 * (ho.err <= 7).mean():.0f}% within a week) "
      f"vs {ho.err_naive.mean():.1f} days if you assume everyone pays on their payment terms")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.9 💬 Ask Genie — cash
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables. For the what-if in step 2.10, Genie calls the `cash_what_if` function you just created.

# COMMAND ----------

ask("What's our projected cash in week 8, bottom-up and top-down?")

# COMMAND ----------

ask("What happens to cash if DSO improves by 1 day?", before=("Steps 2.5b and 2.6 (13-week roll-up and what-if)", 55))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.11 🛠 Try it yourself
# MAGIC Set the **`dso_shift_days`** widget to `5` (customers pay 5 days later), re-run **S.2** and **step 2.6**. In which week does the bottom-up balance fall furthest below the baseline? Then ask Genie in **step 6.1**: type *"What happens to cash if DSO gets 5 days worse?"* into the `custom_question` widget.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Revenue assurance — where are we leaving revenue on the table?
# MAGIC
# MAGIC Revenue leaks between three systems that never get compared: what the **signed contract** says, what the customer **used**, and what we **billed**.
# MAGIC
# MAGIC | Step | What happens | Capability |
# MAGIC |---|---|---|
# MAGIC | 3.1 | Read every contract PDF — backfill, then daily | **Lakeflow Job**: `ai_parse_document` task → `ai_extract` task (started in step 0.8) |
# MAGIC | 3.2–3.3 | Join contract to cash; flag PDF-vs-table mismatches | SQL → gold `contract_to_cash` |
# MAGIC | 3.4 | Rule checks: under-billed usage, price below contract, missed true-ups, expired contracts in use | SQL |
# MAGIC | 3.5–3.6 | Score what passed every rule | **MLflow** Isolation Forest, registered in **Unity Catalog**, scored with `mlflow.pyfunc.spark_udf` |
# MAGIC | 3.7 | Catch under-billing that drifts over time | `ai_detect_anomalies` |
# MAGIC | 3.8 | Triage and document every case | `ai_decide`, `ai_summarize` → `leakage_exceptions` |
# MAGIC | 3.10 | How good is it? | **MLflow** `mlflow.genai.evaluate`: extraction accuracy + precision / recall |
# MAGIC | 3.11 | Tomorrow morning — only the new contracts | the same job, daily |
# MAGIC
# MAGIC > ⚠️ The job needs **Lakeflow AI Function Task** (Beta) and `ai_decide` needs the **AI Functions** previews — if these cells fail with *unknown task type* or *function not found*, ask a workspace admin to enable them on the **Previews** page.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Read the contracts — AI Function job tasks
# MAGIC Two native AI Function tasks, chained. You defined them in step 0.8 (the extraction schema is plain JSON — **no template per layout, no training**); this step shows them and waits for the backfill to finish if it's still running. The parse task's output feeds the extract task through `{{tasks.parse_contracts.output.table_name}}`.

# COMMAND ----------

show_task(parse_task, "Document Parsing", "Parses every new or changed PDF in the contracts volume — layout-aware text and tables.")
show_task(extract_task, "Information Extraction", f"Extracts {', '.join(CONTRACT_SCHEMA)} from each parsed contract, with a confidence score per field.")
backfill = wait_for_run(BACKFILL["run_id"], "one-time backfill")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ai_extract returns {value, confidence_score} per field. One clean row per contract:
# MAGIC CREATE OR REPLACE TABLE contract_terms AS
# MAGIC SELECT regexp_extract(id, '(CTR-[0-9]+)\\.pdf$', 1)                         AS contract_id,
# MAGIC        id                                                                   AS pdf_path,
# MAGIC        r:contract_number:value::STRING                                      AS contract_number,
# MAGIC        r:customer_name:value::STRING                                        AS customer_name,
# MAGIC        r:sku:value::STRING                                                  AS sku,
# MAGIC        r:contracted_price:value::DOUBLE                                     AS contracted_price,
# MAGIC        r:min_commit:value::DOUBLE                                           AS min_commit,
# MAGIC        r:true_up_terms:value::STRING                                        AS true_up_terms,
# MAGIC        try_cast(r:start_date:value::STRING AS DATE)                         AS start_date,
# MAGIC        try_cast(r:end_date:value::STRING AS DATE)                           AS end_date,
# MAGIC        r:payment_terms_days:value::INT                                      AS payment_terms_days,
# MAGIC        least(r:contracted_price:confidence_score::DOUBLE, r:min_commit:confidence_score::DOUBLE,
# MAGIC              r:end_date:confidence_score::DOUBLE)                           AS min_confidence
# MAGIC FROM (SELECT id, response:response AS r FROM contract_terms_extracted WHERE error IS NULL);
# MAGIC
# MAGIC -- The 5% of contracts where the signed PDF disagrees with the billing system
# MAGIC SELECT k.contract_id, cu.name AS customer,
# MAGIC        k.contracted_price AS table_price, t.contracted_price AS pdf_price,
# MAGIC        k.min_commit AS table_min_commit, t.min_commit AS pdf_min_commit,
# MAGIC        k.end_date AS table_end_date, t.end_date AS pdf_end_date, round(t.min_confidence, 2) AS min_confidence
# MAGIC FROM contracts k JOIN contract_terms t USING (contract_id) JOIN customers cu USING (customer_id)
# MAGIC WHERE abs(t.contracted_price - k.contracted_price) > 0.005 * k.contracted_price
# MAGIC    OR abs(t.min_commit - k.min_commit) > 0.5 OR t.end_date <> k.end_date
# MAGIC ORDER BY contract_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Join contract to cash — gold `contract_to_cash`
# MAGIC One row per invoice, with the **signed terms** (the PDF wins when it disagrees with the table), the month's **actual usage**, what we **billed** and the **list price** — plus a flag for **PDF-vs-table mismatches**. The rule checks in step 3.4 run on this table, and `leakage_usd` is what we were entitled to but didn't bill.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE contract_to_cash AS
# MAGIC WITH u AS (SELECT customer_id, sku, last_day(date) AS period, sum(quantity) AS used_qty FROM usage GROUP BY ALL),
# MAGIC j AS (
# MAGIC   SELECT i.invoice_id, i.invoice_date AS period, i.customer_id, cu.name AS customer_name, cu.segment, k.contract_id, i.sku,
# MAGIC          u.used_qty, i.billed_qty, i.billed_price, i.amount AS billed_amount, pl.list_price,
# MAGIC          k.contracted_price AS table_price, t.contracted_price AS pdf_price,
# MAGIC          k.min_commit AS table_min_commit, t.min_commit AS pdf_min_commit,
# MAGIC          k.end_date AS table_end_date, t.end_date AS pdf_end_date,
# MAGIC          coalesce(t.contracted_price, k.contracted_price) AS signed_price,
# MAGIC          coalesce(t.min_commit, k.min_commit)             AS signed_min_commit,
# MAGIC          coalesce(t.end_date, k.end_date)                 AS signed_end_date,
# MAGIC          t.true_up_terms
# MAGIC   FROM ar_invoices i
# MAGIC   JOIN customers cu USING (customer_id)
# MAGIC   JOIN contracts k USING (customer_id)
# MAGIC   JOIN price_list pl ON pl.sku = i.sku
# MAGIC   LEFT JOIN contract_terms t USING (contract_id)
# MAGIC   LEFT JOIN u ON u.customer_id = i.customer_id AND u.sku = i.sku AND u.period = i.invoice_date
# MAGIC ), f AS (
# MAGIC   SELECT *,
# MAGIC     (abs(pdf_price - table_price) > 0.005 * table_price OR abs(pdf_min_commit - table_min_commit) > 0.5
# MAGIC      OR pdf_end_date <> table_end_date)                                                  AS terms_mismatch,
# MAGIC     billed_qty < used_qty AND billed_qty > signed_min_commit                             AS usage_underbilled,
# MAGIC     used_qty > signed_min_commit AND billed_qty <= signed_min_commit                     AS missed_true_up,
# MAGIC     billed_qty < signed_min_commit AND used_qty <= signed_min_commit                     AS below_minimum,
# MAGIC     billed_price < signed_price * (1 - discount_band_pct / 100)                          AS price_below_contract,
# MAGIC     period > signed_end_date                                                             AS expired_in_use
# MAGIC   FROM j
# MAGIC )
# MAGIC SELECT *,
# MAGIC   usage_underbilled OR missed_true_up OR below_minimum OR price_below_contract OR expired_in_use AS any_rule_hit,
# MAGIC   round(greatest(
# MAGIC     greatest(used_qty, signed_min_commit) * CASE WHEN expired_in_use THEN list_price ELSE signed_price END - billed_amount, 0), 2) AS leakage_usd
# MAGIC FROM f;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'Usage billed below the quantity used' AS rule, count_if(usage_underbilled) AS invoices, count(DISTINCT CASE WHEN usage_underbilled THEN contract_id END) AS contracts,
# MAGIC        round(sum(CASE WHEN usage_underbilled THEN leakage_usd END)) AS leakage_usd FROM contract_to_cash
# MAGIC UNION ALL SELECT 'Missed true-up (usage over commit, only the commit billed)', count_if(missed_true_up), count(DISTINCT CASE WHEN missed_true_up THEN contract_id END),
# MAGIC        round(sum(CASE WHEN missed_true_up THEN leakage_usd END)) FROM contract_to_cash
# MAGIC UNION ALL SELECT 'Minimum commitment not billed', count_if(below_minimum), count(DISTINCT CASE WHEN below_minimum THEN contract_id END),
# MAGIC        round(sum(CASE WHEN below_minimum THEN leakage_usd END)) FROM contract_to_cash
# MAGIC UNION ALL SELECT concat('Price more than ', discount_band_pct, '% below the signed price'), count_if(price_below_contract), count(DISTINCT CASE WHEN price_below_contract THEN contract_id END),
# MAGIC        round(sum(CASE WHEN price_below_contract THEN leakage_usd END)) FROM contract_to_cash
# MAGIC UNION ALL SELECT 'Expired contract still in use', count_if(expired_in_use), count(DISTINCT CASE WHEN expired_in_use THEN contract_id END),
# MAGIC        round(sum(CASE WHEN expired_in_use THEN leakage_usd END)) FROM contract_to_cash
# MAGIC UNION ALL SELECT 'PDF terms differ from the billing system (flag)', count_if(terms_mismatch), count(DISTINCT CASE WHEN terms_mismatch THEN contract_id END),
# MAGIC        round(sum(CASE WHEN terms_mismatch THEN leakage_usd END)) FROM contract_to_cash;

# COMMAND ----------

# MAGIC %sql WITH q AS (SELECT 'Usage billed below the quantity used' AS rule, count_if(usage_underbilled) AS invoices, count(DISTINCT CASE WHEN usage_underbilled THEN contract_id END) AS contracts,
# MAGIC        round(sum(CASE WHEN usage_underbilled THEN leakage_usd END)) AS leakage_usd FROM contract_to_cash
# MAGIC UNION ALL SELECT 'Missed true-up (usage over commit, only the commit billed)', count_if(missed_true_up), count(DISTINCT CASE WHEN missed_true_up THEN contract_id END),
# MAGIC        round(sum(CASE WHEN missed_true_up THEN leakage_usd END)) FROM contract_to_cash
# MAGIC UNION ALL SELECT 'Minimum commitment not billed', count_if(below_minimum), count(DISTINCT CASE WHEN below_minimum THEN contract_id END),
# MAGIC        round(sum(CASE WHEN below_minimum THEN leakage_usd END)) FROM contract_to_cash
# MAGIC UNION ALL SELECT concat('Price more than ', 7.0, '% below the signed price'), count_if(price_below_contract), count(DISTINCT CASE WHEN price_below_contract THEN contract_id END),
# MAGIC        round(sum(CASE WHEN price_below_contract THEN leakage_usd END)) FROM contract_to_cash
# MAGIC UNION ALL SELECT 'Expired contract still in use', count_if(expired_in_use), count(DISTINCT CASE WHEN expired_in_use THEN contract_id END),
# MAGIC        round(sum(CASE WHEN expired_in_use THEN leakage_usd END)) FROM contract_to_cash
# MAGIC UNION ALL SELECT 'PDF terms differ from the billing system (flag)', count_if(terms_mismatch), count(DISTINCT CASE WHEN terms_mismatch THEN contract_id END),
# MAGIC        round(sum(CASE WHEN terms_mismatch THEN leakage_usd END)) FROM contract_to_cash) SELECT `rule`,SUM(`leakage_usd`) `column_9640590553` FROM q GROUP BY `rule`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5–3.6 Score the remaining invoices — an MLflow model
# MAGIC **When an AI Function, when an MLflow model?** Default to an **AI Function** when a one-step SQL answer is the point and retraining on each run is fine (`ai_forecast`, `ai_predict_value`, `ai_detect_anomalies`, `ai_top_drivers` and the text functions). Use an **MLflow model** when you need a registered, versioned model, per-row explanations, real-time serving — or when no AI Function fits the algorithm. Here that's the **Isolation Forest** in steps 3.5–3.6. MLflow also **measures how accurate the AI Function results are** — steps 1.10, 2.8 and 3.10.
# MAGIC
# MAGIC Rules only catch what someone thought to write a rule for. For the invoices that **passed every rule**, the Isolation Forest looks for rows that are unusual **across many columns at once** — no AI Function does that:
# MAGIC
# MAGIC * **3.5** builds the features as a governed **feature table** in Unity Catalog (`invoice_outlier_features`, primary key `invoice_id`) with the **Feature Engineering client** — reusable by other models, with lineage back to `contract_to_cash`. Features: price-to-contract ratio, billed-to-used ratio, discount % off list, days since the contract end, and the invoice's deviation from the customer's own norm.
# MAGIC * **3.6** trains it with scikit-learn (**autologged** to the MLflow experiment), **registers it in Unity Catalog**, points the **`@champion` alias** at the new version and scores every invoice from **`models:/…@champion`** with `mlflow.pyfunc.spark_udf` — the same code a nightly job would run, so promoting a better version never changes the scoring code.
# MAGIC * **Where to look in the experiment:** open the *3.6 Isolation Forest* run → **Artifacts** for the anomaly-score histogram (with the flag threshold) and the chart and table of which features make flagged invoices unusual; its **Metrics** include precision and recall against the planted deep discounts.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()
features_df = spark.sql("""
SELECT invoice_id, contract_id, customer_id, customer_name, period, billed_amount, list_price, signed_price, billed_price, billed_qty, used_qty,
       billed_price / signed_price                                          AS price_to_contract_ratio,
       billed_qty / greatest(used_qty, 1)                                   AS billed_to_used_ratio,
       1 - signed_price / list_price                                        AS discount_pct,
       CAST(datediff(period, signed_end_date) AS DOUBLE)                    AS days_since_contract_end,
       (billed_amount - avg(billed_amount) OVER (PARTITION BY customer_id))
         / greatest(stddev(billed_amount) OVER (PARTITION BY customer_id), 1) AS amount_vs_customer_norm
FROM contract_to_cash
WHERE NOT any_rule_hit AND NOT coalesce(terms_mismatch, false)
""")
spark.sql(f"DROP TABLE IF EXISTS {fq('invoice_outlier_features')}")          # rebuilt each run from contract_to_cash
fe.create_table(name=fq("invoice_outlier_features"), primary_keys=["invoice_id"], df=features_df,
                description="Invoice-level leakage features for every invoice that passed the SQL rules (Isolation Forest input)")
display(spark.sql(f"SELECT count(*) AS invoices_that_passed_every_rule, round(avg(discount_pct) * 100, 1) AS avg_discount_pct FROM {fq('invoice_outlier_features')}"))

# COMMAND ----------

import mlflow, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from sklearn.ensemble import IsolationForest
FEATURES = ["price_to_contract_ratio", "billed_to_used_ratio", "discount_pct", "days_since_contract_end", "amount_vs_customer_norm"]
feat_pdf = spark.table(fq("invoice_outlier_features")).select("invoice_id", "contract_id", *FEATURES).toPandas()
train = feat_pdf[FEATURES].astype("float64")
MODEL_NAME = fq("invoice_leakage_isolation_forest")
DEEP = {r.entity_id for r in spark.sql("SELECT entity_id FROM ground_truth WHERE anomaly_type = 'deep_discount'").collect()}
experiment()
mlflow.sklearn.autolog(log_models=False)          # params + training metrics automatically; the model is logged and registered explicitly below
with mlflow.start_run(run_name="3.6 Isolation Forest — invoices that passed the rules") as run:
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42).fit(train)
    scores = -iso.score_samples(train)                 # higher = more anomalous
    flagged = iso.predict(train) == -1
    threshold = -iso.offset_
    mlflow.log_params({"features": ", ".join(FEATURES), "feature_table": fq("invoice_outlier_features")})
    # Evaluation 1 — the anomaly-score distribution and where the flag threshold sits
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=60, color="#5A6F77")
    ax.axvline(threshold, color="#FF3621", linestyle="--", label=f"flag threshold ({flagged.sum()} invoices flagged)")
    ax.set_xlabel("anomaly score (higher = more unusual)"); ax.set_ylabel("invoices"); ax.legend(); ax.set_title("Isolation Forest anomaly scores")
    mlflow.log_figure(fig, "anomaly_score_histogram.png"); plt.close(fig)
    # Evaluation 2 — which features make the flagged invoices unusual
    try:
        import shap
        sv = shap.TreeExplainer(iso).shap_values(train[flagged])
        contrib = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES); method = "mean |SHAP value| on flagged invoices"
    except Exception:
        z = (train - train.mean()) / train.std().replace(0, 1)
        contrib = z[flagged].abs().mean() - z[~flagged].abs().mean(); method = "mean |z| flagged minus normal"
    contrib_df = contrib.sort_values(ascending=False).rename("contribution").reset_index().rename(columns={"index": "feature"})
    contrib_df["method"] = method
    mlflow.log_table(contrib_df, "flagged_feature_contributions.json")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(contrib_df.feature[::-1], contrib_df.contribution[::-1], color="#1B3139"); ax.set_title(f"What makes flagged invoices unusual ({method})")
    plt.tight_layout(); mlflow.log_figure(fig, "flagged_feature_contributions.png"); plt.close(fig)
    # Evaluation 3 — flagged contracts (3+ outlier invoices) vs the planted deep discounts in ground_truth
    per_contract = feat_pdf.assign(flag=flagged).groupby("contract_id").flag.sum()
    FLAGGED_CONTRACTS = set(per_contract[per_contract >= 3].index)
    hits = len(FLAGGED_CONTRACTS & DEEP)
    mlflow.log_metrics({"training_invoices": len(train), "flagged_invoices": int(flagged.sum()), "score_p99": float(np.percentile(scores, 99)),
                        "flagged_contracts": len(FLAGGED_CONTRACTS),
                        "deep_discount_precision_pct": round(100 * hits / max(len(FLAGGED_CONTRACTS), 1), 1),
                        "deep_discount_recall_pct": round(100 * hits / max(len(DEEP), 1), 1)})
    info = mlflow.sklearn.log_model(iso, name="model", input_example=train.head(5),
                                    signature=infer_signature(train, iso.predict(train)), registered_model_name=MODEL_NAME)
    show_run(run, "MLflow model: Isolation Forest")
print("Run artifacts:", [a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)])
print(f"Flagged contracts vs planted deep discounts: {hits} of {len(DEEP)} found, {len(FLAGGED_CONTRACTS)} contracts flagged")
display(contrib_df)
mlflow.sklearn.autolog(disable=True)
version = info.registered_model_version
set_alias(MODEL_NAME, "champion", version)       # scoring always reads @champion; promote a better version by moving the alias
# Score every invoice with the @champion model, exactly as a nightly job would
score = mlflow.pyfunc.spark_udf(spark, f"models:/{MODEL_NAME}@champion", result_type="double", env_manager="local")
(spark.table(fq("invoice_outlier_features"))
      .withColumn("outlier", score(*FEATURES) == -1)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq("invoice_outlier_scores")))
displayHTML(f'<div style="{FONT};font-size:14px">✓ Registered <b>{MODEL_NAME}</b> version {version}, set <b>@champion</b> and scored every invoice from <code>models:/…@champion</code>. '
            f'{button("Open the model in Unity Catalog", f"{HOST}/explore/data/models/{CATALOG}/{SCHEMA}/invoice_leakage_isolation_forest{ORG}", "#1B3139")}</div>')
display(spark.sql("""SELECT customer_name, contract_id, count(*) AS outlier_invoices, round(avg(discount_pct) * 100, 1) AS discount_pct,
                            round(avg(price_to_contract_ratio), 3) AS price_to_contract, round(sum(billed_amount)) AS billed_usd
                     FROM invoice_outlier_scores WHERE outlier GROUP BY ALL ORDER BY outlier_invoices DESC LIMIT 12"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7 Catch under-billing that drifts over time — `ai_detect_anomalies`
# MAGIC One customer's billed price slips about **1% a month** — never enough for a single invoice to break the `discount_band_pct` rule. `ai_detect_anomalies` on each customer's **monthly billed-to-contract price ratio** (`group_col => 'customer_id'`) catches the drift. (The customer over commit who was never charged is caught by the **missed true-up** rule in step 3.4.)

# COMMAND ----------

run_on_sql_warehouse(r"""
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    -- Fallback: each month vs the same customer's previous 12 months (mean ± max(3 sd, 0.5%)) — illustrative only
    CREATE OR REPLACE TABLE price_drift_anomalies AS
    WITH m AS (SELECT customer_id, period, avg(billed_price / signed_price) AS price_ratio FROM contract_to_cash GROUP BY ALL),
    ok AS (SELECT customer_id FROM m GROUP BY customer_id HAVING count(*) >= 12),
    w AS (SELECT m.*, avg(price_ratio) OVER (PARTITION BY customer_id ORDER BY period ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS mu,
                      stddev(price_ratio) OVER (PARTITION BY customer_id ORDER BY period ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS sd,
                      row_number() OVER (PARTITION BY customer_id ORDER BY period) AS n
          FROM m JOIN ok USING (customer_id))
    SELECT period, customer_id, price_ratio, mu AS expected, mu - greatest(3 * sd, 0.005) AS lower_bound, mu + greatest(3 * sd, 0.005) AS upper_bound,
           least(1.0, abs(price_ratio - mu) / greatest(4 * sd, 0.01)) AS anomaly_score,
           n > 8 AND abs(price_ratio - mu) > greatest(3 * sd, 0.005) AS is_anomaly, 'rolling baseline (fallback)' AS method
    FROM w WHERE n > 8;
  END;

  CREATE OR REPLACE TABLE price_drift_anomalies AS
  SELECT a.*, 'ai_detect_anomalies' AS method
  FROM ai_detect_anomalies(
    TABLE(
      WITH m AS (SELECT customer_id, period, avg(billed_price / signed_price) AS price_ratio FROM contract_to_cash GROUP BY ALL)
      SELECT * FROM m WHERE customer_id IN (SELECT customer_id FROM m GROUP BY customer_id HAVING count(*) >= 12)
    ),
    time_col          => 'period',
    value_col         => 'price_ratio',
    group_col         => 'customer_id',
    anomaly_threshold => anomaly_threshold
  ) a;
END;
""", anomaly_threshold=("DOUBLE", num("anomaly_threshold", 0.95)))
display(spark.sql("""SELECT c.name AS customer, a.period, round(a.price_ratio, 4) AS billed_to_contract_ratio, round(a.expected, 4) AS expected, a.method
                     FROM price_drift_anomalies a JOIN customers c USING (customer_id)
                     WHERE a.is_anomaly AND a.price_ratio < a.expected ORDER BY customer, period"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.8 Triage and document — `ai_decide` and `ai_summarize`
# MAGIC Rule hits, Isolation Forest outliers and drift anomalies become **one exception list**, one row per contract and problem, with the **$ at risk**. Then two AI Functions do the analyst's first pass *(as SQL calls in the notebook — they don't have dedicated job task types)*:
# MAGIC
# MAGIC * **`ai_decide`** asks, for every exception, *"billing error or valid discount?"* — based on the **signed terms extracted from the PDF** (signed price, minimum commitment, true-up clause, end date), with the billing-system price shown alongside for comparison.
# MAGIC * **`ai_summarize`** writes a short **case note** for the revenue-operations queue.
# MAGIC
# MAGIC Everything lands in **`leakage_exceptions`**.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE leakage_exceptions AS
# MAGIC WITH rules AS (
# MAGIC   SELECT contract_id, customer_id, customer_name, exception_type, count(*) AS invoices, min(period) AS first_period, max(period) AS last_period,
# MAGIC          sum(CASE WHEN exception_type = 'terms_mismatch' THEN 0 ELSE leakage_usd END) AS amount_at_risk_usd,   -- its $ are counted by the price / commit / end-date rules
# MAGIC          'SQL rule' AS found_by
# MAGIC   FROM contract_to_cash
# MAGIC   LATERAL VIEW explode(filter(array(
# MAGIC     CASE WHEN usage_underbilled THEN 'usage_underbilled' END, CASE WHEN missed_true_up THEN 'missed_true_up' END,
# MAGIC     CASE WHEN below_minimum THEN 'below_minimum' END, CASE WHEN price_below_contract THEN 'price_below_contract' END,
# MAGIC     CASE WHEN expired_in_use THEN 'expired_in_use' END, CASE WHEN terms_mismatch THEN 'terms_mismatch' END), x -> x IS NOT NULL)) e AS exception_type
# MAGIC   GROUP BY ALL
# MAGIC ), outliers AS (   -- Isolation Forest: contracts with 3+ outlier invoices; $ at risk = discount beyond the 25% policy maximum
# MAGIC   SELECT contract_id, customer_id, customer_name, 'statistical_outlier' AS exception_type, count(*) AS invoices, min(period), max(period),
# MAGIC          sum(greatest(list_price * 0.75 - billed_price, 0) * billed_qty) AS amount_at_risk_usd, 'MLflow Isolation Forest' AS found_by
# MAGIC   FROM invoice_outlier_scores WHERE outlier GROUP BY ALL HAVING count(*) >= 3
# MAGIC ), first_drift AS (   -- first month each customer's billed-to-contract ratio drifted down
# MAGIC   SELECT customer_id, min(period) AS first_anomaly FROM price_drift_anomalies WHERE is_anomaly AND price_ratio < expected GROUP BY ALL
# MAGIC ), drift AS (
# MAGIC   SELECT c.contract_id, c.customer_id, c.customer_name, 'price_drift' AS exception_type, count(*) AS invoices, min(c.period), max(c.period),
# MAGIC          sum((c.signed_price - c.billed_price) * c.billed_qty) AS amount_at_risk_usd, 'ai_detect_anomalies' AS found_by
# MAGIC   FROM contract_to_cash c JOIN first_drift f ON f.customer_id = c.customer_id AND c.period >= f.first_anomaly
# MAGIC   WHERE c.billed_price < c.signed_price
# MAGIC   GROUP BY ALL
# MAGIC ), x AS (SELECT * FROM rules UNION ALL SELECT * FROM outliers UNION ALL SELECT * FROM drift),
# MAGIC stats AS (
# MAGIC   SELECT x.contract_id, x.exception_type, round(avg(c.billed_price), 4) AS avg_billed_price,
# MAGIC          round(avg(c.used_qty)) AS avg_used_qty, round(avg(c.billed_qty)) AS avg_billed_qty
# MAGIC   FROM x JOIN contract_to_cash c ON c.contract_id = x.contract_id AND c.period BETWEEN x.first_period AND x.last_period
# MAGIC   GROUP BY ALL
# MAGIC ), facts AS (
# MAGIC   SELECT x.*, t.contracted_price AS signed_price, t.min_commit AS signed_min_commit, t.end_date AS signed_end_date, t.true_up_terms,
# MAGIC          k.contracted_price AS billing_system_price, pl.list_price, s.avg_billed_price, s.avg_used_qty, s.avg_billed_qty
# MAGIC   FROM x JOIN contracts k USING (contract_id) JOIN price_list pl ON pl.sku = k.sku LEFT JOIN contract_terms t USING (contract_id)
# MAGIC   LEFT JOIN stats s ON s.contract_id = x.contract_id AND s.exception_type = x.exception_type
# MAGIC ), d AS (
# MAGIC   SELECT *, to_json(named_struct('exception_type', exception_type, 'found_by', found_by, 'invoices', invoices,
# MAGIC                                  'first_period', first_period, 'last_period', last_period, 'amount_at_risk_usd', round(amount_at_risk_usd),
# MAGIC                                  'signed_contract', named_struct('unit_price', signed_price, 'min_monthly_commit', signed_min_commit,
# MAGIC                                                                  'end_date', signed_end_date, 'true_up_clause', true_up_terms),
# MAGIC                                  'billing_system_price', billing_system_price, 'list_price', list_price,
# MAGIC                                  'avg_billed_price', avg_billed_price, 'avg_used_qty', avg_used_qty, 'avg_billed_qty', avg_billed_qty)) AS case_facts
# MAGIC   FROM facts
# MAGIC )
# MAGIC SELECT md5(concat(contract_id, exception_type)) AS exception_id, contract_id, customer_id, customer_name, exception_type, found_by,
# MAGIC        invoices, first_period, last_period, round(amount_at_risk_usd, 2) AS amount_at_risk_usd,
# MAGIC        decision:response:answers:verdict:choice::STRING            AS verdict,
# MAGIC        decision:response:answers:verdict:confidence::DOUBLE        AS verdict_confidence,
# MAGIC        ai_summarize(concat('Revenue assurance case for ', customer_name, ' (contract ', contract_id, '). ', case_facts,
# MAGIC                            ' Verdict: ', decision:response:answers:verdict:choice::STRING,
# MAGIC                            CASE WHEN decision:response:answers:verdict:choice::STRING = 'valid_discount'
# MAGIC                                 THEN '. The billing matches the signed contract: describe the amount as discount beyond the 25% policy to review, not as lost revenue.'
# MAGIC                                 ELSE '. Describe the amount as revenue at risk to recover.' END), 45) AS case_note,
# MAGIC        case_facts
# MAGIC FROM (
# MAGIC   SELECT *, ai_decide(case_facts,
# MAGIC     '{"verdict": {"type": "choice", "instructions": "Given the SIGNED contract terms and what was billed, is this a billing error that lost revenue, or a valid discount / price the customer is entitled to?",
# MAGIC        "criteria": {"billing_error": "Billing differs from the signed contract terms (price, minimum commitment, true-up or end date) and the company billed less than it was entitled to",
# MAGIC                     "valid_discount": "The billed price and quantity match what the signed contract allows; any discount is written into the signed contract"}}}',
# MAGIC     map('version', '1.0')) AS decision
# MAGIC   FROM d
# MAGIC );
# MAGIC
# MAGIC SELECT exception_type, found_by, count(*) AS cases, round(sum(amount_at_risk_usd)) AS amount_at_risk_usd,
# MAGIC        count_if(verdict = 'billing_error') AS billing_errors, count_if(verdict = 'valid_discount') AS valid_discounts
# MAGIC FROM leakage_exceptions GROUP BY ALL ORDER BY amount_at_risk_usd DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Leakage $ at risk by customer (billing errors only) — chart this as a bar: x = customer, y = amount_at_risk_usd, color = exception_type
# MAGIC SELECT customer_name AS customer, exception_type, round(sum(amount_at_risk_usd)) AS amount_at_risk_usd
# MAGIC FROM leakage_exceptions
# MAGIC WHERE verdict = 'billing_error' AND customer_name IN (
# MAGIC   SELECT customer_name FROM leakage_exceptions WHERE verdict = 'billing_error' GROUP BY ALL ORDER BY sum(amount_at_risk_usd) DESC LIMIT 12)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY amount_at_risk_usd DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Leakage $ at risk by customer (billing errors only) — chart this as a bar: x = customer, y = amount_at_risk_usd, color = exception_type
# MAGIC SELECT customer_name AS customer, exception_type, round(sum(amount_at_risk_usd)) AS amount_at_risk_usd
# MAGIC FROM leakage_exceptions
# MAGIC WHERE verdict = 'billing_error' AND customer_name IN (
# MAGIC   SELECT customer_name FROM leakage_exceptions WHERE verdict = 'billing_error' GROUP BY ALL ORDER BY sum(amount_at_risk_usd) DESC LIMIT 12)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY amount_at_risk_usd DESC) SELECT `customer`,SUM(`amount_at_risk_usd`) `column_5411685597`,`exception_type` FROM q GROUP BY `customer`,`exception_type`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT customer_name, contract_id, exception_type, found_by, round(amount_at_risk_usd) AS amount_at_risk_usd, verdict,
# MAGIC        round(verdict_confidence, 2) AS confidence, case_note
# MAGIC FROM leakage_exceptions ORDER BY amount_at_risk_usd DESC LIMIT 25;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.10 Check accuracy in MLflow — extraction and exception list
# MAGIC Two questions, one experiment:
# MAGIC
# MAGIC 1. **Did `ai_extract` read the contracts correctly?** The true contract terms from `ground_truth` are registered as a versioned **MLflow evaluation dataset in Unity Catalog** (`contract_extraction_eval` — it shows in the experiment's *Datasets* tab). `mlflow.genai.evaluate` runs a **traced `predict_fn`** that returns what the `extract_contract_terms` task wrote for each contract, and **custom scorers** check it field by field — every contract becomes a trace you can open.
# MAGIC 2. **Did the exception list find the planted leakage — and only that?** **Precision and recall at the contract level** against every seeded leakage case, plus **recall per problem type**.

# COMMAND ----------

import mlflow, math, pandas as pd
from mlflow.genai.scorers import scorer
FIELDS = ["customer_name", "sku", "contracted_price", "min_commit", "start_date", "end_date", "true_up_terms"]
# 1) The answer key as a versioned evaluation dataset in Unity Catalog
truth = spark.sql("SELECT entity_id AS contract_id, field, true_value FROM ground_truth WHERE anomaly_type = 'contract_terms'").toPandas() \
             .pivot(index="contract_id", columns="field", values="true_value")
extraction_ds = eval_dataset("contract_extraction_eval",
                             [{"inputs": {"contract_id": cid}, "expectations": {k: t[k] for k in FIELDS}} for cid, t in truth.iterrows()])
# 2) What the ai_extract task produced, served by a traced predict_fn (one trace per contract)
clean = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else v
EXTRACTED = {r.contract_id: {"customer_name": r.customer_name, "sku": r.sku, "contracted_price": clean(r.contracted_price),
                             "min_commit": clean(r.min_commit), "start_date": str(r.start_date), "end_date": str(r.end_date),
                             "true_up_terms": r.true_up_terms} for r in spark.table("contract_terms").toPandas().itertuples()}
@mlflow.trace(span_type="PARSER")
def extracted_terms(contract_id):
    """Returns the fields the extract_contract_terms AI Function task wrote for this contract."""
    return EXTRACTED.get(contract_id, {k: None for k in FIELDS})
num_ok = lambda a, b, tol: a is not None and b is not None and abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1e-9)
@scorer
def price_correct(outputs, expectations): return num_ok(outputs["contracted_price"], expectations["contracted_price"], 0.001)
@scorer
def min_commit_correct(outputs, expectations): return num_ok(outputs["min_commit"], expectations["min_commit"], 0.001)
@scorer
def end_date_correct(outputs, expectations): return outputs["end_date"] == expectations["end_date"]
@scorer
def start_date_correct(outputs, expectations): return outputs["start_date"] == expectations["start_date"]
@scorer
def sku_correct(outputs, expectations): return outputs["sku"] == expectations["sku"]
@scorer
def customer_correct(outputs, expectations):
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    return norm(outputs["customer_name"]) == norm(expectations["customer_name"])
@scorer
def true_up_captured(outputs, expectations):
    s = str(outputs.get("true_up_terms") or "").lower()
    return ("commit" in s or "minimum" in s) and ("price" in s or "overage" in s or "arrears" in s)

experiment()
with mlflow.start_run(run_name="3.10 ai_extract contract accuracy") as run:
    res = mlflow.genai.evaluate(data=extraction_ds, predict_fn=extracted_terms,
                                scorers=[price_correct, min_commit_correct, end_date_correct, start_date_correct, sku_correct, customer_correct, true_up_captured])
    show_run(run, "AI Function: extraction accuracy")
acc = pd.DataFrame([(k.split("/")[0], round(100 * v, 1)) for k, v in res.metrics.items() if k.endswith("/mean")], columns=["field", "accuracy_pct"])

# Precision / recall of the exception list against every planted leakage case (contract level)
TYPE_MAP = {"underbilled_invoice": "usage_underbilled", "missed_true_up": "missed_true_up", "price_slip": "price_drift",
            "pdf_table_mismatch": "terms_mismatch", "expired_in_use": "expired_in_use", "deep_discount": "statistical_outlier"}
seeded = spark.sql("""SELECT CASE WHEN g.entity_type = 'invoice' THEN k.contract_id ELSE g.entity_id END AS contract_id, g.anomaly_type
                      FROM ground_truth g LEFT JOIN ar_invoices i ON g.entity_type = 'invoice' AND i.invoice_id = g.entity_id
                      LEFT JOIN contracts k ON k.customer_id = i.customer_id
                      WHERE g.use_case = 'Revenue assurance'""").toPandas().drop_duplicates()
found = spark.sql("SELECT DISTINCT contract_id, exception_type FROM leakage_exceptions").toPandas()
S, F = set(seeded.contract_id), set(found.contract_id)
precision, recall = len(S & F) / max(len(F), 1), len(S & F) / max(len(S), 1)
per_type = [(t, round(100 * len(set(seeded[seeded.anomaly_type == t].contract_id) & set(found[found.exception_type == e].contract_id))
                     / max(len(set(seeded[seeded.anomaly_type == t].contract_id)), 1), 1)) for t, e in TYPE_MAP.items()]
with mlflow.start_run(run_name="3.10 leakage exception list vs ground truth") as run:
    mlflow.log_metrics({"precision_pct": round(100 * precision, 1), "recall_pct": round(100 * recall, 1),
                        "seeded_contracts": len(S), "flagged_contracts": len(F), **{f"recall_pct_{t}": r for t, r in per_type}})
    show_run(run, "Exception list: precision & recall")
display(acc)
display(pd.DataFrame(per_type, columns=["planted problem", "recall_pct"]))
print(f"Contract-level precision {100 * precision:.0f}% · recall {100 * recall:.0f}% ({len(S & F)} of {len(S)} seeded contracts found, {len(F)} flagged)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.11 Tomorrow morning — the same job, only the new contracts
# MAGIC Four new order forms are signed today and land in the `contracts` volume (one of them was keyed into billing at the wrong price). The **same job** runs again: because AI Function tasks are incremental, it parses and extracts **only the 4 new PDFs** — not the 200 it already read. In production this is the job's **daily 6:00 AM schedule** (created paused in step 0.8 — unpause it on the job page).

# COMMAND ----------

import datetime as dt
before = spark.table("contract_terms_extracted").count()
new_terms = []
for k in range(4):
    cid = f"C{1201 + k}"
    sku = ["PLAT-UNIT", "SEAT-PRO", "API-1K", "STORAGE-TB"][k]
    price = round(SKUS[sku][1] * (0.88 - 0.02 * k), 4 if sku == "API-1K" else 2)
    t = dict(contract_id=f"CTR-{20301 + k}", customer_id=cid, sku=sku, contracted_price=price, min_commit=float([4000, 120, 60000, 400][k]),
             start_date=dt.date.today(), end_date=add_months(dt.date.today(), 24), true_up_terms=TRUEUP_TEXTS[k % 3], layout=k % 3)
    new_terms.append(t)
new_customers = pd.DataFrame([dict(customer_id=f"C{1201 + k}", name=n, segment=s, industry=i, payment_terms_days=d) for k, (n, s, i, d) in enumerate(
    [("Brightwater Health Partners", "Enterprise", "Healthcare", 45), ("Ironclad Fabrication Co", "Mid-Market", "Manufacturing", 30),
     ("Larkspur Capital Advisors", "Mid-Market", "Financial Services", 30), ("Verdant BioLabs", "SMB", "Life Sciences", 30)])])
already_loaded = spark.sql(f"SELECT count(*) FROM {fq('contracts')} WHERE contract_id >= 'CTR-20301'").first()[0] > 0   # re-running is safe
if not already_loaded:
    spark.createDataFrame(new_customers).write.mode("append").saveAsTable(fq("customers"))
keyed = [dict(t, contracted_price=round(t["contracted_price"] * 0.9, 2)) if k == 1 else dict(t) for k, t in enumerate(new_terms)]  # one keyed 10% low
if not already_loaded:
    spark.createDataFrame(pd.DataFrame(keyed).drop(columns=["true_up_terms", "layout"])).write.mode("append").saveAsTable(fq("contracts"))
write_contract_pdfs(new_terms, pd.concat([customers, new_customers]), SKUS, CONTRACTS, SIGNERS)
daily = run_job(JOB_ID, "daily run")
after = spark.table("contract_terms_extracted").count()
displayHTML(f'<div style="{FONT};font-size:14px">Contracts extracted: <b>{before}</b> after the backfill → <b>{after}</b> after the daily run — '
            f'the daily run read only the <b>{after - before}</b> new PDFs.</div>')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Same flatten as step 3.2 for today's contracts, compared with what was keyed into billing
# MAGIC SELECT k.contract_id, cu.name AS customer, k.sku, k.contracted_price AS keyed_price,
# MAGIC        e.response:response:contracted_price:value::DOUBLE AS signed_price,
# MAGIC        e.response:response:min_commit:value::DOUBLE AS signed_min_commit,
# MAGIC        e.response:response:end_date:value::STRING AS signed_end_date,
# MAGIC        CASE WHEN abs(e.response:response:contracted_price:value::DOUBLE - k.contracted_price) > 0.005 * k.contracted_price
# MAGIC             THEN 'MISMATCH — fix before the first invoice' ELSE 'matches' END AS check_result
# MAGIC FROM contract_terms_extracted e
# MAGIC JOIN contracts k ON k.contract_id = regexp_extract(e.id, '(CTR-[0-9]+)\\.pdf$', 1)
# MAGIC JOIN customers cu USING (customer_id)
# MAGIC WHERE k.contract_id >= 'CTR-20301'
# MAGIC ORDER BY k.contract_id;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.13 💬 Ask Genie — revenue leakage
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("How much revenue leakage did we find, and how much more sits in discounts beyond policy?", before=("Section 3 (contract-to-cash rules and triage)", 110))

# COMMAND ----------

ask("Which contracts don't match what we're billing?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.15 🛠 Try it yourself
# MAGIC Set **`discount_band_pct`** to `3`, re-run **S.2**, then **steps 3.3–3.4**. Which customer now trips the *price below contract* rule that only `ai_detect_anomalies` caught before? Then, in **step 0.8**, add `"renewal_notice_days": {"type": "integer", "description": "Days of notice required to cancel or not renew"}` to `CONTRACT_SCHEMA` and re-run step 0.8's job definition — no retraining. *The task is incremental, so the next run adds the field for new or changed contracts only; to re-extract all 200, give the extract task's output table a new name and point step 3.2 at it.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wrap-up — put it in front of the business

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 The Finance Genie agent
# MAGIC The agent you've been asking since step 0.11 now covers every gold table and the `cash_what_if` function:
# MAGIC
# MAGIC **Tables:** `actuals_vs_plan`, `variance_drivers`, `variance_commentary`, `forecast_next_qtr`, `cash_forecast_13wk`, `expected_receipts`, `customers`, `leakage_exceptions`, `contract_to_cash` (+ `opex_anomalies`, `working_capital`, `late_payer_anomalies`, `collections_outreach`) · **SQL function:** `cash_what_if(dso_shift_days)`
# MAGIC
# MAGIC The AI Function results were written to tables **once**; Genie reads those tables and never re-runs a forecast or an LLM call per question.

# COMMAND ----------

ask("Which customers have the biggest billing gaps, and what should we do about each?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 The CFO dashboard — one page
# MAGIC Step 4.2 builds (or updates) a one-page **AI/BI dashboard** on the gold tables, with the Genie agent attached (*Ask Genie* button):
# MAGIC * **Variance by cost center**, with the drivers
# MAGIC * **13-week cash forecast**, bottom-up vs top-down, with the interval
# MAGIC * **Leakage $ at risk by customer**

# COMMAND ----------

DATASETS = {
    "variance": f"""SELECT cost_center, round(sum(variance)) AS variance_usd, round((sum(actual_amount) / sum(plan_amount) - 1) * 100, 1) AS variance_pct
                    FROM {fq('actuals_vs_plan')} WHERE line_type = 'Opex'
                      AND period BETWEEN date_trunc('QUARTER', add_months(current_date(), -3)) AND date_sub(date_trunc('QUARTER', current_date()), 1)
                    GROUP BY cost_center""",
    "drivers": f"SELECT comparison, contributor, change_usd, change_pct FROM {fq('variance_drivers')} WHERE contributor <> 'all' ORDER BY abs(change_usd) DESC LIMIT 12",
    "cash": f"""SELECT week_end, 'Bottom-up' AS method, bottom_up_balance AS balance_usd FROM {fq('cash_forecast_13wk')}
                UNION ALL SELECT week_end, 'Top-down (ai_forecast)', top_down_balance FROM {fq('cash_forecast_13wk')}
                UNION ALL SELECT week_end, 'Top-down low', top_down_balance_low FROM {fq('cash_forecast_13wk')}
                UNION ALL SELECT week_end, 'Top-down high', top_down_balance_high FROM {fq('cash_forecast_13wk')}""",
    "leakage": f"""SELECT customer_name, exception_type, round(sum(amount_at_risk_usd)) AS amount_at_risk_usd FROM {fq('leakage_exceptions')}
                   WHERE verdict = 'billing_error' AND amount_at_risk_usd > 0 AND customer_name IN (
                     SELECT customer_name FROM {fq('leakage_exceptions')} WHERE verdict = 'billing_error'
                     GROUP BY customer_name ORDER BY sum(amount_at_risk_usd) DESC LIMIT 10)
                   GROUP BY ALL""",
    "kpis": f"""SELECT (SELECT round(sum(variance)) FROM {fq('actuals_vs_plan')} WHERE line_type = 'Opex'
                          AND period BETWEEN date_trunc('QUARTER', add_months(current_date(), -3)) AND date_sub(date_trunc('QUARTER', current_date()), 1)) AS opex_over_plan_usd,
                       (SELECT bottom_up_balance FROM {fq('cash_forecast_13wk')} WHERE week_number = 13) AS cash_week_13_usd,
                       (SELECT dso_days FROM {fq('working_capital')}) AS dso_days,
                       (SELECT round(sum(amount_at_risk_usd)) FROM {fq('leakage_exceptions')} WHERE verdict = 'billing_error') AS leakage_usd""",
}
WIDGETS = [
    text_widget("# Office of the CFO — plan, cash & leakage\nOpex over plan last quarter, the next 13 weeks of cash two ways, and the revenue we're leaving on the table. "
                "Built on Databricks AI Functions; every number traces back to its source in Unity Catalog. Use **Ask Genie** for anything else.", 0, 0, 12, 2),
    counter_widget("kpis", "opex_over_plan_usd", "Opex over plan, last quarter ($)", 0, 2, 3),
    counter_widget("kpis", "cash_week_13_usd", "Cash in week 13, bottom-up ($)", 3, 2, 3),
    counter_widget("kpis", "dso_days", "DSO (days)", 6, 2, 3),
    counter_widget("kpis", "leakage_usd", "Leakage at risk — billing errors ($)", 9, 2, 3),
    bar_widget("variance", "cost_center", "variance_usd", None, "Opex variance vs plan by cost center, last quarter ($)", 0, 5, 6, 6),
    table_widget("drivers", ["contributor", "change_usd", "comparison"], "What drove it — ai_top_drivers", 6, 5, 6, 6),
    line_widget("cash", "week_end", "balance_usd", "method", "13-week cash: bottom-up vs top-down (ai_forecast, 80% interval)", 0, 11, 7, 7),
    bar_widget("leakage", "customer_name", "amount_at_risk_usd", "exception_type", "Leakage $ at risk — top 10 customers (billing errors)", 7, 11, 5, 7, horizontal=True),
]
DASHBOARD_URL = publish_dashboard(DASHBOARD_NAME, DATASETS, WIDGETS)
displayHTML(button("Open the CFO dashboard", DASHBOARD_URL) + button("Open the Genie agent", GENIE_URL, "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Lineage — from the contract PDF to the leakage dollar
# MAGIC Unity Catalog records **every** table-to-table hop automatically. Open the cards below → **Lineage**. For example: *contract PDF (volume) → `parsed_contracts` (`ai_parse_document`) → `contract_terms_extracted` (`ai_extract`) → `contract_terms` → `contract_to_cash` → `leakage_exceptions` ($)*. Step 4.3b reads the same lineage from the `system.access.table_lineage` system table.

# COMMAND ----------

show_uc_cards(roles={"AI Function task", "Silver", "Gold", "Feature table", "ai_top_drivers", "ai_detect_anomalies", "ai_forecast", "ai_query", "ai_predict_value",
                     "ai_gen", "MLflow model", "ai_decide · ai_summarize"}, volume="contracts", title="Silver, gold and AI Function tables")

# COMMAND ----------

try:
    display(spark.sql(f"""
    SELECT DISTINCT coalesce(source_table_full_name, source_path) AS source, target_table_full_name AS target
    FROM system.access.table_lineage
    WHERE target_table_catalog = '{CATALOG}' AND target_table_schema = '{SCHEMA}'
      AND coalesce(source_table_full_name, source_path) NOT LIKE '__databricks%'
      AND target_table_name IN ('parsed_contracts', 'contract_terms_extracted', 'contract_terms', 'contract_to_cash', 'leakage_exceptions')
      AND event_date >= current_date() - INTERVAL 2 DAYS
    ORDER BY target"""))
except Exception as e:
    print("system.access.table_lineage isn't readable here (ask an admin to grant access to the system schema) — use the Lineage tab on the cards in step 4.3.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 The MLflow experiment — the trust layer
# MAGIC One experiment — with its traces stored in **Unity Catalog** — holds the proof behind every number in the dashboard: the **forecast backtest** (MAPE, interval coverage, vs a challenger), the **pay-date MAE**, **extraction accuracy** field by field against the `contract_extraction_eval` dataset, and the **Isolation Forest** — registered in Unity Catalog with its **`@champion`** alias. Open the experiment's **Datasets** tab to see the evaluation dataset, and **Traces** to open any single scored contract.

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=[experiment().experiment_id], order_by=["start_time DESC"])
cols = [c for c in runs.columns if c.startswith("metrics.") and any(k in c for k in
        ["mape_pct", "interval_coverage_pct", "mae_days", "within_7_days", "precision_pct", "recall_pct", "/mean", "flagged_invoices"])
        and not any(s in c for s in ["mape_pct_", "coverage_pct_", "recall_pct_"])]
display(runs[["tags.mlflow.runName"] + cols].rename(columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.runName", "run")))
displayHTML(button("Open the MLflow experiment", f"{HOST}/ml/experiments/{experiment().experiment_id}{ORG}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Executive summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'FP&A' AS area, 'Opex vs plan, last quarter (USD)' AS metric,
# MAGIC        format_number(sum(variance), 0) AS value FROM actuals_vs_plan WHERE line_type = 'Opex' AND period BETWEEN lfq_start AND lfq_end
# MAGIC UNION ALL SELECT 'FP&A', 'Revenue vs plan, last quarter (USD)', format_number(sum(variance), 0)
# MAGIC   FROM actuals_vs_plan WHERE line_type = 'Revenue' AND period BETWEEN lfq_start AND lfq_end
# MAGIC UNION ALL SELECT 'FP&A', 'Top driver (ai_top_drivers)', replace(replace(max_by(contributor, abs(change_usd)), 'cost_center=', 'cost center '), '_', ' ') FROM variance_drivers WHERE contributor <> 'all'
# MAGIC UNION ALL SELECT 'FP&A', 'Revenue forecast, next quarter (USD)', format_number(sum(amount_forecast), 0) FROM forecast_next_qtr WHERE cost_center = 'Revenue'
# MAGIC UNION ALL SELECT 'Cash', 'DSO (days)', CAST(dso_days AS STRING) FROM working_capital
# MAGIC UNION ALL SELECT 'Cash', 'Cash in week 8 — bottom-up / top-down (USD)', concat(format_number(bottom_up_balance, 0), ' / ', format_number(top_down_balance, 0))
# MAGIC   FROM cash_forecast_13wk WHERE week_number = 8
# MAGIC UNION ALL SELECT 'Cash', 'Customers paying later and later', CAST(count_if(paying_later_and_later) AS STRING) FROM collections_outreach
# MAGIC UNION ALL SELECT 'Revenue assurance', 'Leakage cases found', CAST(count(*) AS STRING) FROM leakage_exceptions
# MAGIC UNION ALL SELECT 'Revenue assurance', 'Revenue at risk — billing errors (USD)', format_number(sum(CASE WHEN verdict = 'billing_error' THEN amount_at_risk_usd END), 0) FROM leakage_exceptions
# MAGIC UNION ALL SELECT 'Revenue assurance', 'Contracts where the PDF disagrees with billing', CAST(count(DISTINCT CASE WHEN terms_mismatch THEN contract_id END) AS STRING) FROM contract_to_cash;

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
# MAGIC | **Many-model challenger** for the forecast | Log a Prophet model per cost center × account (the many-model forecasting accelerator) in the same experiment as step 1.10, and compare MAPE | 1 hr |
# MAGIC | **Close the books faster** — explain every GL line over threshold | `ai_top_drivers` per account with `is_test` = this month vs last, `ai_query` for the flux note | 30 min |
# MAGIC | **More contract fields** — renewal notice, price escalators, auto-renew | Add fields to `CONTRACT_SCHEMA` (step 0.8) and re-run the job — no retraining | 15 min |
# MAGIC | **Invoices and POs from vendors** | A second AI Function task pipeline on `ap@` attachments: `ai_parse_document` → `ai_classify` → `ai_extract` | 1 day |
# MAGIC | **Score leakage in real time** | Serve the Isolation Forest from step 3.6 with **Model Serving** and call it from billing before invoices go out | 1 hr |
# MAGIC | **Unpause the daily job** | Job page → Schedule → *Unpause*. New contracts are read every morning at 6:00 | 1 min |
# MAGIC | **Share Genie** with the business | Open the Genie agent → **Share** → add your controllers, treasury and rev-ops groups | 2 min |
# MAGIC | **Your real data** | Point the views in *Bring your own data* (step 0.12) at your ERP, bank and CLM data | 1–2 days |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. What you just saw — and where to take it next
# MAGIC
# MAGIC * **Three CFO questions, one governed foundation.** Ledger, plan, headcount, billing, bank, AP and 200 signed PDFs — all in Unity Catalog, with lineage from every source to every number.
# MAGIC * **AI Functions did the heavy lifting in SQL.** `ai_top_drivers`, `ai_detect_anomalies`, `ai_forecast`, `ai_predict_value`, `ai_query`, `ai_gen`, `ai_decide` and `ai_summarize` — no ML pipelines to build and no model endpoints to manage. The contracts were read by **native AI Function tasks** in a Lakeflow Job: backfill once, then only what's new.
# MAGIC * **MLflow shows you can trust the numbers.** Backtests, hold-outs, field-level extraction accuracy and precision / recall against an answer key — and an Isolation Forest, registered in Unity Catalog, where no AI Function fit.
# MAGIC * **Genie and a dashboard put it in front of the business.** Controllers, treasury and revenue operations ask in plain English; Genie reads the precomputed tables and calls the same `cash_what_if` function you did.
# MAGIC
# MAGIC **Production path**
# MAGIC 1. **Land real data** — ERP, planning, bank and CLM feeds via Lakeflow Connect / partner connectors into Unity Catalog; swap the views in *Bring your own data* (step 0.12).
# MAGIC 2. **Schedule it** — unpause the contract job, and run this notebook (or its SQL steps) as a Lakeflow Job after each close and each morning for cash.
# MAGIC 3. **Share the Genie agent and the dashboard** with finance.
# MAGIC 4. **Graduate where needed** — serve the Isolation Forest, or promote a forecast challenger, with Model Serving and the MLflow registry.
# MAGIC
# MAGIC **Docs:** [ai_forecast](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_forecast) · [ai_detect_anomalies](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_detect_anomalies) · [ai_top_drivers](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_top_drivers) · [ai_predict_value](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_predict_value) · [ai_extract](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_extract) · [ai_decide](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_decide) · [AI Function tasks](https://docs.databricks.com/aws/en/jobs/ai-function-task) · [MLflow GenAI evaluation](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) · [Genie](https://docs.databricks.com/aws/en/genie/)
