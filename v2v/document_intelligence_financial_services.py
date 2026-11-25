# Databricks notebook source

# MAGIC %md
# MAGIC # Loan, Investment & Claims Document Intelligence for Financial Services
# MAGIC **For banks, asset managers and insurers:** build a **multi-task Lakeflow Job** of native **AI Function tasks** that parses incoming document packets, classifies and splits every page and extracts loan / deal / claim fields — as a backfill once and then daily — and then reconcile the results against your systems of record, flag the mismatches that matter, and let anyone ask Genie. Powered by Databricks AI Functions.
# MAGIC
# MAGIC **Industry:** Financial Services — **Lending** (mortgage files), **Investments** (fund subscriptions & capital calls) and **Insurance** (auto claims)
# MAGIC
# MAGIC Every operations team in financial services has the same inbox problem: *a 10-page PDF arrives — a loan file, a capital call package, a claim packet — and someone has to figure out what's in it, re-key the numbers, compare them to the system and decide whether it's safe to fund, wire or pay.*
# MAGIC This notebook builds the **document intelligence workflow** — `ai_parse_document` → `ai_classify` → `ai_extract` — as a **Lakeflow Job DAG** of **native AI Function tasks** (*Document Parsing*, *Classification*, *Information Extraction*), glued together by small **SQL tasks** — each AI task manages its own output table and processes only new inputs, with retries, repair runs, per-task row counts and a schedule — with **no OCR vendor, no model training, no endpoints to manage**. Then it uses the workflow's output: reconciliation against your systems of record, `ai_decide` for the judgment calls, and **Genie** for the business.
# MAGIC
# MAGIC | Step | Question | Databricks capability |
# MAGIC |---|---|---|
# MAGIC | 1 | How do parse → classify → extract become a **repeatable, orchestrated workflow**? | **Lakeflow Jobs** — a multi-task **DAG** of native **AI Function tasks** + SQL tasks, classification and extraction **fanned out** in parallel |
# MAGIC | 2 | How do we run it over the **archive** (backfill)? | One Job run — every AI Function task processes only what's new |
# MAGIC | 3 | What's actually in each PDF — text, tables and layout, page by page? | `ai_parse_document` task (`parse_documents`) |
# MAGIC | 4 | What **type of document** is each page, and where does one document end and the next begin (**split the packet**)? | `ai_classify` tasks — line of business, then document type — + a SQL task (`split_documents`) |
# MAGIC | 5 | What are the **loan, deal and claim fields** on each document — and how confident are we? | `ai_extract` tasks (`extract_lending` · `extract_investments` · `extract_claims`) |
# MAGIC | 6 | What happens **tomorrow morning** when new packets land? | The **same DAG** — it processes only the new packets |
# MAGIC | 7 | Do the extracted fields **match the system of record**? Is anything **missing**? | SQL + Unity Catalog functions |
# MAGIC | 8 | Which mismatches are **noise**, which need an **analyst**, and which must **stop the money**? | `ai_decide` |
# MAGIC | 💬 | Along the way: **ask Genie** the same questions in plain English — no SQL | **Genie** agent on Unity Catalog |
# MAGIC | 🛠 | Along the way: **Try it yourself** exercises, and a *Build more* list at the end | |
# MAGIC
# MAGIC ---
# MAGIC ### How to run
# MAGIC 1. Keep the default **Serverless** compute (environment version **4 or later** — check the **Environment** side panel). The Job's AI Function tasks run on **serverless**; its SQL tasks and Genie use a **Serverless or Pro SQL warehouse** in your workspace.
# MAGIC 2. A **workspace admin** enables two Beta previews on the **Previews** page (profile menu → *Previews*): **Lakeflow AI Function Task** (the native *Document intelligence* task types) and **AI Decide** (`ai_decide`). `ai_parse_document`, `ai_classify` and `ai_extract` must be available in your [region](https://docs.databricks.com/aws/en/large-language-models/ai-functions). **If a step fails, this is the first thing to check.**
# MAGIC 3. Run the **Setup** steps (**S.1–S.3**) and set the widgets (`catalog`, `schema`, …) to a location where you can create tables, volumes and jobs. Then either **Run all** (about 25 minutes: the backfill Job run is ~11, the daily run ~7 — each AI Function task has a couple of minutes of start-up), or — better for learning — run section by section: sections end with a **💬 Ask Genie** cell and a **🛠 Try it yourself** exercise.
# MAGIC 4. **Every section and cell is numbered** (S.1, 0.1, 1.1 …) — look for the number in each cell's title.
# MAGIC 5. **New to some of the code?** Ask **Genie Code** to explain any cell: click into it, open Genie Code and type `/explain` (or ask *"what does this cell do?"*).
# MAGIC
# MAGIC > ⚠️ **If a step fails** (e.g. *Document intelligence task type not available*, *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → enable **Lakeflow AI Function Task** and **AI Decide** (`ai_decide`). Also confirm the serverless **environment version is 4 or later**. For a failed **Job task**, open the run from step 2.3, click the red task to see its error, fix it and click **Repair run** — only the failed task (and what depends on it) re-runs.
# MAGIC
# MAGIC > **No documents? No problem.** Section 0 generates realistic, fully synthetic PDF packets (no real customer data) plus matching system-of-record tables — with a few **planted problems** for the AI to find. **Have your own documents?** See *"Bring your own documents"* (step 0.11).
# MAGIC
# MAGIC *All names, institutions, accounts, policies and amounts in this notebook are fictitious.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## S. Setup — choose where to write the demo tables and documents
# MAGIC Three short setup steps. The **widgets** (the notebook's parameters) drive everything below.
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | **S.1** | Creates the widgets |
# MAGIC | **S.2** | Turns the widget values into SQL session variables and creates / selects the schema and the two **volumes** (document inbox, page images) |
# MAGIC | **S.3** | Loads the Python helper functions used later (Job builder & runner, Unity Catalog links, packet viewer, Genie) |
# MAGIC
# MAGIC | Widget | Meaning | Default |
# MAGIC |---|---|---|
# MAGIC | `catalog` / `schema` | Where tables and volumes are created in **Unity Catalog** (schema is created if missing) | your current catalog / `fins_doc_intel_demo` |
# MAGIC | `mode` | `backfill` = **reset** the demo and land an archive of past packets · `keep` = keep everything (to re-run sections after a first backfill) | `backfill` |
# MAGIC | `backfill_days` / `packets_per_day` | Size of the synthetic archive (business days × packets) and of each day's arrivals. Bigger = more scenarios, but a longer backfill run | `3` × `3` = 9 packets |
# MAGIC | `amount_tolerance_pct` | Amount differences within this % are treated as a match (section 7) | `1` |
# MAGIC | `question` / `custom_question` | What to ask Genie in step 10.1 | starter questions |
# MAGIC
# MAGIC > 🧭 **Where are the widgets?** They show up once you run **S.1**, in one of two places depending on your layout:
# MAGIC > * **A bar across the top** of the notebook (the classic layout), or
# MAGIC > * **A panel on the left-hand side** — click the **Parameters** icon (the parameters symbol in the left sidebar) to open it.
# MAGIC >
# MAGIC > Same widgets either way — use whichever you prefer.
# MAGIC
# MAGIC > 🔁 **Changed a widget?** Re-run **S.2 and S.3**, then the cells you want to refresh (or just **Run all**). With the default widget settings, changing a value does **not** re-run SQL cells, so the new value isn't picked up until S.2 and S.3 run again.

# COMMAND ----------

# Widgets — they appear in the bar at the top of the notebook, or in the left-hand Parameters panel
GENIE_QUESTIONS = [
    "Which packets are on hold and why?",
    "What share of packets went straight through with no human touch, by line of business?",
    "Which capital call packets have wire instructions that don't match our records?",
    "Which loan packets have W-2 wages more than 5% below the income in the loan system?",
    "Which claims have a repair estimate above the reserve, and by how much?",
    "Which document types and fields produce the most exceptions?",
]
dbutils.widgets.text("catalog", "", "1. Catalog (blank = current)")
dbutils.widgets.text("schema", "fins_doc_intel_demo", "2. Schema")
dbutils.widgets.dropdown("mode", "backfill", ["backfill", "keep"], "3. Mode (backfill = reset + land archive)")
dbutils.widgets.dropdown("backfill_days", "3", ["3", "5", "10"], "4. Backfill: business days of history")
dbutils.widgets.dropdown("packets_per_day", "3", ["2", "3", "4", "8"], "5. Packets arriving per day")
dbutils.widgets.dropdown("amount_tolerance_pct", "1", ["0.5", "1", "2", "5"], "6. Amount match tolerance (%)")
dbutils.widgets.dropdown("question", GENIE_QUESTIONS[0], GENIE_QUESTIONS, "7. Ask Genie (any time)")
dbutils.widgets.text("custom_question", "", "8. ...or type your own Genie question")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Turn the widgets into session variables (with safe defaults if a widget is left blank).
# MAGIC -- Changed a widget? Re-run this cell (S.2) and the helpers (S.3), then the cells you want to refresh.
# MAGIC DECLARE OR REPLACE VARIABLE target_catalog       STRING DEFAULT coalesce(nullif(:catalog, ''), current_catalog());
# MAGIC DECLARE OR REPLACE VARIABLE target_schema        STRING DEFAULT coalesce(nullif(:schema, ''), 'fins_doc_intel_demo');
# MAGIC DECLARE OR REPLACE VARIABLE run_mode             STRING DEFAULT coalesce(nullif(:mode, ''), 'backfill');
# MAGIC DECLARE OR REPLACE VARIABLE amount_tolerance_pct DOUBLE DEFAULT CAST(coalesce(nullif(:amount_tolerance_pct, ''), '1') AS DOUBLE);
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS IDENTIFIER(target_catalog || '.' || target_schema)
# MAGIC COMMENT 'FINS AI Functions industry play: loan, investment & claims document processing (synthetic data)';
# MAGIC
# MAGIC USE IDENTIFIER(target_catalog || '.' || target_schema);
# MAGIC
# MAGIC -- Unity Catalog volumes: governed storage for files
# MAGIC CREATE VOLUME IF NOT EXISTS inbox       COMMENT 'Incoming document packets (PDF), one folder per arrival date';
# MAGIC CREATE VOLUME IF NOT EXISTS page_images COMMENT 'Page images rendered by ai_parse_document (for review screens)';
# MAGIC
# MAGIC SELECT current_catalog() AS catalog, current_schema() AS schema, run_mode, amount_tolerance_pct;

# COMMAND ----------

# MAGIC %md
# MAGIC ### S.3 Load the helper functions
# MAGIC The next cell **doesn't create any tables or change any data** — it only defines Python helpers the rest of the notebook calls, so later cells stay short and readable:
# MAGIC
# MAGIC | Helper | Used in | What it does |
# MAGIC |---|---|---|
# MAGIC | `WorkspaceClient()`, `CATALOG`, `SCHEMA`, `INBOX` | everywhere | Connects to this workspace with **your** identity and remembers where the demo tables and documents live (from S.2) |
# MAGIC | `refresh_inbox_view()` | step 0.6 | Defines the `inbox_files` view over the inbox volume with `read_files` (it needs a literal path, so Python builds it) |
# MAGIC | `define_ai_task()`, `define_sql_task()`, `job_settings()` | sections 1–2 | Registers each Job task — a native **AI Function task** (function, input, output table, options) or a **SQL task** (its SQL) — and builds the Jobs API definition |
# MAGIC | `show_dag()`, `run_pipeline()`, `show_run()` | sections 1, 2 and 6 | Draws the DAG, runs the Job and waits, and renders a per-task timeline with each AI task's row counts |
# MAGIC | `show_packet()` | steps 3.3, 4.2 | Renders a packet's page images — and, once the classification tasks have run, the predicted document type and where each document starts |
# MAGIC | `uc_url()`, `button()`, `show_uc_cards()` | steps 0.7, 9.2 | Builds clickable **Unity Catalog** links and the table cards (schema · sample · lineage) |
# MAGIC | `pick_warehouse()`, `ask_genie()`, `show_genie_answer()` | every 💬 Ask Genie step | Asks the **Genie agent** a question through the Genie API and renders the answer, the SQL Genie wrote and a *Continue in Genie* link |
# MAGIC
# MAGIC When it finishes it prints the catalog / schema, the mode, the Job name and the SQL warehouse Genie will use. **Re-run this cell whenever you change a widget** (after S.2).
# MAGIC
# MAGIC > 💡 **Don't fully understand a cell?** Ask **Genie Code** to explain it: click into the cell, open Genie Code and type `/explain` — or just ask *"what does this cell do?"* in plain English.

# COMMAND ----------

# Helpers: Unity Catalog links, inbox view, Job builder & runner, packet viewer and Genie
import json, time, uuid, html, re, os, io, base64
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
CATALOG, SCHEMA = spark.sql("SELECT current_catalog(), current_schema()").first()
HOST = w.config.host.rstrip("/")
try:
    ORG = f"?o={w.get_workspace_id()}"
except Exception:
    ORG = ""
MODE = dbutils.widgets.get("mode") or "backfill"
BACKFILL_DAYS = int(dbutils.widgets.get("backfill_days") or 5)
PACKETS_PER_DAY = int(dbutils.widgets.get("packets_per_day") or 4)
INBOX = f"/Volumes/{CATALOG}/{SCHEMA}/inbox"
PAGE_IMAGES = f"/Volumes/{CATALOG}/{SCHEMA}/page_images"
fq = lambda t: f"`{CATALOG}`.`{SCHEMA}`.`{t}`"
NOTEBOOK_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
NOTEBOOK_DIR = NOTEBOOK_PATH.rsplit("/", 1)[0]
JOB_NAME = f"FINS document intelligence — parse, classify, extract ({SCHEMA})"

def refresh_inbox_view():
    """read_files needs a literal path, so the inbox view is (re)defined here with the volume path filled in."""
    spark.sql(f"""CREATE OR REPLACE TEMP VIEW inbox_files AS
        SELECT path, regexp_extract(path, '([^/]+)\\\\.pdf$', 1) AS packet_id,
               to_date(regexp_extract(path, '/inbox/(\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}})/', 1)) AS received_date,
               length, content
        FROM read_files('{INBOX}/', format => 'binaryFile', pathGlobFilter => '*.pdf', recursiveFileLookup => 'true')""")

# ---------------------------------------------------------------- Lakeflow Job: task registry, job definition, runner
TASKS = {}   # filled in by section 1, in dependency order: native AI Function tasks and SQL tasks

def _card(key, kind, badge, title, deps, detail):
    color = "#FF3621" if kind == "ai" else "#1B3139"
    displayHTML(f"""<div style="{FONT};border:1px solid #e0e0e0;border-left:5px solid {color};border-radius:8px;padding:10px 14px;max-width:950px">
      <span style="font-size:15px;font-weight:700">Task <code>{key}</code></span>
      <span style="background:{color};color:white;border-radius:4px;padding:2px 6px;font-size:11px;margin-left:6px">{html.escape(badge)}</span>
      <div style="font-size:13px;margin-top:4px">{html.escape(title)}</div>
      <div style="font-size:12px;color:#555;margin-top:4px">Runs after: <b>{html.escape(", ".join(deps) or "nothing — first task")}</b> · {detail}</div></div>""")

def define_ai_task(key, title, function, depends_on, input, output_table, options):
    """Registers a native AI Function task: one function over every file in a volume or every row of a table (with change data feed)."""
    spec = {"ai_function_task": {**input, "output": {"catalog": CATALOG, "schema": SCHEMA, "table": output_table}, function: options}}
    TASKS[key] = dict(key=key, kind="ai", title=title, badge=f"AI Function task · {function}", function=function,
                      depends_on=list(depends_on), output_table=output_table, spec=spec)
    src = input.get("volume", {}).get("volume_path") or input["delta_table"]["table_name"] + "." + input["delta_table"]["input_column"]
    _card(key, "ai", TASKS[key]["badge"], title, depends_on, f"input <code>{html.escape(src)}</code> → output table <code>{output_table}</code>")
    print(json.dumps(spec, indent=2)[:3000])

def define_sql_task(key, title, depends_on, sql):
    """Registers a SQL task: plain SQL (no AI) run on the SQL warehouse — the glue between AI Function tasks."""
    TASKS[key] = dict(key=key, kind="sql", title=title, badge="SQL task", function="SQL", depends_on=list(depends_on), sql=sql.strip())
    _card(key, "sql", "SQL task", title, depends_on, f"{len(sql.strip().splitlines())} lines of SQL on warehouse <b>{html.escape(WAREHOUSE.name)}</b>")

SQL_DIR = f"{NOTEBOOK_DIR}/document_intelligence_sql"
def sql_file_path(key):
    return f"/Workspace{SQL_DIR}/{list(TASKS).index(key) + 1:02d}_{key}.sql"

def sql_file_content(spec):
    return (f"-- SQL task `{spec['key']}` of the Lakeflow Job \"{JOB_NAME}\": {spec['title']}\n"
            f"-- Generated by step 2.1 of the notebook {NOTEBOOK_PATH} - edit it there (section 1) and re-run 2.1.\n"
            f"USE CATALOG `{CATALOG}`;\nUSE SCHEMA `{SCHEMA}`;\n\n{spec['sql']}\n")

def job_settings():
    """The Jobs API definition: one task per TASKS entry, wired by depends_on."""
    tasks = []
    for s in TASKS.values():
        t = {"task_key": s["key"], "description": s["title"][:1000],
             "depends_on": [{"task_key": d} for d in s["depends_on"]],
             "max_retries": 1, "min_retry_interval_millis": 30000, "retry_on_timeout": True, "timeout_seconds": 3600}
        if s["kind"] == "ai":
            t.update(s["spec"])
        else:
            t["sql_task"] = {"warehouse_id": WAREHOUSE.id, "file": {"path": sql_file_path(s["key"]), "source": "WORKSPACE"}}
        tasks.append(t)
    return {"name": JOB_NAME,
            "description": "Document intelligence DAG of native AI Function tasks: ai_parse_document -> ai_classify (line of business, then document type) "
                           "-> split -> ai_extract (one task per line of business) -> assemble. Every AI task processes only new inputs, "
                           "so the first run is the backfill and each scheduled run handles only new packets.",
            "tasks": tasks,
            "schedule": {"quartz_cron_expression": "0 0 6 * * ?", "timezone_id": "America/New_York", "pause_status": "PAUSED"},
            "max_concurrent_runs": 1,          # required for jobs with AI Function tasks
            "queue": {"enabled": True},
            "tags": {"demo": "fins-ai-functions-document-intelligence"}}

def dag_levels():
    level = {}
    for k, s in TASKS.items():
        level[k] = 1 + max([level[d] for d in s["depends_on"]] or [0])
    return level

def show_dag():
    level = dag_levels()
    cols = []
    for lv in sorted(set(level.values())):
        boxes = "".join(f"""<div style="border:2px solid {'#FF3621' if TASKS[k]['kind'] == 'ai' else '#1B3139'};border-radius:8px;padding:6px 8px;margin:5px 0;background:white;width:150px">
            <div style="font-weight:700;font-size:12px">{k}</div>
            <div style="display:inline-block;background:{'#FF3621' if TASKS[k]['kind'] == 'ai' else '#1B3139'};color:white;border-radius:4px;padding:1px 5px;font-size:10px;margin-top:3px">{html.escape(TASKS[k]['function'] if TASKS[k]['kind'] == 'ai' else 'SQL task')}</div></div>""" for k in TASKS if level[k] == lv)
        cols.append(f'<div style="display:flex;flex-direction:column;justify-content:center">{boxes}</div>')
    arrow = '<div style="display:flex;align-items:center;font-size:22px;color:#FF3621;padding:0 3px">➜</div>'
    n_ai = sum(s["kind"] == "ai" for s in TASKS.values())
    displayHTML(f'<div style="{FONT}"><div style="font-size:15px;font-weight:700;margin-bottom:4px">Lakeflow Job DAG — {len(TASKS)} tasks: '
                f'<span style="color:#FF3621">{n_ai} AI Function tasks</span> + {len(TASKS) - n_ai} SQL tasks</div>'
                f'<div style="display:flex;background:#F9F7F4;border-radius:10px;padding:10px;overflow-x:auto">{arrow.join(cols)}</div></div>')

def get_job_id():
    found = list(w.jobs.list(name=JOB_NAME))
    if not found:
        raise Exception(f"Job '{JOB_NAME}' not found — run step 2.1 first.")
    return found[0].job_id

def _task_rows(task_run_id):
    """Row counts an AI Function task reports for a run: {'rows': total, 'succeeded': .., 'failed': ..} ({} for SQL tasks)."""
    try:
        o = _api("GET", "/api/2.2/jobs/runs/get-output", query={"run_id": task_run_id}).get("ai_function_task_output") or {}
        return {"rows": o.get("total_row_count", 0), "succeeded": o.get("succeeded_row_count", 0), "failed": o.get("failed_row_count", 0)} if o else {}
    except Exception:
        return {}

def show_run(run_id):
    """Per-task timeline of a Job run: status, duration bar, and the rows each AI Function task processed."""
    r = w.jobs.get_run(run_id)
    tasks = [t for t in r.tasks if t.start_time]
    t0 = min(t.start_time for t in tasks) if tasks else 0
    t1 = max((t.end_time or t.start_time) for t in tasks) if tasks else 1
    span = max(t1 - t0, 1)
    rows = []
    for t in sorted(r.tasks, key=lambda t: (t.start_time or 0, t.task_key)):
        state = (t.state.result_state.value if t.state.result_state else t.state.life_cycle_state.value) if t.state else "?"
        color = {"SUCCESS": "#00A972", "FAILED": "#FF3621"}.get(state, "#9AA5AB")
        left = ((t.start_time or t0) - t0) / span * 100
        dur = ((t.end_time or t.start_time or t0) - (t.start_time or t0)) / 1000
        width = max(dur * 1000 / span * 100, 1)
        spec = TASKS.get(t.task_key, {})
        m = _task_rows(t.run_id) if spec.get("kind") == "ai" else {}
        metrics = (f"<b>{m['rows']}</b> new inputs · {m['succeeded']} succeeded · " + (f"<span style='color:#FF3621'>{m['failed']} failed</span>" if m["failed"] else "0 failed")) if m else ""
        retry = f' <span style="color:#FF3621">(attempt {t.attempt_number + 1})</span>' if t.attempt_number else ""
        rows.append(f"""<tr><td style="padding:4px 8px;font-weight:700">{t.task_key}{retry}</td>
          <td style="padding:4px 8px;font-size:11px">{html.escape(spec.get('badge', ''))}</td>
          <td style="padding:4px 8px;color:{color};font-weight:700">{state}</td>
          <td style="padding:4px 8px;width:300px"><div style="position:relative;height:14px;background:#F0F0F0;border-radius:3px">
            <div style="position:absolute;left:{left:.1f}%;width:{width:.1f}%;height:14px;background:{color};border-radius:3px"></div></div>
            <div style="font-size:11px;color:#555">{dur:.0f} s</div></td>
          <td style="padding:4px 8px;font-size:12px">{metrics}</td></tr>""")
    total = ((r.end_time or t1) - (r.start_time or t0)) / 1000
    displayHTML(f"""<div style="{FONT}"><div style="font-size:15px;font-weight:700">Job run {run_id} · {r.state.result_state.value if r.state.result_state else r.state.life_cycle_state.value} · {total / 60:.1f} min</div>
      {button("Open this run (graph, task output, Repair run)", r.run_page_url)}
      <table style="border-collapse:collapse;margin-top:6px;font-size:13px"><tr style="background:#F9F7F4;text-align:left">
      <th style="padding:4px 8px">Task</th><th style="padding:4px 8px">Type</th><th style="padding:4px 8px">Status</th><th style="padding:4px 8px">Timeline</th><th style="padding:4px 8px">AI Function task rows</th></tr>
      {''.join(rows)}</table></div>""")

def run_pipeline(wait=True):
    """Triggers the Job (Run now), prints task progress until it finishes, then shows the per-task timeline."""
    job_id = get_job_id()
    waiter = w.jobs.run_now(job_id=job_id)
    run_id = getattr(waiter, "run_id", None) or waiter.response.run_id
    print(f"Started Job run {run_id}: {HOST}/jobs/{job_id}/runs/{run_id}{ORG}")
    if not wait:
        return run_id
    t0, last = time.time(), None
    while True:
        r = w.jobs.get_run(run_id)
        done = sum(1 for t in r.tasks if t.state and t.state.result_state)
        running = [t.task_key for t in r.tasks if t.state and t.state.life_cycle_state and t.state.life_cycle_state.value == "RUNNING"]
        line = f"{done}/{len(r.tasks)} tasks done" + (f" · running: {', '.join(running)}" if running else "")
        if line != last:
            el = int(time.time() - t0)
            print(f"[{el // 60}m{el % 60:02d}s] {line}")
            last = line
        if r.state.life_cycle_state.value in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(15)
    show_run(run_id)
    if not r.state.result_state or r.state.result_state.value != "SUCCESS":
        raise Exception(f"The Job run did not succeed — open it, check the red task, fix and click Repair run: {r.run_page_url}")
    return run_id

def pick_warehouse():
    """A Serverless (or Pro) SQL warehouse for Genie and the Job's SQL tasks. Only warehouse type PRO qualifies —
    serverless SQL warehouses report PRO; Real-Time and other special-purpose warehouses don't work for Genie or SQL tasks."""
    whs = [x for x in w.warehouses.list() if x.warehouse_type is not None and x.warehouse_type.value == "PRO"]
    if not whs:
        raise Exception("Create a Serverless (or Pro) SQL warehouse — Genie and the Job's SQL tasks run on it.")
    whs.sort(key=lambda x: (not x.enable_serverless_compute, "RUNNING" not in str(x.state)))   # serverless first, then running
    return whs[0]
WAREHOUSE = pick_warehouse()

def uc_url(*parts, tab=None):
    q = ORG + ((("&" if ORG else "?") + f"activeTab={tab}") if tab else "")
    return f"{HOST}/explore/data/" + "/".join(parts) + q

FONT = "font-family:DM Sans,Helvetica,Arial,sans-serif"
def button(label, url, color="#FF3621"):
    return (f'<a href="{url}" target="_blank" style="display:inline-block;margin:4px 8px 4px 0;padding:8px 14px;'
            f'background:{color};color:white;border-radius:6px;text-decoration:none;font-weight:600;{FONT};font-size:13px">'
            f'{html.escape(label)} ↗</a>')

TABLE_INFO = {   # table -> (what produces it, description)
    "sor_loans": ("Source", "Loan origination system — mortgage applications in underwriting"),
    "sor_investor_commitments": ("Source", "Fund administration — investor commitments, capital calls, wire details"),
    "sor_claims": ("Source", "Claims system — open auto claims, policy period, VIN, reserve"),
    "inbox_documents": ("SQL task · register_new_files", "Every PDF that has landed in the inbox, with its content (input to parse_documents)"),
    "parsed_packets": ("AI Function task · parse_documents", "ai_parse_document output per PDF (managed by the task)"),
    "packet_pages": ("SQL task · prepare_pages", "One row per page: page text and page image"),
    "page_line_of_business": ("AI Function task · classify_line_of_business", "ai_classify: Lending / Investments / Claims / Other per page"),
    "pages_lending": ("SQL task · route_pages", "Pages routed to the lending document classifier"),
    "pages_investments": ("SQL task · route_pages", "Pages routed to the investments document classifier"),
    "pages_claims": ("SQL task · route_pages", "Pages routed to the claims document classifier"),
    "page_types_lending": ("AI Function task · classify_lending_pages", "ai_classify: lending document type per page"),
    "page_types_investments": ("AI Function task · classify_investment_pages", "ai_classify: fund document type per page"),
    "page_types_claims": ("AI Function task · classify_claim_pages", "ai_classify: claim document type per page"),
    "page_classifications": ("SQL task · split_documents", "Final document type of every page"),
    "packet_documents": ("SQL task · split_documents", "Packets split into documents (type, page range, text)"),
    "documents_lending": ("SQL task · split_documents", "Lending documents waiting for extraction"),
    "documents_investments": ("SQL task · split_documents", "Fund documents waiting for extraction"),
    "documents_claims": ("SQL task · split_documents", "Claim documents waiting for extraction"),
    "extracted_lending": ("AI Function task · extract_lending", "ai_extract output for loan-file documents"),
    "extracted_investments": ("AI Function task · extract_investments", "ai_extract output for fund documents"),
    "extracted_claims": ("AI Function task · extract_claims", "ai_extract output for claim documents"),
    "extracted_fields": ("SQL task · assemble_extractions", "One row per extracted field: value + confidence"),
    "packet_index": ("SQL task · assemble_extractions", "One row per packet: line of business and system-of-record key"),
    "recon_results": ("Notebook · section 7", "Every reconciled field: document vs system of record"),
    "exception_decisions": ("Notebook · section 8 (ai_decide)", "ai_decide output per exception (cached by fact hash)"),
    "exceptions": ("Notebook · section 8 (ai_decide)", "Exceptions with action, fraud probability and materiality"),
    "packet_status": ("Notebook · section 8", "Packet decision: STRAIGHT_THROUGH / REVIEW / HOLD"),
}
ROLE_COLOR = {"Source": "#5A6F77"}
role_color = lambda r: ROLE_COLOR.get(r, "#FF3621" if r.startswith("AI Function task") else "#1B3139" if r.startswith("SQL task") else "#00A972")

def show_uc_cards(roles=None, title=None):
    existing = {r.tableName for r in spark.sql(f"SHOW TABLES IN `{CATALOG}`.`{SCHEMA}`").collect()}
    cards = []
    for t, (role, desc) in TABLE_INFO.items():
        if t not in existing or (roles and not any(role.startswith(x) for x in roles)):
            continue
        if spark.table(fq(t)).columns == ["placeholder"]:
            continue
        n = spark.table(fq(t)).count()
        cards.append(f"""<div style="border:1px solid #e0e0e0;border-radius:10px;padding:12px 14px;width:280px;{FONT}">
          <div style="font-size:11px;font-weight:700;color:{role_color(role)};text-transform:uppercase;letter-spacing:.04em">{role}</div>
          <div style="font-size:15px;font-weight:700;margin:4px 0"><a href="{uc_url(CATALOG, SCHEMA, t)}" target="_blank" style="color:#1B3139">{t}</a></div>
          <div style="font-size:12px;color:#555;min-height:32px">{desc}</div>
          <div style="font-size:12px;margin-top:6px">{n:,} rows · <a href="{uc_url(CATALOG, SCHEMA, t, tab='sample')}" target="_blank">sample</a> ·
          <a href="{uc_url(CATALOG, SCHEMA, t, tab='lineage')}" target="_blank">lineage</a></div></div>""")
    displayHTML(f"""<div style="{FONT}">{f'<div style="font-size:16px;font-weight:700;margin-bottom:6px">{title}</div>' if title else ''}
      {button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139")}
      {button("Open the document inbox (volume)", uc_url("volumes", CATALOG, SCHEMA, "inbox"), "#5A6F77")}
      <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:10px">{''.join(cards)}</div></div>""")

FAMILY_COLOR = {"Lending": "#1B5E9B", "Investments": "#7B3FA0", "Claims": "#00875A", "Other": "#9AA5AB"}
def _thumb(path, width=210):
    data = open(path, "rb").read()
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data)); im.thumbnail((width * 2, width * 3))
        buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=70); data = buf.getvalue()
    except Exception:
        pass
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()

def show_packet(packet_id=None, labels=False):
    """Shows every page of a packet. labels=True adds the ai_classify document type and document boundaries (step 2)."""
    if packet_id is None:   # default: the packet with the most pages (usually a loan file with a fax cover)
        packet_id = spark.sql(f"SELECT packet_id FROM {fq('packet_pages')} GROUP BY 1 ORDER BY count(*) DESC, packet_id DESC LIMIT 1").first()[0]
    q = f"SELECT p.page_number, p.image_uri, p.page_text"
    if labels:
        q += ", d.doc_type, d.doc_seq, d.first_page, d.doc_family"
    q += f" FROM {fq('packet_pages')} p"
    if labels:
        q += f" LEFT JOIN {fq('packet_documents')} d ON d.packet_id = p.packet_id AND p.page_number BETWEEN d.first_page AND d.last_page"
    rows = spark.sql(q + f" WHERE p.packet_id = '{packet_id}' ORDER BY p.page_number").collect()
    cards = []
    for r in rows:
        color = FAMILY_COLOR.get(r["doc_family"], "#9AA5AB") if labels else "#D0D5D8"
        top = ""
        if labels:
            starts = r["page_number"] == r["first_page"]
            top = (f'<div style="font-size:11px;font-weight:700;color:{color};height:16px">{"▶ document " + str(r["doc_seq"]) + " starts" if starts else "&nbsp;↳ continued"}</div>'
                   f'<div style="display:inline-block;background:{color};color:white;border-radius:4px;padding:2px 6px;font-size:11px;font-weight:700;margin:2px 0 4px">{html.escape(str(r["doc_type"]))}</div>')
        try:
            img = f'<img src="{_thumb(r["image_uri"])}" style="width:100%;border:1px solid #ddd">'
        except Exception:
            img = f'<pre style="font-size:9px;white-space:pre-wrap;height:260px;overflow:hidden">{html.escape((r["page_text"] or "")[:600])}</pre>'
        cards.append(f'<div style="width:210px;border-top:5px solid {color};padding-top:4px">{top}{img}'
                     f'<div style="font-size:11px;color:#666">page {r["page_number"]}</div></div>')
    displayHTML(f'<div style="{FONT}"><div style="font-size:15px;font-weight:700;margin-bottom:8px">Packet {html.escape(packet_id)} · {len(rows)} pages</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:12px">{"".join(cards)}</div></div>')

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

def _api(method, path, body=None, query=None):
    return w.api_client.do(method, path, body=body, query=query)

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
    tables = sorted(set(re.findall(r"\b" + re.escape(SCHEMA) + r"`?\.`?(\w+)", sql))) if sql else []
    answer = md_to_html("\n".join(res["texts"])) or "Genie is still working on this one — open it in Genie to see the answer."
    compare = ""
    if before:
        compare = f"""<div style="display:flex;gap:12px;margin:10px 0 12px">
          <div style="flex:1;background:#F9F7F4;border-radius:8px;padding:10px 12px">
            <div style="font-size:11px;font-weight:700;color:#5A6F77;text-transform:uppercase">Without Genie</div>
            <div style="font-size:14px;margin-top:4px">{html.escape(before[0])} needed <b>{before[1]} lines of hand-written SQL</b> — written and maintained by an analyst.</div></div>
          <div style="flex:1;background:#FFF1EE;border-radius:8px;padding:10px 12px">
            <div style="font-size:11px;font-weight:700;color:#FF3621;text-transform:uppercase">With Genie</div>
            <div style="font-size:14px;margin-top:4px">One plain-English question. Genie wrote <b>{sql_lines} lines of SQL</b>{f" across {len(tables)} governed table" + ("s" if len(tables) != 1 else "") if tables else ""} in <b>{res['seconds']} seconds</b>.</div></div></div>"""
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

print(f"Writing to {CATALOG}.{SCHEMA} · mode = {MODE} · inbox = {INBOX}\nJob: '{JOB_NAME}' · SQL warehouse for Genie: '{WAREHOUSE.name}'")
displayHTML(button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Generate synthetic documents & systems of record *(skip 0.1–0.5 if you bring your own)*
# MAGIC
# MAGIC *This section plays the part of the outside world: it drops PDFs into the inbox and fills the systems of record. Everything from section 1 on is what you would build for real (except step 6.1, which plays "tomorrow's" arrivals).*
# MAGIC
# MAGIC A fictitious diversified institution — **Harborline Financial Group** — with three operations teams whose inboxes look like everyone's:
# MAGIC
# MAGIC | Line of business | Packet (one PDF) | Documents inside | System of record |
# MAGIC |---|---|---|---|
# MAGIC | **Lending** — Harborline Bank mortgage ops | Loan file | Loan application (1003), W-2, **two** pay stubs, bank statement, appraisal, Loan Estimate | Loan origination system → `sor_loans` |
# MAGIC | **Investments** — Harborline Capital investor services | Subscription / capital call package | Term summary, subscription agreement, W-9, capital call notice, wire instructions | Fund administration → `sor_investor_commitments` |
# MAGIC | **Insurance** — Harborline Mutual claims intake | Auto claim packet | First notice of loss, police report, repair estimate, medical bill (sometimes), proof of loss | Claims system → `sor_claims` |
# MAGIC
# MAGIC Packets behave like the real thing: documents arrive **in any order**, multi-page documents sit next to single-page ones, two pay stubs come **back to back**, and about a third of packets start with a **fax cover sheet**.
# MAGIC
# MAGIC Most packets have a **planted problem** for the pipeline to find — some that matter, and some that only *look* like problems. The default 9-packet backfill (kept small so the backfill Job run stays short) already includes **all three HOLD cases**, real REVIEW issues, look-alikes that should auto-clear and clean packets; raise `backfill_days` / `packets_per_day` to add the rest:
# MAGIC
# MAGIC | Should stop the money (HOLD) | Needs an analyst (REVIEW) | Looks wrong, but isn't (AUTO-CLEAR) |
# MAGIC |---|---|---|
# MAGIC | Wire instructions with a **changed bank account** · repair estimate for a **different VIN** · loss date **outside the policy period** | W-2 wages **below** stated income · **low appraisal** · **transposed** loan amount · capital call / fee mismatch · estimate **above reserve** · police vs FNOL **date conflict** · **missing** document | **Nicknames** (Jon vs Jonathan) · **legal-name variants** (LLC vs Limited Liability Company) · **address abbreviations** (Ave vs Avenue, ZIP+4) |
# MAGIC
# MAGIC The generator also writes an **answer key** (`answer_key_pages`, `answer_key_issues`) so steps 4.3 and 8.5 can measure how well the AI did. Everything is seeded, so results are reproducible, and dates are relative to today, so the data always looks current.

# COMMAND ----------

# --- Minimal, dependency-free PDF writer (text, rules, shaded bars, simple tables) ---
def _esc(s):
    return str(s).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1", "replace").decode("latin-1")

class Page:
    W, H, M = 612, 792, 54          # US Letter, 0.75in margins
    def __init__(self):
        self.ops, self.y = [], self.H - self.M
    def _text(self, x, y, s, size=10, bold=False, gray=0.0):
        self.ops.append(f"BT {gray:.2f} g /{'F2' if bold else 'F1'} {size} Tf {x:.1f} {y:.1f} Td ({_esc(s)}) Tj ET 0 g")
    def bar(self, text, sub=None, shade=0.15):
        self.ops.append(f"{shade:.2f} g {self.M} {self.y - 34} {self.W - 2 * self.M} 40 re f 0 g")
        self._text(self.M + 10, self.y - 14, text, 15, True, 1.0)
        if sub:
            self._text(self.M + 10, self.y - 28, sub, 8.5, False, 0.85)
        self.y -= 52
    def title(self, text, size=12):
        self._text(self.M, self.y, text, size, True); self.y -= size + 8
    def line(self, text, size=10, bold=False, indent=0, gray=0.0):
        self._text(self.M + indent, self.y, text, size, bold, gray); self.y -= size + 5
    def para(self, text, size=9.5, width=None):
        width = width or (self.W - 2 * self.M)
        max_chars = int(width / (size * 0.5))
        words, cur = str(text).split(), ""
        for wd in words:
            if len(cur) + len(wd) + 1 > max_chars:
                self.line(cur, size); cur = wd
            else:
                cur = (cur + " " + wd).strip()
        if cur:
            self.line(cur, size)
        self.y -= 4
    def rule(self):
        self.ops.append(f"0.6 G 0.5 w {self.M} {self.y + 4} m {self.W - self.M} {self.y + 4} l S 0 G")
        self.y -= 10
    def fields(self, pairs, cols=2):
        colw = (self.W - 2 * self.M) / cols
        for i in range(0, len(pairs), cols):
            for j, (k, v) in enumerate(pairs[i:i + cols]):
                x = self.M + j * colw
                self._text(x, self.y, k, 7.5, False, 0.4)
                self._text(x, self.y - 12, v, 10.5, True)
            self.y -= 30
    def table(self, headers, rows, widths=None):
        n = len(headers)
        widths = widths or [(self.W - 2 * self.M) / n] * n
        def row(vals, bold=False):
            x = self.M
            for v, wd in zip(vals, widths):
                self._text(x + 3, self.y, v, 9, bold); x += wd
            self.y -= 15
        self.ops.append(f"0.90 g {self.M} {self.y - 4} {sum(widths)} 15 re f 0 g")
        row(headers, True)
        for r in rows:
            row(r)
        self.rule()
    def footer(self, left, right):
        self._text(self.M, 30, left, 7.5, False, 0.45)
        self._text(self.W - self.M - 70, 30, right, 7.5, False, 0.45)

def write_pdf(pages, path):
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"]
    kids = []
    for p in pages:
        stream = "\n".join(p.ops).encode("latin-1")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {Page.W} {Page.H}] "
                    f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {len(objs)} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + (o if isinstance(o, bytes) else o.encode("latin-1")) + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    with open(path, "wb") as f:
        f.write(out)

# COMMAND ----------

# Fictitious people, institutions and amounts. Each packet is seeded by its packet_id, so results are reproducible.
# --- Synthetic packet generator: system-of-record rows + multi-document PDF packets + an answer key ---
import random, datetime as dt, os

FIRST = ["Robert", "William", "Elizabeth", "Katherine", "Jonathan", "Michael", "Margaret", "Christopher",
         "Jennifer", "Daniel", "Patricia", "Thomas", "Rebecca", "Anthony", "Victoria", "Joseph"]
NICK = {"Robert": "Bob", "William": "Bill", "Elizabeth": "Liz", "Katherine": "Kate", "Jonathan": "Jon",
        "Michael": "Mike", "Margaret": "Peggy", "Christopher": "Chris", "Jennifer": "Jen", "Daniel": "Dan",
        "Patricia": "Trish", "Thomas": "Tom", "Rebecca": "Becky", "Anthony": "Tony", "Victoria": "Vicky", "Joseph": "Joe"}
LAST = ["Okafor", "Lindqvist", "Ramirez", "Chen", "Whitaker", "Novak", "Haddad", "Brennan", "Castillo",
        "Kowalski", "Ashworth", "Nakamura", "Delgado", "Fitzgerald", "Mbeki", "Sorensen", "Patel", "Iverson"]
STREETS = [("Maple", "Ave", "Avenue"), ("Juniper", "St", "Street"), ("Harbor View", "Dr", "Drive"),
           ("Old Mill", "Rd", "Road"), ("Briarwood", "Ln", "Lane"), ("Cedar Crest", "Blvd", "Boulevard")]
CITIES = [("Riverton", "OH", "43215"), ("Lakemont", "NC", "27601"), ("Fairhaven", "MA", "02719"),
          ("Westbrook", "CO", "80202"), ("Pine Bluff", "TX", "75201"), ("Glen Arbor", "MN", "55401")]
EMPLOYERS = ["Northwind Logistics Inc", "Brightpath Health Partners", "Keystone Aerospace Corp",
             "Bluewater Analytics LLC", "Granite State Utilities", "Summit Ridge School District"]
ENTITIES = [("Cobalt Peak Holdings LLC", "Cobalt Peak Holdings Limited Liability Company", "LLC"),
            ("Halvorsen Family Trust", "The Halvorsen Family Trust", "Trust"),
            ("Teachers Retirement System of Westbrook", "Westbrook Teachers' Retirement System", "Pension"),
            ("Marlowe Endowment Foundation", "The Marlowe Endowment Foundation, Inc.", "Foundation"),
            ("Ironbridge Family Office LP", "Ironbridge Family Office, Limited Partnership", "LP"),
            ("St. Brendan University Endowment", "Saint Brendan University Endowment", "Endowment")]
FUNDS = [("Harborline Private Credit Fund III, L.P.", 1.50, "Granite Trust Bank", "021000089", "4417"),
         ("Harborline Infrastructure Partners II, L.P.", 1.25, "Granite Trust Bank", "021000089", "7730"),
         ("Harborline Growth Equity Fund IV, L.P.", 2.00, "Atlantic Commerce Bank", "026009593", "1902")]
VEHICLES = [("2022", "Toyota", "RAV4 XLE"), ("2021", "Honda", "Accord EX"), ("2023", "Ford", "F-150 XLT"),
            ("2020", "Subaru", "Outback Premium"), ("2024", "Hyundai", "Tucson SEL"), ("2019", "Chevrolet", "Equinox LT")]
SHOPS = ["Precision Collision Center", "Lakemont Auto Body", "Riverton Paint & Body", "Westside Collision Works"]
PROVIDERS = ["Lakemont Urgent Care", "Riverton Orthopedic Associates", "Fairhaven Physical Therapy"]

ISSUES = {  # issue -> (doc_type, field, expected action)
    "Lending":     {"income_overstated": ("w2", "wages", "review"), "appraisal_low": ("appraisal_report", "appraised_value", "review"),
                    "loan_amount_transposed": ("loan_estimate", "loan_amount", "review"), "borrower_nickname": ("w2", "employee_name", "auto_clear"),
                    "address_abbreviation": ("appraisal_report", "property_address", "auto_clear"), "missing_appraisal": ("appraisal_report", None, "review")},
    "Investments": {"wire_instructions_changed": ("wire_instructions", None, "hold"), "call_amount_mismatch": ("capital_call_notice", "call_amount", "review"),
                    "fee_mismatch": ("term_sheet", "management_fee_pct", "review"), "investor_name_variant": ("subscription_agreement", "investor_name", "auto_clear"),
                    "missing_kyc": ("kyc_w9", None, "review")},
    "Claims":      {"loss_outside_policy": ("fnol_claim_form", "date_of_loss", "hold"), "vin_mismatch": ("repair_estimate", "vehicle_vin", "hold"),
                    "estimate_over_reserve": ("repair_estimate", "estimate_total", "review"), "loss_date_conflict": ("police_report", "incident_date", "review"),
                    "insured_nickname": ("proof_of_loss", "insured_name", "auto_clear"), "missing_proof_of_loss": ("proof_of_loss", None, "review")},
}

usd = lambda x: f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}"
mdy = lambda d: d.strftime("%m/%d/%Y")
long_date = lambda d: d.strftime("%B %d, %Y").replace(" 0", " ")

def _doc(pages, form, doc_type):
    for i, p in enumerate(pages, 1):
        p.footer(form, f"Page {i} of {len(pages)}")
    return [(doc_type, p) for p in pages]

def fax_cover(rng, to, frm, n, received):
    p = Page(); p.y -= 40
    p.line("FAX TRANSMISSION", 22, True); p.y -= 10
    p.fields([("TO", to), ("FROM", frm), ("DATE", mdy(received)), ("PAGES (INCL. COVER)", str(n + 1))])
    p.rule(); p.para("CONFIDENTIALITY NOTICE: The documents accompanying this transmission contain confidential information "
                     "belonging to the sender. If you are not the intended recipient, please notify the sender immediately and destroy all copies.")
    p.line("Please see attached documents for processing.", 10)
    return [("other", p)]

# ---------------- Lending ----------------
def lending_packet(rng, pid, received, issues):
    first, last = rng.choice([f for f in FIRST]), rng.choice(LAST)
    mi = rng.choice("ABCDEJKLMRST")
    name = f"{first} {mi}. {last}"
    num, (st, ab, full) = rng.randint(100, 9899), rng.choice(STREETS)
    city, state, zp = rng.choice(CITIES)
    addr_sor = f"{num} {st} {ab}, {city}, {state} {zp}"
    employer = rng.choice(EMPLOYERS)
    income = round(rng.uniform(72, 260)) * 1000.0
    loan_amt = round(rng.uniform(180, 780)) * 1000.0
    while len(set(str(int(loan_amt))[1:3])) < 2:
        loan_amt += 10000
    value = round(loan_amt / rng.uniform(0.62, 0.9) / 1000) * 1000.0
    rate = rng.choice([5.875, 6.125, 6.25, 6.375, 6.5, 6.625, 6.875])
    loan_no = f"HL-{rng.randint(2000000, 2999999)}"
    sor = dict(loan_number=loan_no, borrower_name=name, property_address=addr_sor, loan_amount=loan_amt, interest_rate=rate,
               term_months=360, loan_purpose=rng.choice(["Purchase", "Purchase", "Rate/Term Refinance"]), employer_name=employer,
               annual_income=income, appraised_value=value, application_date=received - dt.timedelta(days=rng.randint(10, 25)),
               loan_officer=f"{rng.choice(FIRST)} {rng.choice(LAST)}", loan_status="In Underwriting")
    # values as they appear on the documents (planted discrepancies applied here)
    w2_wages = income * rng.uniform(0.995, 1.0)
    if "income_overstated" in issues: w2_wages = income * rng.uniform(0.78, 0.86)
    w2_name = f"{NICK[first]} {mi} {last}".upper() if "borrower_nickname" in issues else f"{first} {mi} {last}".upper()
    appr_value = value * rng.uniform(0.86, 0.91) if "appraisal_low" in issues else value
    appr_addr = f"{num} {st} {full}, {city}, {state} {zp}-{rng.randint(1000, 9999)}" if "address_abbreviation" in issues else addr_sor
    le_amt = loan_amt
    if "loan_amount_transposed" in issues:
        s = str(int(loan_amt)); le_amt = float(s[0] + s[2] + s[1] + s[3:])
    docs = []
    # Loan application (2 pages)
    p1, p2 = Page(), Page()
    p1.bar("Uniform Residential Loan Application", f"Harborline Bank, N.A. - Mortgage Lending - Lender Loan No. {loan_no}")
    p1.title("Section 1: Borrower Information")
    p1.fields([("Name (First, Middle, Last)", name), ("Social Security Number", f"XXX-XX-{rng.randint(1000, 9999)}"),
               ("Date of Birth", mdy(dt.date(rng.randint(1962, 1995), rng.randint(1, 12), rng.randint(1, 28)))),
               ("Home Phone", f"({rng.randint(201, 989)}) 555-{rng.randint(1000, 9999)}"),
               ("Current Address", f"{rng.randint(10, 999)} {rng.choice(STREETS)[0]} St, {city}, {state}"), ("Marital Status", rng.choice(["Married", "Unmarried"]))])
    p1.title("Section 2: Employment and Income")
    p1.fields([("Employer or Business Name", employer), ("Position or Title", rng.choice(["Senior Analyst", "Operations Manager", "Engineer III", "Director"])),
               ("Start Date", mdy(received - dt.timedelta(days=rng.randint(800, 4000)))), ("Gross Annual Base Income", usd(income))])
    p1.title("Section 4: Loan and Property Information")
    p1.fields([("Loan Amount", usd(loan_amt)), ("Loan Purpose", sor["loan_purpose"]), ("Property Address", addr_sor), ("Occupancy", "Primary Residence")])
    p2.title("Section 2 (continued): Assets and Liabilities", 12)
    p2.table(["Account Type", "Financial Institution", "Cash or Market Value"],
             [["Checking", "Lakeshore Community Bank", usd(rng.uniform(8000, 40000))], ["Savings", "Lakeshore Community Bank", usd(rng.uniform(20000, 160000))],
              ["Retirement (401k)", "Fidelis Retirement Services", usd(rng.uniform(40000, 400000))]], [140, 220, 144])
    p2.title("Section 5: Declarations")
    for q in ["Will you occupy the property as your primary residence?  YES", "Are there any outstanding judgments against you?  NO",
              "Have you declared bankruptcy within the past 7 years?  NO", "Are you a party to a lawsuit with potential financial liability?  NO"]:
        p2.line(q, 9.5)
    p2.title("Section 7: Acknowledgments and Agreements")
    p2.para("I agree that the information provided in this application is true, accurate and complete as of the date signed. "
            "I understand that any intentional or negligent misrepresentation may result in civil liability and criminal penalties.")
    p2.fields([("Borrower Signature", f"/s/ {name}"), ("Date", mdy(sor["application_date"]))])
    docs.append(_doc([p1, p2], "Uniform Residential Loan Application - Freddie Mac Form 65 / Fannie Mae Form 1003", "loan_application"))
    # W-2
    p = Page(); p.bar("Form W-2  Wage and Tax Statement", f"Tax Year {received.year - 1}  -  Copy C, For EMPLOYEE'S RECORDS")
    p.fields([("a  Employee's social security number", f"XXX-XX-{rng.randint(1000, 9999)}"), ("b  Employer identification number (EIN)", f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"),
              ("c  Employer's name", employer), ("c  Employer's address and ZIP code", f"{rng.randint(100, 9999)} Commerce Pkwy, {city}, {state} {zp}"),
              ("e  Employee's name", w2_name), ("d  Control number", f"{rng.randint(100000, 999999)}"),
              ("1  Wages, tips, other compensation", usd(w2_wages)), ("2  Federal income tax withheld", usd(w2_wages * 0.17)),
              ("3  Social security wages", usd(min(w2_wages, 176100))), ("5  Medicare wages and tips", usd(w2_wages))])
    p.para("This information is being furnished to the Internal Revenue Service. Department of the Treasury - Internal Revenue Service.")
    docs.append(_doc([p], "Form W-2 (Department of the Treasury - Internal Revenue Service)", "w2"))
    # Two consecutive pay stubs (same type back to back - the splitter must separate them)
    period_pay = income / 26
    for k in range(2):
        end = received - dt.timedelta(days=7 + 14 * k)
        p = Page(); p.bar(f"{employer}", "EARNINGS STATEMENT - Direct Deposit Advice")
        p.fields([("Employee", f"{first} {last}"), ("Pay Period", f"{mdy(end - dt.timedelta(days=13))} - {mdy(end)}"), ("Pay Date", mdy(end + dt.timedelta(days=5))),
                  ("Check / Advice No.", str(rng.randint(100000, 999999)))])
        ytd = period_pay * max(1, (end.timetuple().tm_yday // 14))
        p.table(["Earnings", "Rate", "Hours", "Current", "Year to Date"],
                [["Regular", usd(period_pay / 80), "80.00", usd(period_pay), usd(ytd)], ["Gross Pay", "", "", usd(period_pay), usd(ytd)],
                 ["Federal Tax", "", "", usd(-period_pay * .15), ""], ["Net Pay", "", "", usd(period_pay * .72), ""]], [120, 90, 70, 110, 114])
        docs.append(_doc([p], f"{employer} - Payroll", "pay_stub"))
    # Bank statement (2 pages)
    bal = rng.uniform(15000, 90000)
    p1, p2 = Page(), Page()
    p1.bar("Lakeshore Community Bank", "Personal Checking - Monthly Statement")
    p1.fields([("Account Holder", name), ("Account Number", f"XXXXXX{rng.randint(1000, 9999)}"),
               ("Statement Period", f"{mdy(received - dt.timedelta(days=45))} - {mdy(received - dt.timedelta(days=15))}"),
               ("Beginning Balance", usd(bal)), ("Total Deposits", usd(period_pay * 2 * 0.72)), ("Ending Balance", usd(bal + period_pay * 0.4))])
    tx = [[mdy(received - dt.timedelta(days=44 - 3 * i)), d, usd(a)] for i, (d, a) in enumerate(
        [("Payroll Deposit - " + employer[:22], period_pay * .72), ("Mortgage/Rent Payment", -2450), ("Grocery - FreshMart", -182.4),
         ("Utility - Granite State", -146.2), ("Transfer to Savings", -1000), ("Card Purchase - Hardware", -96.15)])]
    p1.table(["Date", "Description", "Amount"], tx, [90, 300, 114])
    p2.title("Transaction Detail (continued)", 12)
    p2.table(["Date", "Description", "Amount"], [[mdy(received - dt.timedelta(days=26 - 2 * i)), d, usd(a)] for i, (d, a) in enumerate(
        [("Payroll Deposit - " + employer[:22], period_pay * .72), ("Insurance Premium", -212.0), ("Restaurant - Bistro 41", -88.3), ("ATM Withdrawal", -200)])], [90, 300, 114])
    p2.para("Member FDIC. Please examine this statement promptly and report any discrepancies within 30 days.")
    docs.append(_doc([p1, p2], "Lakeshore Community Bank - Statement", "bank_statement"))
    # Appraisal
    if "missing_appraisal" not in issues:
        p = Page(); p.bar("Uniform Residential Appraisal Report", f"File No. A-{rng.randint(10000, 99999)}  -  Fannie Mae Form 1004")
        p.fields([("Property Address", appr_addr), ("Borrower", name), ("Lender/Client", "Harborline Bank, N.A."), ("Lender Loan No.", loan_no),
                  ("Property Rights Appraised", "Fee Simple"), ("Year Built", str(rng.randint(1960, 2018))),
                  ("Gross Living Area", f"{rng.randint(1400, 3600):,} sq ft"), ("Effective Date of Appraisal", mdy(received - dt.timedelta(days=rng.randint(3, 9))))])
        p.title("Reconciliation")
        p.para(f"Based on the scope of work, the sales comparison approach and the three comparable sales analyzed, my (our) opinion "
               f"of the market value, as defined, of the real property that is the subject of this report is {usd(appr_value)}.")
        p.fields([("OPINION OF MARKET VALUE", usd(appr_value)), ("Appraiser", f"{rng.choice(FIRST)} {rng.choice(LAST)}, SRA - Lic. #{rng.randint(10000, 99999)}")])
        docs.append(_doc([p], "Uniform Residential Appraisal Report - Form 1004", "appraisal_report"))
    # Loan Estimate
    pi = le_amt * (rate / 1200) / (1 - (1 + rate / 1200) ** -360)
    p = Page(); p.bar("Loan Estimate", "Harborline Bank, N.A.  -  Save this Loan Estimate to compare with your Closing Disclosure.")
    p.fields([("Date Issued", mdy(sor["application_date"] + dt.timedelta(days=2))), ("Applicants", name), ("Property", addr_sor), ("Loan Term", "30 years"),
              ("Purpose", sor["loan_purpose"]), ("Product", "Fixed Rate"), ("Loan Type", "Conventional"), ("Loan ID #", loan_no)])
    p.title("Loan Terms")
    p.table(["Term", "Amount", "Can this amount increase after closing?"],
            [["Loan Amount", usd(le_amt), "NO"], ["Interest Rate", f"{rate:.3f} %", "NO"], ["Monthly Principal & Interest", usd(pi), "NO"]], [170, 130, 204])
    p.fields([("Estimated Closing Costs", usd(le_amt * 0.021)), ("Estimated Cash to Close", usd(max(value - le_amt, 0) + le_amt * 0.021))])
    docs.append(_doc([p], "Loan Estimate - CFPB Form H-24", "loan_estimate"))
    return sor, docs, "Harborline Mortgage Operations"

# ---------------- Investments ----------------
def investment_packet(rng, pid, received, issues):
    legal, variant, itype = rng.choice(ENTITIES)
    fund, fee, bank, aba, acct4 = rng.choice(FUNDS)
    commit = rng.choice([5, 7.5, 10, 15, 20, 25, 40, 50]) * 1_000_000.0
    call_no, pct = rng.randint(2, 7), rng.choice([0.05, 0.08, 0.10, 0.12])
    inv = f"INV-{rng.randint(300000, 399999)}"
    tin4 = f"{rng.randint(1000, 9999)}"
    due = received + dt.timedelta(days=10)
    sor = dict(investor_account=inv, investor_name=legal, investor_type=itype, fund_name=fund, commitment_amount=commit, management_fee_pct=fee,
               tax_id_last4=tin4, call_number=call_no, call_amount=commit * pct, call_due_date=due, wire_bank_name=bank,
               wire_routing_number=aba, wire_account_last4=acct4, relationship_manager=f"{rng.choice(FIRST)} {rng.choice(LAST)}")
    sub_name = variant if "investor_name_variant" in issues else legal
    doc_fee = fee + 0.25 if "fee_mismatch" in issues else fee
    doc_call = commit * (pct + 0.03) if "call_amount_mismatch" in issues else commit * pct
    w_bank, w_aba, w_acct = (("First Meridian Trust Co.", "121042882", f"{rng.randint(1000, 9999)}") if "wire_instructions_changed" in issues else (bank, aba, acct4))
    docs = []
    p = Page(); p.bar("Summary of Principal Terms", f"{fund}  -  Confidential Private Placement")
    p.fields([("Fund", fund), ("General Partner", fund.split(",")[0].replace("Fund", "GP").replace("Partners", "GP") + ", LLC"),
              ("Target Fund Size", usd(rng.choice([750, 1000, 1500]) * 1e6)), ("Minimum Commitment", "$5,000,000.00"),
              ("Management Fee", f"{doc_fee:.2f}% per annum of committed capital"), ("Carried Interest", "15% subject to an 8% preferred return"),
              ("Term", "8 years, with two 1-year extensions"), ("Investment Period", "4 years from the final closing")], 2)
    p.para("This summary is qualified in its entirety by the Limited Partnership Agreement and the Subscription Agreement of the Fund. "
           "In the event of any inconsistency, the Limited Partnership Agreement shall control.")
    docs.append(_doc([p], "Harborline Capital Partners - Term Summary", "term_sheet"))
    p1, p2 = Page(), Page()
    p1.bar("Subscription Agreement", f"{fund}")
    p1.para(f"The undersigned (the \"Investor\") hereby subscribes for a limited partnership interest in {fund} (the \"Fund\") "
            "and agrees to make capital contributions to the Fund in the aggregate amount of its Capital Commitment set forth below, "
            "upon the terms and subject to the conditions of the Limited Partnership Agreement.")
    p1.fields([("Name of Investor", sub_name), ("Investor Type", itype), ("Capital Commitment", usd(commit)), ("Investor Account No.", inv),
               ("Taxpayer Identification No.", f"XX-XXX{tin4}"), ("Jurisdiction of Organization", rng.choice(["Delaware", "New York", "Massachusetts", "Colorado"]))])
    p2.title("Investor Representations and Signature", 12)
    for rep in ["The Investor is an \"accredited investor\" within the meaning of Rule 501(a) of Regulation D.",
                "The Investor is a \"qualified purchaser\" as defined in Section 2(a)(51) of the Investment Company Act of 1940.",
                "The Investor is not a Prohibited Person under applicable anti-money laundering and sanctions laws."]:
        p2.para("- " + rep)
    p2.fields([("Authorized Signatory", f"/s/ {rng.choice(FIRST)} {rng.choice(LAST)}, Authorized Officer"), ("Date", mdy(received - dt.timedelta(days=rng.randint(60, 400))))])
    docs.append(_doc([p1, p2], "Harborline Capital Partners - Subscription Agreement", "subscription_agreement"))
    if "missing_kyc" not in issues:
        p = Page(); p.bar("Form W-9  Request for Taxpayer Identification Number and Certification", "Department of the Treasury - Internal Revenue Service")
        p.fields([("1  Name of entity/individual", legal), ("3a  Federal tax classification", itype),
                  ("5  Address", f"{rng.randint(100, 999)} {rng.choice(STREETS)[0]} St, {rng.choice(CITIES)[0]}"), ("Part I  Employer identification number", f"XX-XXX{tin4}")])
        p.para("Under penalties of perjury, I certify that the number shown on this form is my correct taxpayer identification number "
               "and I am not subject to backup withholding.")
        docs.append(_doc([p], "Form W-9 (Rev. March 2024)", "kyc_w9"))
    p = Page(); p.bar("Capital Call Notice", f"{fund}  -  Capital Call No. {call_no}")
    p.fields([("To (Limited Partner)", legal), ("Investor Account No.", inv), ("Capital Commitment", usd(commit)), ("Notice Date", mdy(received)),
              ("This Capital Call", usd(doc_call)), ("Due Date", mdy(due))])
    p.para("Pursuant to the Limited Partnership Agreement, the General Partner hereby calls capital from the Limited Partners for the "
           "purposes described below. Please remit the amount due by wire transfer no later than the Due Date.")
    p.table(["Purpose", "Amount"], [["Investments", usd(doc_call * 0.9)], ["Management fees", usd(doc_call * 0.08)], ["Fund expenses", usd(doc_call * 0.02)]], [354, 150])
    docs.append(_doc([p], "Harborline Capital Partners - Investor Services", "capital_call_notice"))
    p = Page(); p.bar("Wire Transfer Instructions", f"{fund}")
    p.fields([("Beneficiary", fund), ("Beneficiary Bank", w_bank), ("ABA Routing Number", w_aba), ("Account Number", f"XXXXXXXX{w_acct}"),
              ("Reference", f"{inv} / Call {call_no}"), ("SWIFT", "GRTBUS33" if w_bank == bank else "FMTCUS44")])
    p.para("Please confirm these instructions verbally with Investor Services at the phone number on file before sending any wire. "
           "Harborline Capital Partners will never change its wire instructions by email.")
    docs.append(_doc([p], "Harborline Capital Partners - Treasury", "wire_instructions"))
    return sor, docs, "Harborline Capital - Investor Services"

# ---------------- Claims ----------------
def claims_packet(rng, pid, received, issues):
    first, last = rng.choice(FIRST), rng.choice(LAST)
    name = f"{first} {last}"
    year, make, model = rng.choice(VEHICLES)
    vin = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(17))
    dol = received - dt.timedelta(days=rng.randint(3, 20))
    eff = dol - dt.timedelta(days=rng.randint(30, 300))
    exp = eff + dt.timedelta(days=365)
    if "loss_outside_policy" in issues:
        exp = dol - dt.timedelta(days=rng.randint(4, 25)); eff = exp - dt.timedelta(days=365)
    clm, pol = f"CLM-{rng.randint(40000000, 49999999)}", f"PA-{rng.randint(1000000, 1999999)}"
    city, state, zp = rng.choice(CITIES)
    lines = [["Front bumper cover - replace", "Part", usd(rng.uniform(450, 1100))], ["Headlamp assembly LH", "Part", usd(rng.uniform(380, 1400))],
             ["Hood panel - repair", "Labor 6.5 hr", usd(6.5 * 68)], ["Refinish hood & bumper", "Paint 5.0 hr", usd(5 * 72)],
             ["Radiator support - replace", "Part", usd(rng.uniform(300, 900))], ["Frame pull / measure", "Labor 3.0 hr", usd(3 * 95)]]
    est = sum(float(x[2].strip("$").replace(",", "")) for x in lines) * rng.uniform(1.0, 2.4)
    reserve = round(est * rng.uniform(1.0, 1.015), 2)
    if "estimate_over_reserve" in issues: reserve = round(est / rng.uniform(1.3, 1.45), 2)
    med = rng.uniform(600, 4800) if rng.random() < 0.5 else 0.0
    sor = dict(claim_number=clm, policy_number=pol, insured_name=name, policy_effective_date=eff, policy_expiration_date=exp, date_of_loss=dol,
               vehicle_vin=vin, vehicle_desc=f"{year} {make} {model}", coverage_limit=float(rng.choice([50000, 75000, 100000])),
               deductible=float(rng.choice([500, 1000])), reserve_amount=reserve, adjuster=f"{rng.choice(FIRST)} {rng.choice(LAST)}", claim_status="Open")
    est_vin = vin[:12] + "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(5)) if "vin_mismatch" in issues else vin
    police_date = dol + dt.timedelta(days=rng.randint(2, 5)) if "loss_date_conflict" in issues else dol
    pol_name = f"{NICK[first]} {last}" if "insured_nickname" in issues else name
    loc = f"{rng.choice(['Intersection of', 'Near'])} {rng.choice(STREETS)[0]} {rng.choice(STREETS)[1]} and Route {rng.randint(2, 99)}, {city}, {state}"
    docs = []
    p = Page(); p.bar("First Notice of Loss - Auto Claim", "Harborline Mutual Insurance Company  -  Claims Intake")
    p.fields([("Claim Number", clm), ("Policy Number", pol), ("Named Insured", name), ("Contact Phone", f"({rng.randint(201, 989)}) 555-{rng.randint(1000, 9999)}"),
              ("Date of Loss", mdy(dol)), ("Time of Loss", f"{rng.randint(6, 21)}:{rng.choice(['05', '20', '35', '50'])}"),
              ("Vehicle", f"{year} {make} {model}"), ("VIN", vin)])
    p.title("Loss Description")
    p.para(f"Loss location: {loc}. Insured reports that while proceeding through the intersection, a third-party vehicle failed to yield "
           f"and struck the front of the insured vehicle. Vehicle was towed from the scene. Police responded. Injuries reported: {'YES' if med else 'NO'}.")
    docs.append(_doc([p], "Harborline Mutual - FNOL Form AU-101", "fnol_claim_form"))
    p = Page(); p.bar("Traffic Collision Report", f"{city} Police Department  -  Records Division")
    p.fields([("Report Number", f"{rng.randint(24, 26)}-{rng.randint(100000, 999999)}"), ("Date of Incident", mdy(police_date)),
              ("Location", loc[:60]), ("Reporting Officer", f"Ofc. {rng.choice(LAST)} #{rng.randint(100, 999)}"),
              ("Unit 1 Driver", name), ("Unit 1 Vehicle / VIN", f"{year} {make} {model} / {vin}")])
    p.title("Narrative")
    p.para("Unit 2 failed to yield the right of way and collided with Unit 1. Unit 2 driver was cited for failure to yield. "
           "Both vehicles sustained front-end damage. Weather clear, road surface dry.")
    docs.append(_doc([p], f"{city} PD - Form TCR-1", "police_report"))
    p = Page(); p.bar("Collision Repair Estimate", f"{rng.choice(SHOPS)}  -  I-CAR Gold Class")
    p.fields([("Claim Number", clm), ("Insured / Owner", name), ("Vehicle", f"{year} {make} {model}"), ("VIN", est_vin), ("Estimate Date", mdy(dol + dt.timedelta(days=2)))])
    p.table(["Operation", "Type", "Amount"], lines, [264, 120, 120])
    p.fields([("Supplements & Sublet", usd(est - sum(float(x[2].strip('$').replace(',', '')) for x in lines))), ("GRAND TOTAL ESTIMATE", usd(est))])
    docs.append(_doc([p], "Estimate prepared with CCC-style estimating", "repair_estimate"))
    if med:
        p = Page(); p.bar(rng.choice(PROVIDERS), "Itemized Statement of Charges")
        p.fields([("Patient", name), ("Date of Service", mdy(dol + dt.timedelta(days=1))), ("Insurance Claim Ref.", clm), ("Account No.", f"MRN-{rng.randint(100000, 999999)}")])
        p.table(["CPT", "Description", "Charge"], [["99204", "New patient visit, moderate complexity", usd(med * 0.35)], ["72040", "X-ray cervical spine, 2-3 views", usd(med * 0.25)],
                                                    ["97110", "Therapeutic exercise, 15 min x4", usd(med * 0.40)]], [80, 300, 124])
        p.fields([("TOTAL CHARGES", usd(med)), ("Balance Due", usd(med))])
        docs.append(_doc([p], "Itemized Statement - UB/HCFA", "medical_bill"))
    if "missing_proof_of_loss" not in issues:
        p = Page(); p.bar("Sworn Statement in Proof of Loss", "Harborline Mutual Insurance Company")
        p.fields([("Policy Number", pol), ("Claim Number", clm), ("Insured", pol_name), ("Date of Loss", mdy(dol)),
                  ("Amount Claimed", usd(est + med)), ("Less Deductible", usd(sor["deductible"]))])
        p.para("The said loss did not originate by any act, design or procurement on the part of your insured. "
               "The undersigned swears that the foregoing statements are true and correct.")
        p.fields([("Signature of Insured", f"/s/ {pol_name}"), ("Date Signed", mdy(received - dt.timedelta(days=1)))])
        docs.append(_doc([p], "Harborline Mutual - Form POL-7", "proof_of_loss"))
    return sor, docs, "Harborline Mutual - Claims Intake"

BUILDERS = {"Lending": lending_packet, "Investments": investment_packet, "Claims": claims_packet}

# Which problems to plant in the k-th packet of each line of business. The first 3 per line of business (the default
# 9-packet backfill) already cover all three HOLD cases, real REVIEW issues, look-alikes that should auto-clear and clean
# packets; a bigger backfill adds the remaining scenarios.
SCHEDULE = {
    "Lending":     [[], ["income_overstated", "borrower_nickname"], ["appraisal_low", "address_abbreviation"],
                    ["loan_amount_transposed"], ["missing_appraisal"], [], []],
    "Investments": [["wire_instructions_changed"], [], ["fee_mismatch", "investor_name_variant"],
                    ["call_amount_mismatch"], ["missing_kyc"], ["investor_name_variant"], []],
    "Claims":      [["vin_mismatch"], ["insured_nickname"], ["loss_outside_policy", "missing_proof_of_loss"],
                    ["estimate_over_reserve"], ["loss_date_conflict"], [], []],
}

def generate_packet(packet_id, received, out_dir, family, k):
    """Builds one packet PDF (the k-th of its line of business). Returns (system-of-record row, answer-key pages, answer-key issues)."""
    rng = random.Random(packet_id)
    issues = SCHEDULE[family][k % len(SCHEDULE[family])]
    sor, docs, dept = BUILDERS[family](rng, packet_id, received, issues)
    rng.shuffle(docs)                                   # real packets arrive in any order
    pages = [pg for d in docs for pg in d]
    if rng.random() < 0.35:
        pages = fax_cover(rng, dept, f"{rng.choice(['Title', 'Broker', 'Agency', 'Counsel'])} Office - {rng.choice(LAST)} & Associates", len(pages), received) + pages
    os.makedirs(out_dir, exist_ok=True)
    write_pdf([p for _, p in pages], f"{out_dir}/{packet_id}.pdf")
    key_pages, doc_idx, prev = [], 0, None
    for i, (dtyp, p) in enumerate(pages, 1):
        first_page = any("Page 1 of" in op for op in p.ops) or dtyp == "other"
        doc_idx += 1 if first_page else 0
        key_pages.append((packet_id, i, dtyp, doc_idx))
    key_issues = [(packet_id, family, iss, *ISSUES[family][iss]) for iss in issues]
    return sor, key_pages, key_issues

FAMILIES = ["Lending", "Investments", "Claims"]

def land_packets(days, packets_per_day, seq):
    """Lands packets for each day into the inbox volume and appends the matching system-of-record and answer-key rows.
    Used by step 0.5 (the backfill archive) and by the Job's land_new_packets task (today's arrivals). Days that already have packets are skipped."""
    sor_rows = {f: [] for f in FAMILIES}
    key_pages, key_issues, landed = [], [], []
    for day in sorted(days):
        folder = f"{INBOX}/{day.isoformat()}"
        if os.path.isdir(folder) and os.listdir(folder):
            print(f"{day}: packets already landed - nothing to generate")
            continue
        for n in range(packets_per_day):
            pid = f"PKT-{day:%Y%m%d}-{n + 1:03d}"
            family, k = FAMILIES[seq % 3], seq // 3
            seq += 1
            sor, kp, ki = generate_packet(pid, day, folder, family, k)
            sor_rows[family].append(sor); key_pages += kp; key_issues += ki
            landed.append((pid, str(day), family, len(kp), ", ".join(i[2] for i in ki) or "-"))
    for family, table in [("Lending", "sor_loans"), ("Investments", "sor_investor_commitments"), ("Claims", "sor_claims")]:
        if sor_rows[family]:
            schema = spark.table(fq(table)).schema
            spark.createDataFrame([tuple(r[c] for c in schema.fieldNames()) for r in sor_rows[family]], schema).write.mode("append").saveAsTable(fq(table))
    for rows, table in [(key_pages, "answer_key_pages"), (key_issues, "answer_key_issues")]:
        if rows:
            spark.createDataFrame(rows, spark.table(fq(table)).schema).write.mode("append").saveAsTable(fq(table))
    return landed

# COMMAND ----------

# Backfill = start over: drop this notebook's tables (only the ones it creates) and empty the inbox, then rebuild below.
# Keep = leave everything as it is (the Job only processes new packets anyway).
DEMO_TABLES = list(TABLE_INFO) + ["sor_values", "recon_rules", "required_documents", "answer_key_pages", "answer_key_issues"]
if MODE == "backfill":
    for j in w.jobs.list(name=JOB_NAME):   # the Job's AI Function tasks own their output tables: remove the Job first
        w.jobs.delete(j.job_id)
    for t in DEMO_TABLES:
        spark.sql(f"DROP TABLE IF EXISTS {fq(t)}")
    for p in (INBOX, PAGE_IMAGES):
        for f in dbutils.fs.ls(p):
            dbutils.fs.rm(f.path, True)
    print(f"Backfill mode: removed the Job, reset {len(DEMO_TABLES)} tables and emptied the inbox and page-image volumes.")
else:
    print("Keep mode: existing tables and documents are left as they are.")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Systems of record: what the bank / fund administrator / insurer *believes* is true. Documents get reconciled against these.
# MAGIC CREATE TABLE IF NOT EXISTS sor_loans (
# MAGIC   loan_number STRING COMMENT 'Loan number, e.g. HL-2xxxxxx', borrower_name STRING, property_address STRING,
# MAGIC   loan_amount DOUBLE, interest_rate DOUBLE COMMENT 'Note rate in percent', term_months INT, loan_purpose STRING,
# MAGIC   employer_name STRING, annual_income DOUBLE COMMENT 'Stated gross annual base income (USD)',
# MAGIC   appraised_value DOUBLE, application_date DATE, loan_officer STRING, loan_status STRING
# MAGIC ) COMMENT 'Loan origination system (LOS): mortgage applications in underwriting';
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS sor_investor_commitments (
# MAGIC   investor_account STRING COMMENT 'Investor account number, e.g. INV-3xxxxx', investor_name STRING, investor_type STRING,
# MAGIC   fund_name STRING, commitment_amount DOUBLE, management_fee_pct DOUBLE, tax_id_last4 STRING,
# MAGIC   call_number INT, call_amount DOUBLE COMMENT 'Amount due on the current capital call (USD)', call_due_date DATE,
# MAGIC   wire_bank_name STRING, wire_routing_number STRING, wire_account_last4 STRING COMMENT 'Fund bank account on file (last 4)',
# MAGIC   relationship_manager STRING
# MAGIC ) COMMENT 'Fund administration system: LP commitments, current capital call and the wire instructions on file';
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS sor_claims (
# MAGIC   claim_number STRING COMMENT 'Claim number, e.g. CLM-4xxxxxxx', policy_number STRING, insured_name STRING,
# MAGIC   policy_effective_date DATE, policy_expiration_date DATE, date_of_loss DATE, vehicle_vin STRING, vehicle_desc STRING,
# MAGIC   coverage_limit DOUBLE, deductible DOUBLE, reserve_amount DOUBLE COMMENT 'Current loss reserve (USD)', adjuster STRING, claim_status STRING
# MAGIC ) COMMENT 'Claims administration system: open auto claims';
# MAGIC
# MAGIC -- Answer key (from the generator) — used only to score the AI in steps 4.3 and 8.5. Real data has no answer key.
# MAGIC CREATE TABLE IF NOT EXISTS answer_key_pages (packet_id STRING, page_number INT, doc_type STRING, doc_seq INT)
# MAGIC   COMMENT 'Synthetic-data answer key: true document type and document number of every page';
# MAGIC CREATE TABLE IF NOT EXISTS answer_key_issues (packet_id STRING, line_of_business STRING, issue STRING, doc_type STRING, field_name STRING, expected_action STRING)
# MAGIC   COMMENT 'Synthetic-data answer key: planted problems and the action a good reviewer would take';

# COMMAND ----------

import datetime as dt
# The archive: the last N business days of packets (land_packets is defined in step 0.2; step 6.1 reuses it for "today")
today = dt.date.today()
days, d = [], today - dt.timedelta(days=1)
while len(days) < BACKFILL_DAYS:
    if d.weekday() < 5:
        days.append(d)
    d -= dt.timedelta(days=1)
landed = land_packets(days, PACKETS_PER_DAY, seq=0) if MODE == "backfill" else []
refresh_inbox_view()
print(f"Landed {len(landed)} packets ({sum(p[3] for p in landed)} pages) into {INBOX}")
if landed:
    display(spark.createDataFrame(landed, "packet_id string, arrival_date string, line_of_business string, pages int, planted_issues string"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The inbox is just files in a Unity Catalog volume — read_files lists them (it never opens a PDF here)
# MAGIC SELECT received_date AS arrival_date, count(*) AS packets, round(sum(length) / 1024) AS total_kb,
# MAGIC        array_join(array_sort(collect_list(packet_id)), ', ') AS packet_ids
# MAGIC FROM inbox_files
# MAGIC GROUP BY ALL
# MAGIC ORDER BY arrival_date;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.7 Your source data in Unity Catalog
# MAGIC Every table **and every document** is governed by **Unity Catalog** — permissions, lineage and audit come for free, for structured tables and for the PDF files in the volume. Click a card to see the schema, sample data or lineage; the grey button opens the inbox volume so you can open any PDF.

# COMMAND ----------

show_uc_cards(roles={"Source"})

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.8 Meet your Genie agent
# MAGIC Sections 0, 4, 5, 7 and 8 end with **💬 Ask Genie**: the same insight, asked in plain English instead of SQL. That's how underwriters, investor-services analysts and claims supervisors will use these results — no SQL, no waiting on the data team, and every answer respects **Unity Catalog** permissions.
# MAGIC
# MAGIC Step 0.8 creates (or reuses) the **Document Intelligence Genie** agent over this schema. Tables that the Job and later steps will build start as empty placeholders and fill in as you go.

# COMMAND ----------

# Create (or reuse) the Genie agent now, so you can ask it questions after every step
TABLE_COMMENTS = {
    "sor_loans": "Loan origination system: one row per mortgage loan in underwriting (loan_number is the key).",
    "sor_investor_commitments": "Fund administration system: one row per investor commitment with the current capital call and the fund wire instructions on file (investor_account is the key).",
    "sor_claims": "Claims system: one row per open auto claim with policy period, VIN and reserve (claim_number is the key).",
    "packet_documents": "Each packet split into documents: doc_type, doc_family (Lending / Investments / Claims / Other), page range and text.",
    "extracted_fields": "ai_extract output, one row per document field: field_name, field_value, confidence (0-1).",
    "packet_index": "One row per packet: received_date, line_of_business and record_key (loan_number / investor_account / claim_number).",
    "recon_results": "Every document field reconciled against the system of record: field_value vs system_value, status (MATCH / MISMATCH / MISSING_DOCUMENT / NOT_ON_DOCUMENT / NO_SYSTEM_RECORD), pct_diff for amounts.",
    "exceptions": "Every non-matching recon result with the ai_decide verdict: action (auto_clear / analyst_review / hold), fraud_probability (0-1), materiality_score (0-3).",
    "packet_status": "One row per packet: packet_decision (STRAIGHT_THROUGH / REVIEW / HOLD), counts of exceptions, open_reasons and exposure_usd.",
}
COLUMN_COMMENTS = {
    ("packet_status", "packet_decision"): "STRAIGHT_THROUGH = no human touch needed; REVIEW = analyst must look; HOLD = stop funding / wire / payment",
    ("packet_status", "exposure_usd"): "Loan amount (Lending), capital call amount (Investments) or claim reserve (Claims)",
    ("exceptions", "action"): "ai_decide verdict: auto_clear, analyst_review or hold",
    ("exceptions", "fraud_probability"): "ai_decide probability (0-1) that the discrepancy is a fraud / tampering signal",
    ("recon_results", "pct_diff"): "For amounts: (document - system) / system * 100",
}
GENIE_TABLES = sorted(TABLE_COMMENTS)
_q = lambda s: s.replace("'", "''")

def describe_tables():
    """(Re)applies business descriptions — CREATE OR REPLACE resets them, and Genie relies on them."""
    for t, c in TABLE_COMMENTS.items():
        cols = spark.table(fq(t)).columns
        if cols == ["placeholder"]:
            continue
        spark.sql(f"COMMENT ON TABLE {fq(t)} IS '{_q(c)}'")
        for (tt, col), cc in COLUMN_COMMENTS.items():
            if tt == t and col in cols:
                spark.sql(f"ALTER TABLE {fq(t)} ALTER COLUMN `{col}` COMMENT '{_q(cc)}'")

# Genie needs every table to exist when the agent is created: create empty placeholders for the
# tables the next steps will build (each step's CREATE OR REPLACE fills them in).
for t in GENIE_TABLES:
    spark.sql(f"CREATE TABLE IF NOT EXISTS {fq(t)} (placeholder STRING) COMMENT 'Populated by a later step of this notebook'")
describe_tables()

hexid = lambda: uuid.uuid4().hex
n = lambda t: f"{CATALOG}.{SCHEMA}.{t}"
instructions = f"""This agent answers questions about loan, investment and insurance-claim document packets processed with AI for Harborline Financial Group.
- Packet outcomes: use {n('packet_status')}. packet_decision is STRAIGHT_THROUGH (no human touch), REVIEW or HOLD. "Straight-through rate" = share of packets with STRAIGHT_THROUGH. line_of_business is Lending, Investments or Claims.
- Exceptions and reasons: use {n('exceptions')}. action is auto_clear, analyst_review or hold; auto_clear exceptions are noise and are not open issues. fraud_probability > 0.5 = fraud signal.
- Field-level comparisons: use {n('recon_results')} (field_value from the document vs system_value from the system of record; pct_diff for amounts).
- Extracted values: use {n('extracted_fields')} joined to {n('packet_index')} on packet_id; doc_type examples: loan_application, w2, pay_stub, bank_statement, appraisal_report, loan_estimate, term_sheet, subscription_agreement, kyc_w9, capital_call_notice, wire_instructions, fnol_claim_form, police_report, repair_estimate, medical_bill, proof_of_loss.
- Systems of record: {n('sor_loans')}, {n('sor_investor_commitments')}, {n('sor_claims')}; join to packet_index.record_key.
- Always state the date range used and round money to whole dollars."""
serialized = {
    "version": 2,
    "config": {"sample_questions": sorted([{"id": hexid(), "question": [x]} for x in GENIE_QUESTIONS], key=lambda x: x["id"])},
    "data_sources": {"tables": [{"identifier": n(t)} for t in GENIE_TABLES]},
    "instructions": {
        "text_instructions": [{"id": hexid(), "content": [instructions]}],
        "example_question_sqls": sorted([
            {"id": hexid(), "question": ["Which packets are on hold and why?"],
             "sql": [f"SELECT packet_id, received_date, line_of_business, record_key, exposure_usd, open_reasons\\nFROM {n('packet_status')}\\n"
                     f"WHERE packet_decision = 'HOLD'\\nORDER BY exposure_usd DESC"]},
            {"id": hexid(), "question": ["Straight-through rate by line of business"],
             "sql": [f"SELECT line_of_business, count(*) AS packets, round(100 * avg(CASE WHEN packet_decision = 'STRAIGHT_THROUGH' THEN 1 ELSE 0 END), 1) AS straight_through_pct\\n"
                     f"FROM {n('packet_status')}\\nGROUP BY line_of_business ORDER BY line_of_business"]},
        ], key=lambda x: x["id"]),
    },
}
GENIE_TITLE = f"Loan, Investment & Claims Document Genie ({SCHEMA})"
existing = next((s for s in _api("GET", "/api/2.0/genie/spaces").get("spaces", []) if s.get("title") == GENIE_TITLE), None)
if existing:
    GENIE_SPACE_ID = existing["space_id"]
    print(f"Reusing Genie agent '{GENIE_TITLE}'")
    if _api("GET", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}").get("warehouse_id") != WAREHOUSE.id:   # keep it on the warehouse chosen in S.3
        _api("PATCH", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}", body={"warehouse_id": WAREHOUSE.id})
        print(f"Moved the Genie agent to SQL warehouse '{WAREHOUSE.name}'")
else:
    GENIE_SPACE_ID = _api("POST", "/api/2.0/genie/spaces", body={
        "title": GENIE_TITLE,
        "description": "Ask about document packets, extracted loan / deal / claim fields, reconciliation mismatches, holds and straight-through rates — built on Databricks AI Functions outputs.",
        "warehouse_id": WAREHOUSE.id,
        "parent_path": f"/Workspace/Users/{w.current_user.me().user_name}",
        "serialized_space": json.dumps(serialized),
    })["space_id"]
    print(f"Created Genie agent '{GENIE_TITLE}'")
GENIE_URL = f"{HOST}/genie/rooms/{GENIE_SPACE_ID}{ORG}"

def ask(question, before=None):
    """Ask the Genie agent a question in plain English and show the answer here."""
    describe_tables()
    show_genie_answer(ask_genie(question), before=before)

displayHTML(button("Open the Genie agent", GENIE_URL) + button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.9 💬 Ask Genie — explore the systems of record
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("How much money is in flight: total loan amount in underwriting, total capital called, and total open claim reserves?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.10 🛠 Try it yourself
# MAGIC Click **Open the document inbox (volume)** in step 0.7, open a date folder and **preview one of the PDFs**. Count the documents in it and note the order they're in — the classification and `split_documents` tasks will have to work that out on their own.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.11 Bring your own documents
# MAGIC Every downstream cell reads only the **inbox volume** and the three **system-of-record** tables. To run on real data, skip the generator cells (0.1–0.5), drop your PDFs (or TIFF / PNG / DOCX) into `inbox/<yyyy-mm-dd>/`, and point the system-of-record names at your own tables with views:
# MAGIC
# MAGIC | Table | Required columns | Typical source |
# MAGIC |---|---|---|
# MAGIC | `sor_loans` | `loan_number` + the fields you reconcile (amount, rate, borrower, income, appraised value…) | LOS — Encompass, Black Knight Empower, MeridianLink, nCino |
# MAGIC | `sor_investor_commitments` | `investor_account` + commitment, call amount, due date, fee, wire bank / routing / account | Fund admin / investor CRM — SS&C Geneva, Investran, Allvue, Salesforce FSC |
# MAGIC | `sor_claims` | `claim_number` + policy number & period, insured, VIN, reserve, limits | Claims — Guidewire ClaimCenter, Duck Creek, Majesco |
# MAGIC
# MAGIC ```sql
# MAGIC -- Example: point the notebook at your LOS data
# MAGIC CREATE OR REPLACE VIEW sor_loans AS
# MAGIC SELECT loan_id AS loan_number, concat_ws(' ', b_first, b_middle, b_last) AS borrower_name, subject_property_address AS property_address,
# MAGIC        base_loan_amount AS loan_amount, note_rate AS interest_rate, loan_term AS term_months, purpose AS loan_purpose,
# MAGIC        employer AS employer_name, base_income_monthly * 12 AS annual_income, appraised_value, application_date,
# MAGIC        loan_officer_name AS loan_officer, milestone AS loan_status
# MAGIC FROM my_catalog.lending.los_loans_current;
# MAGIC ```
# MAGIC Then edit the document labels (step 1.5), the field schemas (step 1.7) and the rules (step 7.1) to match your forms, and re-run step 2.1 to redeploy the Job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Design the document intelligence workflow — a Lakeflow Job of native AI Function tasks
# MAGIC
# MAGIC Lakeflow Jobs has **native AI Function task types** — *Document Parsing* (`ai_parse_document`), *Classification* (`ai_classify`) and *Information Extraction* (`ai_extract`), under **Document intelligence** in the Jobs UI. Each one runs its function over **every file in a volume** or **every row of a table**, writes the results to an **output table it creates and manages**, and on every run processes **only the inputs that are new or changed**. No notebooks, no code to run them.
# MAGIC
# MAGIC This workflow chains **8 AI Function tasks** with **5 small SQL tasks** (on the SQL warehouse) that move data between them:
# MAGIC
# MAGIC ```
# MAGIC register_new_files ─► parse_documents ─► prepare_pages ─► classify_line_of_business ─► route_pages ─┬─► classify_lending_pages ────┐                  ┌─► extract_lending ──────┐
# MAGIC SQL: new PDFs in       ai_parse_document   SQL: 1 row       ai_classify: Lending /       SQL: send     ├─► classify_investment_pages ─┼─► split_documents ┼─► extract_investments ──┼─► assemble_extractions
# MAGIC the inbox volume ->    (every new PDF)     per page         Investments / Claims /       each page to  └─► classify_claim_pages ──────┘   SQL: split       └─► extract_claims ───────┘   SQL: one row per
# MAGIC inbox_documents                                             Other (per page)             its LOB                ai_classify: doc type     packets into docs      ai_extract                  field + packet key
# MAGIC ```
# MAGIC
# MAGIC | Design choice | Why |
# MAGIC |---|---|
# MAGIC | **AI Function tasks** for every AI step | The platform handles batching, retries, incremental processing (new / changed / deleted inputs) and row-level success / failure counts |
# MAGIC | **Two-level classification** — line of business, then document type | An `ai_classify` task takes **2–10 labels**; there are 16 document types. Routing first keeps every classifier small and accurate |
# MAGIC | **Fan-out** per line of business | The three classification and three extraction tasks run **in parallel**, each with its own labels / schema and output table |
# MAGIC | **SQL tasks** between AI tasks | Register new files and reshape outputs into the next task's input (pages, routed pages, split documents) — append-only tables with **change data feed**, so the next AI task only sees new rows |
# MAGIC | **One definition, two uses** | The same DAG is the **backfill** (step 2.2) and the **daily job** (section 6) |
# MAGIC
# MAGIC Steps **1.1–1.9** define the tasks. Each **AI Function task** is a few lines of configuration (input, output table, function options); each **SQL task** is its SQL. Nothing runs yet: step 2.1 turns them into the Job.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Tasks `register_new_files` → `parse_documents` — SQL task, then AI Function task · `ai_parse_document`
# MAGIC **1.1a** `register_new_files` appends every PDF that has landed in the inbox volume since the last run (all date folders) to `inbox_documents` — file name, arrival date and the file's **content** as BINARY — with **change data feed** on.
# MAGIC
# MAGIC **1.1b** `parse_documents` runs `ai_parse_document` on the content of every **new** row. Output: `parsed_packets`, one row per packet with the full parse (`response`: text, tables as HTML, layout, bounding boxes) and an `error` column. Page images are rendered into the `page_images` volume for the viewer in section 3.
# MAGIC
# MAGIC > 💡 An `ai_parse_document` task can also read a **volume** directly. It then discovers new files with **file events**, which a workspace admin has to enable on the volume's external location — otherwise the first run spends many minutes retrying before it falls back to listing the folder. Registering files in a table keeps this notebook fast in any workspace.

# COMMAND ----------

define_sql_task(
    "register_new_files",
    title="Register every new PDF in the inbox volume (name, arrival date, content)",
    depends_on=[],
    sql=f"""-- SQL task register_new_files: append every PDF that landed in the inbox since the last run.
-- Change data feed on inbox_documents lets parse_documents pick up only the new files.
CREATE TABLE IF NOT EXISTS inbox_documents (
  packet_id STRING COMMENT 'File name without .pdf', path STRING, received_date DATE COMMENT 'Arrival date (the inbox date folder)',
  content BINARY COMMENT 'The PDF', registered_at TIMESTAMP
) COMMENT 'Every PDF that has landed in the inbox volume (input to parse_documents)'
  TBLPROPERTIES (delta.enableChangeDataFeed = true);

INSERT INTO inbox_documents
SELECT packet_id, path, received_date, content, current_timestamp() AS registered_at
FROM (
  SELECT regexp_extract(path, '([^/]+)\\\\.pdf$', 1) AS packet_id, path,
         to_date(regexp_extract(path, '/inbox/(\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}})/', 1)) AS received_date, content
  FROM read_files('{INBOX}/', format => 'binaryFile', pathGlobFilter => '*.pdf', recursiveFileLookup => 'true')
)
WHERE packet_id NOT IN (SELECT packet_id FROM inbox_documents);""",
)

# COMMAND ----------

define_ai_task(
    "parse_documents",
    title="Parse every new PDF: text, tables, layout and an image of every page",
    function="ai_parse_document",
    depends_on=["register_new_files"],
    input={"delta_table": {"table_name": f"{CATALOG}.{SCHEMA}.inbox_documents", "input_column": "content", "merge_key_columns": ["packet_id"]}},
    output_table="parsed_packets",
    options={"version": "2.0", "image_output_path": PAGE_IMAGES + "/"},   # no description_element_types = no AI captions (cheaper)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Task `prepare_pages` — SQL task
# MAGIC Flattens each new parse into **one row per page** (`packet_pages`): the page's text in reading order, its image and its page number. Classification works page by page — that's what lets the workflow **split** a 10-page packet into its documents later. The table has **change data feed** on, so the next AI task only classifies the new pages.

# COMMAND ----------

define_sql_task(
    "prepare_pages",
    title="One row per page from the ai_parse_document output (new packets only)",
    depends_on=["parse_documents"],
    sql=r"""
-- SQL task prepare_pages: one row per page from the ai_parse_document output — only packets not seen before.
-- Change data feed on packet_pages lets the next AI Function task pick up only the new pages.
CREATE TABLE IF NOT EXISTS packet_pages (
  page_id STRING, packet_id STRING, path STRING, received_date DATE, page_number INT, image_uri STRING,
  page_text STRING COMMENT 'All elements on the page in reading order (tables kept as HTML)', elements INT, tables INT
) COMMENT 'One row per page: text and page image (input to classify_line_of_business)'
  TBLPROPERTIES (delta.enableChangeDataFeed = true);

INSERT INTO packet_pages
WITH new_packets AS (   -- parsed packets that aren't in packet_pages yet
  SELECT x.packet_id, d.path, d.received_date, x.response
  FROM parsed_packets x
  JOIN inbox_documents d ON d.packet_id = x.packet_id
  WHERE x.response IS NOT NULL AND nullif(x.error::STRING, '') IS NULL
    AND x.packet_id NOT IN (SELECT packet_id FROM packet_pages)
),
elements AS (
  SELECT p.packet_id, e.pos,
         e.value:bbox[0]:page_id::INT AS page_id,
         e.value:type::STRING         AS element_type,   -- text, table, title, section_header, page_footer, page_number ...
         e.value:content::STRING      AS content
  FROM new_packets p, LATERAL variant_explode(p.response:document:elements) AS e
),
pages AS (
  SELECT p.packet_id, p.path, p.received_date, pg.value:id::INT AS page_id, pg.value:image_uri::STRING AS image_uri
  FROM new_packets p, LATERAL variant_explode(p.response:document:pages) AS pg
),
page_text AS (
  SELECT packet_id, page_id,
         array_join(transform(array_sort(collect_list(struct(pos, content))), x -> x.content), '\n') AS page_text,
         count(*)                    AS elements,
         count_if(element_type = 'table') AS tables
  FROM elements
  GROUP BY ALL
)
SELECT pg.packet_id || '-P' || lpad(CAST(pg.page_id + 1 AS STRING), 2, '0') AS page_id,
       pg.packet_id, pg.path, pg.received_date, pg.page_id + 1 AS page_number, pg.image_uri,
       coalesce(t.page_text, '') AS page_text, coalesce(t.elements, 0) AS elements, coalesce(t.tables, 0) AS tables
FROM pages pg
LEFT JOIN page_text t ON t.packet_id = pg.packet_id AND t.page_id = pg.page_id;
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Task `classify_line_of_business` — AI Function task · `ai_classify`
# MAGIC First-level routing: is each page part of a **loan file**, a **fund package** or a **claim** (or a fax cover)? Each label carries a short **description** — that's how you teach the classifier your taxonomy in plain English, no training data needed. Confidence scores come back with every label.

# COMMAND ----------

LOB_LABELS = [
    {"name": "Lending",     "description": "Mortgage loan file. Includes the loan application (Form 1003), Loan Estimate and appraisal AND every borrower income or asset document: IRS Form W-2, employer pay stubs / earnings statements / direct deposit advice, and personal bank account statements"},
    {"name": "Investments", "description": "Private fund investor documents: summary of principal terms / term sheet, subscription agreement, IRS Form W-9 for an investing entity, capital call notice, wire transfer instructions"},
    {"name": "Claims",      "description": "Auto insurance claim documents: first notice of loss, police / traffic collision report, collision repair estimate, medical provider bill, sworn proof of loss"},
    {"name": "Other",       "description": "ONLY fax cover sheets, transmittal cover letters and blank pages. Never use Other for a form, statement, pay stub, bank statement or any financial document"},
]
define_ai_task(
    "classify_line_of_business",
    title="Classify every new page by line of business: Lending, Investments, Claims or Other",
    function="ai_classify",
    depends_on=["prepare_pages"],
    input={"delta_table": {"table_name": f"{CATALOG}.{SCHEMA}.packet_pages", "input_column": "page_text", "merge_key_columns": ["page_id"]}},
    output_table="page_line_of_business",
    options={"version": "2.1", "enable_confidence_scores": True, "inline_labels": {"labels": LOB_LABELS},
             "instructions": "Each input is ONE page from a packet received by a bank (mortgage loan files), a fund administrator (investor documents) or an insurer (auto claims). "
                             "Assign the line of business whose packet this page belongs to. Income and asset documents (W-2, pay stubs, bank statements) are part of a mortgage loan file. "
                             "Continuation pages (Page 2 of N) belong to the same form named in their footer."},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Task `route_pages` — SQL task
# MAGIC Appends each newly classified page to the input table of its line of business — `pages_lending`, `pages_investments` or `pages_claims` (fax covers and other pages skip the second classifier).

# COMMAND ----------

LINES_OF_BUSINESS = ["Lending", "Investments", "Claims"]
route_sql = "-- SQL task route_pages: send each new page to the document-type classifier of its line of business.\n"
for lob in LINES_OF_BUSINESS:
    t = f"pages_{lob.lower()}"
    route_sql += f"""
CREATE TABLE IF NOT EXISTS {t} (page_id STRING, packet_id STRING, page_number INT, page_text STRING)
  COMMENT 'Pages classified as {lob} (input to the {lob} document-type classifier)' TBLPROPERTIES (delta.enableChangeDataFeed = true);
INSERT INTO {t}
SELECT p.page_id, p.packet_id, p.page_number, p.page_text
FROM packet_pages p
JOIN page_line_of_business c ON c.page_id = p.page_id
WHERE c.response:response[0]:value::STRING = '{lob}'
  AND p.page_id NOT IN (SELECT page_id FROM {t});
"""
define_sql_task("route_pages", title="Route each new page to its line of business (Other pages go straight to the splitter)",
                depends_on=["classify_line_of_business"], sql=route_sql)
print(route_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Tasks `classify_lending_pages` · `classify_investment_pages` · `classify_claim_pages` — AI Function tasks · `ai_classify`, fanned out
# MAGIC Second-level classification, **in parallel**: each line of business has its own classifier with **its own document types** (6 or 5, plus `other`), well within the 10-label limit of an `ai_classify` task — and each label is described in plain English.

# COMMAND ----------

DOC_TYPE_LABELS = {
    "Lending": {
        "loan_application": "Uniform Residential Loan Application (Form 1003): borrower, employment, income, assets, declarations, loan and property",
        "w2":               "IRS Form W-2 Wage and Tax Statement",
        "pay_stub":         "Employer earnings statement / pay stub / direct deposit advice for one pay period",
        "bank_statement":   "Bank account statement with balances and transactions",
        "appraisal_report": "Residential property appraisal report with an opinion of market value",
        "loan_estimate":    "Lender Loan Estimate disclosure with loan amount, rate, monthly payment and closing costs",
    },
    "Investments": {
        "term_sheet":             "Fund summary of principal terms: management fee, carried interest, fund term",
        "subscription_agreement": "Investor subscription agreement for a limited partnership interest in a fund",
        "kyc_w9":                 "IRS Form W-9 or other KYC / taxpayer identification form",
        "capital_call_notice":    "Capital call / drawdown notice to a limited partner",
        "wire_instructions":      "Wire transfer or payment instructions with bank, routing and account numbers",
    },
    "Claims": {
        "fnol_claim_form": "First notice of loss / insurance claim intake form",
        "police_report":   "Police or traffic collision report",
        "repair_estimate": "Auto body / collision repair estimate",
        "medical_bill":    "Medical provider itemized bill or statement of charges",
        "proof_of_loss":   "Sworn statement in proof of loss for an insurance claim",
    },
}
CLASSIFY_TASK = {"Lending": "classify_lending_pages", "Investments": "classify_investment_pages", "Claims": "classify_claim_pages"}
for lob, labels in DOC_TYPE_LABELS.items():
    define_ai_task(
        CLASSIFY_TASK[lob],
        title=f"Classify every new {lob.lower()} page into its document type",
        function="ai_classify",
        depends_on=["route_pages"],
        input={"delta_table": {"table_name": f"{CATALOG}.{SCHEMA}.pages_{lob.lower()}", "input_column": "page_text", "merge_key_columns": ["page_id"]}},
        output_table=f"page_types_{lob.lower()}",
        options={"version": "2.1", "enable_confidence_scores": True,
                 "inline_labels": {"labels": [{"name": k, "description": v} for k, v in labels.items()]
                                             + [{"name": "other", "description": "Not one of the document types above"}]},
                 "instructions": f"Each input is ONE page of a {lob.lower()} packet. Continuation pages (Page 2 of N) belong to the same form named in their footer."},
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.6 Task `split_documents` — SQL task (fan-in)
# MAGIC Waits for **all three** classifiers, then:
# MAGIC 1. combines both levels into the final type of every page (`page_classifications`);
# MAGIC 2. **splits each packet into documents** (`packet_documents`) — a new document starts when the footer says **"Page 1 of N"** (this separates the two back-to-back pay stubs) or, with no footer, when the **document type changes** (e.g. the fax cover). A *"Page 2 of 2"* page is a **continuation** and inherits the type of its document's first page;
# MAGIC 3. appends **new documents** to the input table of the extraction task for their line of business.
# MAGIC
# MAGIC Only packets whose pages have **all** been classified are split, so a failed classification never produces half a packet.

# COMMAND ----------

split_sql = r"""
-- SQL task split_documents: combine both classification levels, split each fully classified packet into documents,
-- and hand new documents to the extraction task of their line of business.

-- 1. Final document type of every page
CREATE OR REPLACE TABLE page_classifications COMMENT 'Final document type of every page (line of business + document type)' AS
WITH lob AS (
  SELECT page_id, response:response[0]:value::STRING AS line_of_business, response:response[0]:confidence_score::DOUBLE AS lob_confidence
  FROM page_line_of_business
),
types AS (
  SELECT page_id, response:response[0]:value::STRING AS doc_type, response:response[0]:confidence_score::DOUBLE AS type_confidence FROM page_types_lending
  UNION ALL SELECT page_id, response:response[0]:value::STRING, response:response[0]:confidence_score::DOUBLE FROM page_types_investments
  UNION ALL SELECT page_id, response:response[0]:value::STRING, response:response[0]:confidence_score::DOUBLE FROM page_types_claims
)
SELECT p.page_id, p.packet_id, p.page_number, l.line_of_business,
       CASE WHEN l.line_of_business = 'Other' THEN 'other' ELSE t.doc_type END AS doc_type,
       l.lob_confidence, t.type_confidence
FROM packet_pages p
JOIN lob l ON l.page_id = p.page_id
LEFT JOIN types t ON t.page_id = p.page_id
WHERE l.line_of_business = 'Other' OR t.doc_type IS NOT NULL;

-- 2. Split packets into documents. A new document starts at a "Page 1 of N" footer, or (no footer) when the type changes;
--    a "Page 2 of N" page is a continuation and inherits the type of its document's first page.
CREATE OR REPLACE TABLE packet_documents COMMENT 'Packets split into documents: type, family, page range and text' AS
WITH complete_packets AS (   -- only packets whose pages have ALL been classified
  SELECT p.packet_id FROM packet_pages p LEFT JOIN page_classifications c ON c.page_id = p.page_id
  GROUP BY p.packet_id HAVING count(*) = count(c.page_id)
),
pages AS (
  SELECT p.packet_id, p.received_date, p.page_number, p.page_text, c.doc_type AS page_type,
         try_cast(nullif(regexp_extract(p.page_text, 'Page (\\d+) of (\\d+)', 1), '') AS INT) AS page_of,   -- "Page k of n" footer
         lag(c.doc_type) OVER (PARTITION BY p.packet_id ORDER BY p.page_number)              AS prev_type
  FROM packet_pages p
  JOIN page_classifications c ON c.page_id = p.page_id
  WHERE p.packet_id IN (SELECT packet_id FROM complete_packets)
),
boundaries AS (
  SELECT *,
    CASE WHEN prev_type IS NULL       THEN 1   -- first page of the packet
         WHEN page_of = 1             THEN 1   -- footer says "Page 1 of N": a new document starts
         WHEN page_of > 1             THEN 0   -- footer says "Page 2 of N": continuation
         WHEN page_type <> prev_type  THEN 1   -- no footer: new document when the type changes
         ELSE 0 END AS starts_document
  FROM pages
),
numbered AS (
  SELECT *, sum(starts_document) OVER (PARTITION BY packet_id ORDER BY page_number) AS doc_seq FROM boundaries
),
docs AS (
  SELECT packet_id, doc_seq,
         any_value(received_date)       AS received_date,
         min_by(page_type, page_number) AS doc_type,      -- a document's type = the type of its first page
         min(page_number) AS first_page, max(page_number) AS last_page, count(*) AS pages,
         array_join(transform(array_sort(collect_list(struct(page_number, page_text))), x -> x.page_text), '\n\n') AS doc_text
  FROM numbered
  GROUP BY packet_id, doc_seq
)
SELECT packet_id || '-D' || lpad(CAST(doc_seq AS STRING), 2, '0') AS document_id, packet_id, doc_seq, received_date, doc_type,
       CASE WHEN doc_type IN ('loan_application', 'w2', 'pay_stub', 'bank_statement', 'appraisal_report', 'loan_estimate') THEN 'Lending'
            WHEN doc_type IN ('term_sheet', 'subscription_agreement', 'kyc_w9', 'capital_call_notice', 'wire_instructions') THEN 'Investments'
            WHEN doc_type IN ('fnol_claim_form', 'police_report', 'repair_estimate', 'medical_bill', 'proof_of_loss') THEN 'Claims'
            ELSE 'Other' END AS doc_family,
       first_page, last_page, pages, doc_text
FROM docs;
"""
for lob in LINES_OF_BUSINESS:   # 3. new documents -> the input table of the matching extraction task
    t = f"documents_{lob.lower()}"
    split_sql += f"""
CREATE TABLE IF NOT EXISTS {t} (document_id STRING, packet_id STRING, doc_type STRING, doc_text STRING)
  COMMENT '{lob} documents (input to extract_{lob.lower()})' TBLPROPERTIES (delta.enableChangeDataFeed = true);
INSERT INTO {t}
SELECT document_id, packet_id, doc_type, doc_text FROM packet_documents
WHERE doc_family = '{lob}' AND document_id NOT IN (SELECT document_id FROM {t});
"""
define_sql_task("split_documents", title="Split each fully classified packet into documents and queue new documents for extraction",
                depends_on=list(CLASSIFY_TASK.values()), sql=split_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.7 Field schemas — what `ai_extract` pulls from each document type
# MAGIC Each document type gets its own **field schema** — names, types and a short description. Edit these to match your forms. An `ai_extract` task takes **one schema**, so step 1.8 merges the schemas of each line of business into one, and step 1.9 keeps only the fields that belong to each document's type.

# COMMAND ----------

date = lambda d: {"type": "string", "description": f"{d} in YYYY-MM-DD format"}
num = lambda d: {"type": "number", "description": d}
txt = lambda d: {"type": "string", "description": d}
EXTRACTION_SCHEMAS = {
    # ---- Lending ----
    "loan_application": {"loan_number": txt("Lender loan number, e.g. HL-1234567"), "borrower_name": txt("Borrower full name"),
                         "employer_name": txt("Employer or business name"), "annual_income": num("Gross annual base income in USD"),
                         "loan_amount": num("Requested loan amount in USD"), "loan_purpose": txt("Loan purpose"), "property_address": txt("Subject property address")},
    "w2": {"tax_year": {"type": "integer", "description": "Tax year"}, "employee_name": txt("Employee name (box e)"),
           "employer_name": txt("Employer name only, without address (box c)"), "wages": num("Box 1 wages, tips, other compensation in USD"),
           "federal_tax_withheld": num("Box 2 federal income tax withheld in USD")},
    "pay_stub": {"employee_name": txt("Employee name"), "employer_name": txt("Employer name"), "pay_period_end": date("Pay period end date"),
                 "gross_pay": num("Current period gross pay in USD"), "ytd_gross": num("Year-to-date gross pay in USD")},
    "bank_statement": {"account_holder": txt("Account holder name"), "bank_name": txt("Bank name"),
                       "statement_end_date": date("Statement period end date"), "ending_balance": num("Ending balance in USD")},
    "appraisal_report": {"loan_number": txt("Lender loan number"), "property_address": txt("Subject property address"),
                         "appraised_value": num("Opinion of market value in USD"), "effective_date": date("Effective date of appraisal"), "appraiser_name": txt("Appraiser name")},
    "loan_estimate": {"loan_number": txt("Loan ID number"), "borrower_name": txt("Applicant name"), "loan_amount": num("Loan amount in USD"),
                      "interest_rate": num("Interest rate in percent, e.g. 6.375"), "loan_term_months": {"type": "integer", "description": "Loan term in months"},
                      "monthly_principal_interest": num("Monthly principal and interest in USD")},
    # ---- Investments ----
    "term_sheet": {"fund_name": txt("Fund legal name"), "management_fee_pct": num("Management fee in percent per annum, e.g. 1.5"),
                   "carried_interest_pct": num("Carried interest in percent"), "preferred_return_pct": num("Preferred return in percent")},
    "subscription_agreement": {"investor_account": txt("Investor account number, e.g. INV-123456"), "investor_name": txt("Name of investor"),
                               "fund_name": txt("Fund legal name"), "commitment_amount": num("Capital commitment in USD"),
                               "tax_id_last4": txt("Last 4 digits of the taxpayer identification number")},
    "kyc_w9": {"investor_name": txt("Name of entity or individual (line 1)"), "tax_classification": txt("Federal tax classification"),
               "tax_id_last4": txt("Last 4 digits of the taxpayer identification number")},
    "capital_call_notice": {"investor_account": txt("Investor account number"), "investor_name": txt("Limited partner name"), "fund_name": txt("Fund legal name"),
                            "call_number": {"type": "integer", "description": "Capital call number"}, "call_amount": num("Amount of this capital call in USD"),
                            "due_date": date("Payment due date")},
    "wire_instructions": {"investor_account": txt("Investor account number from the reference line, e.g. INV-123456 (without the call number)"),
                          "beneficiary_name": txt("Beneficiary name"), "bank_name": txt("Beneficiary bank name"),
                          "routing_number": txt("ABA routing number"), "account_last4": txt("Last 4 digits of the account number")},
    # ---- Claims ----
    "fnol_claim_form": {"claim_number": txt("Claim number, e.g. CLM-12345678"), "policy_number": txt("Policy number"), "insured_name": txt("Named insured"),
                        "date_of_loss": date("Date of loss"), "vehicle_vin": txt("17-character vehicle identification number"),
                        "injuries_reported": {"type": "boolean", "description": "Were injuries reported"}},
    "police_report": {"report_number": txt("Police report number"), "incident_date": date("Date of incident"),
                      "driver_name": txt("Unit 1 driver name"), "vehicle_vin": txt("Unit 1 vehicle identification number (17 characters)")},
    "repair_estimate": {"claim_number": txt("Claim number"), "vehicle_vin": txt("Vehicle identification number (17 characters)"),
                        "estimate_total": num("Grand total of the estimate in USD")},
    "medical_bill": {"claim_number": txt("Insurance claim reference number"), "patient_name": txt("Patient name"),
                     "date_of_service": date("Date of service"), "total_charges": num("Total charges in USD")},
    "proof_of_loss": {"claim_number": txt("Claim number"), "policy_number": txt("Policy number"), "insured_name": txt("Insured name"),
                      "date_of_loss": date("Date of loss"), "amount_claimed": num("Amount claimed in USD")},
}
DOC_FAMILY = {**{t: "Lending" for t in ["loan_application", "w2", "pay_stub", "bank_statement", "appraisal_report", "loan_estimate"]},
              **{t: "Investments" for t in ["term_sheet", "subscription_agreement", "kyc_w9", "capital_call_notice", "wire_instructions"]},
              **{t: "Claims" for t in ["fnol_claim_form", "police_report", "repair_estimate", "medical_bill", "proof_of_loss"]}}
display(spark.createDataFrame([(DOC_FAMILY[k], k, len(v), ", ".join(v)) for k, v in EXTRACTION_SCHEMAS.items()],
                              "line_of_business string, doc_type string, fields int, field_names string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.8 Tasks `extract_lending` · `extract_investments` · `extract_claims` — AI Function tasks · `ai_extract`, fanned out
# MAGIC One extraction task **per line of business**, all three **in parallel** after `split_documents`. Each reads its own documents table and returns typed values with a **confidence score** per field, so low-confidence values can go to a human instead of being trusted blindly.

# COMMAND ----------

def lob_schema(lob):
    """One ai_extract schema per line of business: the union of its document types' fields, each description saying where the field appears."""
    merged = {}
    for doc_type, fields in EXTRACTION_SCHEMAS.items():
        if DOC_FAMILY[doc_type] != lob:
            continue
        for f, spec in fields.items():
            if f not in merged:
                merged[f] = dict(spec, description=f"{doc_type}: {spec['description']}")
            else:
                merged[f]["description"] += f"; {doc_type}: {spec['description']}"
    return merged

EXTRACT_TASK = {"Lending": "extract_lending", "Investments": "extract_investments", "Claims": "extract_claims"}
for lob in LINES_OF_BUSINESS:
    doc_types = [t for t in EXTRACTION_SCHEMAS if DOC_FAMILY[t] == lob]
    define_ai_task(
        EXTRACT_TASK[lob],
        title=f"Extract fields from every new {lob.lower()} document ({', '.join(doc_types)})",
        function="ai_extract",
        depends_on=["split_documents"],
        input={"delta_table": {"table_name": f"{CATALOG}.{SCHEMA}.documents_{lob.lower()}", "input_column": "doc_text", "merge_key_columns": ["document_id"]}},
        output_table=f"extracted_{lob.lower()}",
        options={"version": "2.1", "enable_confidence_scores": True, "schema": json.dumps(lob_schema(lob)),
                 "instructions": f"The input is ONE {lob.lower()} document: one of {', '.join(doc_types)}. Each field description names the document types that "
                                 "carry it. Fill only the fields that appear on this document and leave every other field null. "
                                 "Return amounts as plain numbers without currency symbols, and dates as YYYY-MM-DD."},
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.9 Task `assemble_extractions` — SQL task (fan-in)
# MAGIC Waits for **all three** extraction tasks, then builds the two tables the rest of the notebook works from: `extracted_fields` (one row per field, only the fields of each document's type) and `packet_index` (which loan, investor account or claim each packet is about — W-2s, pay stubs and police reports don't carry that number, so they inherit it from the packet).

# COMMAND ----------

doc_type_fields = ",\n  ".join(f"('{t}', '{f}')" for t, fields in EXTRACTION_SCHEMAS.items() for f in fields)
assemble_sql = f"""-- SQL task assemble_extractions (fan-in): one row per extracted field — only the fields that belong to each document's type —
-- and the system-of-record key (loan / investor account / claim number) of every packet.
CREATE OR REPLACE TABLE extracted_fields COMMENT 'One row per extracted field: value and confidence' AS
WITH raw AS (
  SELECT document_id, response FROM extracted_lending     WHERE response IS NOT NULL AND nullif(error::STRING, '') IS NULL
  UNION ALL SELECT document_id, response FROM extracted_investments WHERE response IS NOT NULL AND nullif(error::STRING, '') IS NULL
  UNION ALL SELECT document_id, response FROM extracted_claims      WHERE response IS NOT NULL AND nullif(error::STRING, '') IS NULL
),
exploded AS (
  SELECT r.document_id, f.key AS field_name, f.value AS v FROM raw r, LATERAL variant_explode(r.response:response) AS f
),
doc_type_fields AS (
  SELECT * FROM VALUES
  {doc_type_fields}
  AS t(doc_type, field_name)
)
SELECT d.document_id, d.packet_id, d.doc_type, e.field_name,
       e.v:value::STRING            AS field_value,
       e.v:confidence_score::DOUBLE AS confidence
FROM exploded e
JOIN packet_documents d ON d.document_id = e.document_id
JOIN doc_type_fields m  ON m.doc_type = d.doc_type AND m.field_name = e.field_name;

""" + r"""
-- Several documents carry the record key: take the most common value. W-2s, pay stubs, police reports and term sheets
-- don't carry it — they inherit it from the packet.
CREATE OR REPLACE TABLE packet_index COMMENT 'One row per packet: line of business and system-of-record key' AS
WITH lob AS (
  SELECT packet_id, any_value(received_date) AS received_date, mode(doc_family) AS line_of_business,
         count(*) AS documents, sum(pages) AS doc_pages
  FROM packet_documents WHERE doc_family <> 'Other' GROUP BY packet_id
),
keys AS (
  SELECT packet_id, upper(trim(field_value)) AS record_key, count(*) AS votes
  FROM extracted_fields
  WHERE field_name IN ('loan_number', 'investor_account', 'claim_number') AND nullif(trim(field_value), '') IS NOT NULL
  GROUP BY ALL
)
SELECT l.*, max_by(k.record_key, k.votes) AS record_key, count(k.record_key) AS key_variants_seen
FROM lob l LEFT JOIN keys k USING (packet_id)
GROUP BY ALL;
"""
define_sql_task("assemble_extractions", title="Fan-in: one row per extracted field, and each packet's system-of-record key",
                depends_on=list(EXTRACT_TASK.values()), sql=assemble_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.10 The DAG
# MAGIC Drawn from the task definitions above — the same dependencies step 2.1 hands to Lakeflow Jobs. **Red** = native AI Function task, **dark** = SQL task.

# COMMAND ----------

show_dag()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deploy the Job and run the backfill
# MAGIC
# MAGIC **2.1** writes the SQL tasks' **SQL files** (into a `document_intelligence_sql` folder next to this notebook) and creates — or updates — the **Lakeflow Job** with the Jobs API: 8 AI Function tasks + 5 SQL tasks, one retry per task, a 6 a.m. daily schedule (created **paused**) and `max_concurrent_runs = 1` (required for AI Function tasks).
# MAGIC
# MAGIC **2.2** runs it. This first run is the **backfill**: every packet in the archive goes through parse → classify → split → extract. It takes about 11 minutes; the cell prints progress as tasks finish. **2.3** shows the run as a per-task timeline, with the number of rows each AI Function task processed.
# MAGIC
# MAGIC > 💡 Re-run **2.1** whenever you change a task in section 1 — it rewrites the SQL files and updates the Job in place. Changes to an AI task's labels, schema or instructions apply to inputs processed *after* the change.

# COMMAND ----------

from databricks.sdk.service import workspace

w.workspace.mkdirs(SQL_DIR)
for s in TASKS.values():
    if s["kind"] == "sql":
        w.workspace.import_(path=sql_file_path(s["key"]), format=workspace.ImportFormat.RAW, overwrite=True,
                            content=base64.b64encode(sql_file_content(s).encode()).decode())

settings = job_settings()
found = list(w.jobs.list(name=JOB_NAME))
if found:
    JOB_ID = found[0].job_id
    _api("POST", "/api/2.2/jobs/reset", body={"job_id": JOB_ID, "new_settings": settings})
    print(f"Updated Job '{JOB_NAME}' ({JOB_ID}) — {len(TASKS)} tasks")
else:
    JOB_ID = _api("POST", "/api/2.2/jobs/create", body=settings)["job_id"]
    print(f"Created Job '{JOB_NAME}' ({JOB_ID}) — {len(TASKS)} tasks · schedule 6:00 a.m. America/New_York (PAUSED)")

display(spark.createDataFrame([(s["key"], s["badge"], ", ".join(s["depends_on"]) or "—",
                                (s["output_table"] if s["kind"] == "ai" else sql_file_path(s["key"]))) for s in TASKS.values()],
                              "task string, type string, runs_after string, output_table_or_sql_file string"))
displayHTML(button("Open the Job (Tasks tab shows the DAG)", f"{HOST}/jobs/{JOB_ID}{ORG}")
            + button("Open the SQL files", f"{HOST}/browse/folders/{w.workspace.get_status(SQL_DIR).object_id}{ORG}", "#1B3139"))
print("Job definition (Jobs API) — first two tasks:\n" + json.dumps({**settings, "tasks": settings["tasks"][:2]}, indent=2)[:2500])

# COMMAND ----------

BACKFILL_RUN_ID = run_pipeline()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 What the backfill run did, task by task
# MAGIC The same view the Jobs UI gives you, plus the **row counts** each AI Function task reports for the run (new inputs processed, succeeded, failed). Notice the classification and extraction tasks running **side by side**.

# COMMAND ----------

show_run(BACKFILL_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4 🛠 Try it yourself
# MAGIC Click **Open this run** in step 2.3. In the **Graph** view, click `extract_claims`: the run output shows the **rows, succeeded and failed** counts and a **Sample Data** tab with rows of its output table. Then open the **Tasks** tab of the Job and click `classify_lending_pages` — the labels and descriptions from step 1.5 are right there in the *Classification* task form, editable without code. *If a task ever fails, fix it and click **Repair run**: only that task and the ones after it re-run.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. What `ai_parse_document` produced — tasks `parse_documents` and `prepare_pages`
# MAGIC
# MAGIC One SQL function turned each PDF into **structured content**: text, **tables** (as HTML, rows and columns intact), titles, section headers, page headers / footers and page numbers, each with its **bounding box** and **page** — plus a rendered **image of every page** for review screens. No OCR vendor, no templates per form.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- parsed_packets is the output table the parse_documents AI Function task created and manages: packet_id (the merge key), response, error
# MAGIC SELECT count(*) AS packets_parsed,
# MAGIC        sum(size(CAST(response:document:pages AS ARRAY<VARIANT>))) AS pages,
# MAGIC        sum(size(CAST(response:document:elements AS ARRAY<VARIANT>))) AS elements,
# MAGIC        count_if(nullif(error::STRING, '') IS NOT NULL) AS packets_with_errors
# MAGIC FROM parsed_packets;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT packet_id, page_number, elements, tables, left(page_text, 160) AS page_text_preview
# MAGIC FROM packet_pages
# MAGIC WHERE packet_id = (SELECT max(packet_id) FROM packet_pages)
# MAGIC ORDER BY page_number;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Look at a packet
# MAGIC These are the page images `ai_parse_document` rendered into the `page_images` volume. To `ai_parse_document` they're just pages — nothing knows yet what each one *is*. That's the classification tasks' job.

# COMMAND ----------

show_packet()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.4 🛠 Try it yourself
# MAGIC Look at the full parse for one packet: `SELECT response FROM parsed_packets LIMIT 1` and expand the VARIANT. Find a **table** element — notice the rows and columns come back as HTML, so the pay stub and repair-estimate line items survive intact. Then find an element of type `page_number` — the `split_documents` task uses it to split the packet.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. What `ai_classify` produced — tasks `classify_*` and `split_documents`
# MAGIC Every page labelled first with its line of business, then with its document type — and every packet split into its documents (see step 1.6 for the splitting rules).

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT line_of_business, doc_type, count(*) AS pages,
# MAGIC        round(avg(lob_confidence), 2) AS avg_lob_confidence, round(avg(type_confidence), 2) AS avg_type_confidence
# MAGIC FROM page_classifications
# MAGIC GROUP BY ALL
# MAGIC ORDER BY line_of_business, pages DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT packet_id, count(*) AS documents, sum(pages) AS pages,
# MAGIC        array_join(transform(array_sort(collect_list(struct(doc_seq, doc_type, first_page, last_page))),
# MAGIC                   d -> d.doc_type || ' (p' || d.first_page || IF(d.last_page > d.first_page, '-' || d.last_page, '') || ')'), ' → ') AS packet_contents
# MAGIC FROM packet_documents
# MAGIC GROUP BY packet_id
# MAGIC ORDER BY packet_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 The same packet — now classified and split
# MAGIC Each colour is a line of business; **▶ document N starts** marks where the splitter cut the packet. Look for the two back-to-back pay stubs and the fax cover sheet.

# COMMAND ----------

show_packet(labels=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Synthetic data only: compare every page with the generator's answer key.
# MAGIC WITH pred AS (
# MAGIC   SELECT pp.packet_id, pp.page_number, d.doc_type AS predicted_type, d.doc_seq AS predicted_doc,
# MAGIC          c.doc_type AS page_level_type
# MAGIC   FROM packet_pages pp
# MAGIC   JOIN packet_documents d      ON d.packet_id = pp.packet_id AND pp.page_number BETWEEN d.first_page AND d.last_page
# MAGIC   JOIN page_classifications c  ON c.packet_id = pp.packet_id AND c.page_number = pp.page_number
# MAGIC ),
# MAGIC scored AS (
# MAGIC   SELECT p.*, k.doc_type AS true_type, k.doc_seq AS true_doc
# MAGIC   FROM pred p JOIN answer_key_pages k USING (packet_id, page_number)
# MAGIC ),
# MAGIC per_packet AS (
# MAGIC   SELECT packet_id, bool_and(predicted_doc = true_doc) AS split_correct FROM scored GROUP BY packet_id
# MAGIC )
# MAGIC SELECT
# MAGIC   (SELECT count(*) FROM scored)                                                                   AS pages_scored,
# MAGIC   (SELECT round(100 * avg(CAST(page_level_type = true_type AS INT)), 1) FROM scored)              AS page_classification_accuracy_pct,
# MAGIC   (SELECT round(100 * avg(CAST(predicted_type = true_type AS INT)), 1) FROM scored)               AS accuracy_after_split_pct,
# MAGIC   (SELECT count(*) FROM per_packet)                                                                AS packets_scored,
# MAGIC   (SELECT round(100 * avg(CAST(split_correct AS INT)), 1) FROM per_packet)                        AS packets_split_exactly_right_pct;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.5 💬 Ask Genie — what came in?
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("How many documents of each type did we receive, by line of business?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.6 🛠 Try it yourself
# MAGIC In a new cell, run `ai_classify` on 20 pages from `packet_pages` with **plain labels and no descriptions** — e.g. `'["loan_application","w2","pay_stub","other"]'` — and compare with `page_classifications`. Then put the descriptions back. *(Experimenting in a separate cell doesn't touch the Job's tables. When you like a change, put it in step 1.5 and re-run 2.1 to redeploy — or edit the labels right in the task's *Classification* form in the Jobs UI.)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. What `ai_extract` produced — tasks `extract_*` and `assemble_extractions`
# MAGIC Typed loan, deal and claim fields for every document, each with a **confidence score**, and every packet linked to its **system-of-record key** (loan number, investor account or claim number) — ready to reconcile in section 7.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The extract_* AI Function tasks' output tables: document_id, response (fields with value + confidence_score), error
# MAGIC SELECT task, d.doc_type, count(*) AS documents, count_if(nullif(e.error::STRING, '') IS NOT NULL) AS errors
# MAGIC FROM (SELECT 'extract_lending' AS task, * FROM extracted_lending
# MAGIC       UNION ALL SELECT 'extract_investments', * FROM extracted_investments
# MAGIC       UNION ALL SELECT 'extract_claims', * FROM extracted_claims) e
# MAGIC JOIN packet_documents d ON d.document_id = e.document_id
# MAGIC GROUP BY ALL
# MAGIC ORDER BY task, doc_type;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The loan file with the most documents: everything ai_extract read from it
# MAGIC SELECT d.doc_seq, d.doc_type,
# MAGIC        to_json(map_from_entries(collect_list(struct(f.field_name, f.field_value)))) AS extracted,
# MAGIC        round(min(f.confidence), 2) AS lowest_confidence
# MAGIC FROM packet_documents d JOIN extracted_fields f ON f.document_id = d.document_id
# MAGIC WHERE d.packet_id = (SELECT max_by(packet_id, documents) FROM packet_index)
# MAGIC GROUP BY ALL ORDER BY d.doc_seq;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT doc_type, field_name, count(*) AS extracted, count_if(field_value IS NULL) AS not_found,
# MAGIC        round(avg(confidence), 3) AS avg_confidence, count_if(confidence < 0.8) AS below_0_8
# MAGIC FROM extracted_fields
# MAGIC GROUP BY ALL
# MAGIC ORDER BY avg_confidence, doc_type;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4 💬 Ask Genie — extracted fields
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("For each loan packet, show the loan number, the W-2 wages and the appraised value we extracted from the documents.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.5 🛠 Try it yourself
# MAGIC Add a field to the `pay_stub` schema in **step 1.7** — e.g. `"pay_frequency": txt("Weekly, biweekly, semi-monthly or monthly")` — and re-run **steps 1.7–1.9 and 2.1** to update the Job. AI Function tasks apply setting changes to inputs processed *after* the change, so queue the existing pay stubs again: `DELETE FROM documents_lending WHERE doc_type = 'pay_stub'`. Then run `run_pipeline()` in a new cell: `split_documents` re-adds them and only `extract_lending` does any AI work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tomorrow morning — the same DAG, only the new packets
# MAGIC
# MAGIC The backfill ran once over the archive. From now on the Job runs **every morning at 6 a.m.** (once you unpause the schedule) — and because every AI Function task only processes inputs that are new (and every SQL task only appends new rows), a daily run touches only that day's packets.
# MAGIC
# MAGIC **6.1** lands today's packets in the inbox (in real life they arrive from your scanner, email or SFTP drop). **6.2** runs the Job — exactly what the schedule would do. **6.3** compares the runs side by side: the daily run parses, classifies and extracts only the new packets.

# COMMAND ----------

landed_today = land_packets([today], PACKETS_PER_DAY, seq=today.toordinal() * PACKETS_PER_DAY)
refresh_inbox_view()
print(f"Landed {len(landed_today)} new packets for {today}")
if landed_today:
    display(spark.createDataFrame(landed_today, "packet_id string, arrival_date string, line_of_business string, pages int, planted_issues string"))

# COMMAND ----------

DAILY_RUN_ID = run_pipeline()

# COMMAND ----------

# Every run of the Job, with the rows its AI Function tasks processed
runs = []
for r in w.jobs.list_runs(job_id=get_job_id(), expand_tasks=True, limit=10):
    rows = {t.task_key: _task_rows(t.run_id) for t in (r.tasks or []) if TASKS.get(t.task_key, {}).get("kind") == "ai"}
    n = lambda k: (rows.get(k) or {}).get("rows")
    runs.append((r.run_id, r.state.result_state.value if r.state and r.state.result_state else str(r.state.life_cycle_state.value),
                 round(((r.end_time or 0) - (r.start_time or 0)) / 60000, 1),
                 n("parse_documents"), n("classify_line_of_business"),
                 sum((n(k) or 0) for k in CLASSIFY_TASK.values()), sum((n(k) or 0) for k in EXTRACT_TASK.values()),
                 sum((v or {}).get("failed", 0) for v in rows.values())))
display(spark.createDataFrame(runs, "job_run_id long, result string, minutes double, packets_parsed long, pages_classified_by_lob long, "
                                    "pages_classified_by_type long, documents_extracted long, failed_rows long"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.4 🛠 Try it yourself — make it production-shaped
# MAGIC * **Turn on the schedule:** open the Job (step 2.1) → **Schedules & Triggers** → unpause the 6 a.m. schedule.
# MAGIC * **Or react to arrivals instead of the clock:** switch the trigger to **File arrival** on the inbox volume, so packets are processed minutes after they land:
# MAGIC   ```python
# MAGIC   w.jobs.update(job_id=get_job_id(), fields_to_remove=["schedule"],   # a Job has either a schedule or a trigger
# MAGIC       new_settings=jobs.JobSettings(trigger=jobs.TriggerSettings(
# MAGIC           file_arrival=jobs.FileArrivalTriggerConfiguration(url=INBOX + "/"), pause_status=jobs.PauseStatus.UNPAUSED)))
# MAGIC   ```
# MAGIC * **Get told when it breaks:** add an email or Slack **notification on failure** to the Job.
# MAGIC
# MAGIC Sections 7 and 8 below work on **everything the Job has produced** — the archive *and* today's packets. After future scheduled runs, re-run them (or make them Job tasks too — see *Build more*, step 10.2).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Reconcile against the systems of record
# MAGIC
# MAGIC The Job has turned paper into fields. Now the question that decides whether documents can flow straight through: **does the document agree with what the bank, the fund administrator or the insurer has on file?**
# MAGIC
# MAGIC * **7.1** — the **rules**, as data: which document field is compared with which system-of-record column, and how (`amount` within tolerance, `name`, `id`, `date`, *within policy period* …). Plus the **required documents** for each line of business. Small **Unity Catalog SQL functions** normalize names, amounts and dates — governed and reusable from any query, dashboard or Genie.
# MAGIC * **7.2** — every field reconciled: `MATCH`, `MISMATCH`, `MISSING_DOCUMENT`, `NOT_ON_DOCUMENT` or `NO_SYSTEM_RECORD`.
# MAGIC
# MAGIC Rules are deliberately **strict and cheap**: they settle the easy 95% (formatting, case, punctuation, middle initials) and hand everything else to `ai_decide` in section 8. Change the `amount_tolerance_pct` widget to see more or fewer mismatches.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Normalizing functions, governed in Unity Catalog
# MAGIC CREATE OR REPLACE FUNCTION norm_name(x STRING) RETURNS STRING
# MAGIC   COMMENT 'Person / entity name for comparison: lower case, no punctuation, no middle initials, tokens sorted'
# MAGIC   RETURN array_join(array_sort(filter(split(trim(regexp_replace(lower(x), '[^a-z0-9 ]', ' ')), ' +'), t -> length(t) > 1)), ' ');
# MAGIC CREATE OR REPLACE FUNCTION norm_text(x STRING) RETURNS STRING
# MAGIC   COMMENT 'Text / identifier for comparison: lower case letters and digits only'
# MAGIC   RETURN regexp_replace(lower(x), '[^a-z0-9]', '');
# MAGIC CREATE OR REPLACE FUNCTION to_amount(x STRING) RETURNS DOUBLE
# MAGIC   COMMENT 'Parses $1,234.50 / 6.375 % / 1234.5 into a number'
# MAGIC   RETURN coalesce(try_cast(x AS DOUBLE), try_cast(regexp_replace(x, '[^0-9.\\-]', '') AS DOUBLE));
# MAGIC CREATE OR REPLACE FUNCTION to_date_any(x STRING) RETURNS DATE
# MAGIC   COMMENT 'Parses yyyy-MM-dd, MM/dd/yyyy or "March 3, 2026" into a date'
# MAGIC   RETURN coalesce(try_to_date(x, 'yyyy-MM-dd'), try_to_date(x, 'MM/dd/yyyy'), try_to_date(x, 'MMMM d, yyyy'));
# MAGIC
# MAGIC -- Which document field is checked against which system-of-record column, and how
# MAGIC CREATE OR REPLACE TABLE recon_rules COMMENT 'Reconciliation rules: document field -> system-of-record column, compare type' AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('Lending', 'loan_application', 'borrower_name',  'borrower_name',  'name',   'Borrower on application = LOS borrower'),
# MAGIC   ('Lending', 'loan_application', 'employer_name',  'employer_name',  'name',   'Employer on application = LOS employer'),
# MAGIC   ('Lending', 'loan_application', 'annual_income',  'annual_income',  'amount', 'Stated income = LOS income'),
# MAGIC   ('Lending', 'loan_application', 'loan_amount',    'loan_amount',    'amount', 'Requested amount = LOS loan amount'),
# MAGIC   ('Lending', 'loan_application', 'property_address','property_address','text', 'Subject property = LOS property'),
# MAGIC   ('Lending', 'w2',               'employee_name',  'borrower_name',  'name',   'W-2 employee is the borrower'),
# MAGIC   ('Lending', 'w2',               'employer_name',  'employer_name',  'name',   'W-2 employer = LOS employer'),
# MAGIC   ('Lending', 'w2',               'wages',          'annual_income',  'amount', 'W-2 wages support LOS income'),
# MAGIC   ('Lending', 'pay_stub',         'employee_name',  'borrower_name',  'name',   'Pay stub employee is the borrower'),
# MAGIC   ('Lending', 'pay_stub',         'employer_name',  'employer_name',  'name',   'Pay stub employer = LOS employer'),
# MAGIC   ('Lending', 'bank_statement',   'account_holder', 'borrower_name',  'name',   'Bank account belongs to the borrower'),
# MAGIC   ('Lending', 'appraisal_report', 'loan_number',    'loan_number',    'id',     'Appraisal is for this loan'),
# MAGIC   ('Lending', 'appraisal_report', 'property_address','property_address','text', 'Appraised property = LOS property'),
# MAGIC   ('Lending', 'appraisal_report', 'appraised_value','appraised_value','amount', 'Appraised value = LOS value used for LTV'),
# MAGIC   ('Lending', 'loan_estimate',    'loan_amount',    'loan_amount',    'amount', 'Disclosed amount = LOS loan amount'),
# MAGIC   ('Lending', 'loan_estimate',    'interest_rate',  'interest_rate',  'number', 'Disclosed rate = LOS note rate'),
# MAGIC   ('Lending', 'loan_estimate',    'loan_term_months','term_months',   'number', 'Disclosed term = LOS term'),
# MAGIC   ('Investments', 'term_sheet',             'fund_name',          'fund_name',          'text',   'Terms are for this fund'),
# MAGIC   ('Investments', 'term_sheet',             'management_fee_pct', 'management_fee_pct', 'number', 'Fee in terms = fee in fund admin'),
# MAGIC   ('Investments', 'subscription_agreement', 'investor_name',      'investor_name',      'name',   'Subscriber = investor on file'),
# MAGIC   ('Investments', 'subscription_agreement', 'fund_name',          'fund_name',          'text',   'Subscription is for this fund'),
# MAGIC   ('Investments', 'subscription_agreement', 'commitment_amount',  'commitment_amount',  'amount', 'Commitment = commitment on file'),
# MAGIC   ('Investments', 'subscription_agreement', 'tax_id_last4',       'tax_id_last4',       'last4',  'TIN = TIN on file'),
# MAGIC   ('Investments', 'kyc_w9',                 'investor_name',      'investor_name',      'name',   'W-9 entity = investor on file'),
# MAGIC   ('Investments', 'kyc_w9',                 'tax_id_last4',       'tax_id_last4',       'last4',  'W-9 TIN = TIN on file'),
# MAGIC   ('Investments', 'capital_call_notice',    'investor_name',      'investor_name',      'name',   'Notice addressed to investor on file'),
# MAGIC   ('Investments', 'capital_call_notice',    'call_number',        'call_number',        'number', 'Call number = current call'),
# MAGIC   ('Investments', 'capital_call_notice',    'call_amount',        'call_amount',        'amount', 'Amount called = amount in fund admin'),
# MAGIC   ('Investments', 'capital_call_notice',    'due_date',           'call_due_date',      'date',   'Due date = due date on file'),
# MAGIC   ('Investments', 'wire_instructions',      'bank_name',          'wire_bank_name',     'text',   'Wire bank = bank on file'),
# MAGIC   ('Investments', 'wire_instructions',      'routing_number',     'wire_routing_number','id',     'Routing number = routing on file'),
# MAGIC   ('Investments', 'wire_instructions',      'account_last4',      'wire_account_last4', 'last4',  'Account = account on file'),
# MAGIC   ('Claims', 'fnol_claim_form', 'policy_number', 'policy_number',  'id',             'FNOL policy = policy on claim'),
# MAGIC   ('Claims', 'fnol_claim_form', 'insured_name',  'insured_name',   'name',           'FNOL insured = named insured'),
# MAGIC   ('Claims', 'fnol_claim_form', 'date_of_loss',  'date_of_loss',   'date',           'FNOL date of loss = claim date of loss'),
# MAGIC   ('Claims', 'fnol_claim_form', 'date_of_loss',  'policy_period',  'date_in_period', 'Loss occurred while the policy was in force'),
# MAGIC   ('Claims', 'fnol_claim_form', 'vehicle_vin',   'vehicle_vin',    'id',             'FNOL vehicle = insured vehicle'),
# MAGIC   ('Claims', 'police_report',   'incident_date', 'date_of_loss',   'date',           'Police report date = date of loss'),
# MAGIC   ('Claims', 'police_report',   'driver_name',   'insured_name',   'name',           'Driver is the insured'),
# MAGIC   ('Claims', 'police_report',   'vehicle_vin',   'vehicle_vin',    'id',             'Police report vehicle = insured vehicle'),
# MAGIC   ('Claims', 'repair_estimate', 'vehicle_vin',   'vehicle_vin',    'id',             'Estimate is for the insured vehicle'),
# MAGIC   ('Claims', 'repair_estimate', 'estimate_total','reserve_amount', 'amount_not_above','Estimate within the reserve'),
# MAGIC   ('Claims', 'medical_bill',    'patient_name',  'insured_name',   'name',           'Patient is the insured'),
# MAGIC   ('Claims', 'proof_of_loss',   'policy_number', 'policy_number',  'id',             'Proof of loss policy = policy on claim'),
# MAGIC   ('Claims', 'proof_of_loss',   'insured_name',  'insured_name',   'name',           'Sworn by the named insured'),
# MAGIC   ('Claims', 'proof_of_loss',   'date_of_loss',  'date_of_loss',   'date',           'Sworn date of loss = claim date of loss'),
# MAGIC   ('Claims', 'proof_of_loss',   'amount_claimed','coverage_limit', 'amount_not_above','Amount claimed within the coverage limit')
# MAGIC AS t(line_of_business, doc_type, field_name, system_column, compare_type, rule_description);
# MAGIC
# MAGIC CREATE OR REPLACE TABLE required_documents COMMENT 'Documents every packet must contain, by line of business' AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('Lending', 'loan_application'), ('Lending', 'w2'), ('Lending', 'pay_stub'), ('Lending', 'bank_statement'),
# MAGIC   ('Lending', 'appraisal_report'), ('Lending', 'loan_estimate'),
# MAGIC   ('Investments', 'term_sheet'), ('Investments', 'subscription_agreement'), ('Investments', 'kyc_w9'),
# MAGIC   ('Investments', 'capital_call_notice'), ('Investments', 'wire_instructions'),
# MAGIC   ('Claims', 'fnol_claim_form'), ('Claims', 'police_report'), ('Claims', 'repair_estimate'), ('Claims', 'proof_of_loss')
# MAGIC AS t(line_of_business, doc_type);
# MAGIC
# MAGIC -- The three systems of record as one long (key, column, value) table, so one join reconciles every line of business
# MAGIC CREATE OR REPLACE TABLE sor_values COMMENT 'Systems of record unpivoted to (line_of_business, record_key, system_column, system_value)' AS
# MAGIC SELECT 'Lending' AS line_of_business, upper(loan_number) AS record_key, system_column, system_value
# MAGIC FROM (SELECT loan_number, borrower_name, property_address, CAST(CAST(loan_amount AS DECIMAL(18, 2)) AS STRING) AS loan_amount,
# MAGIC              CAST(interest_rate AS STRING) AS interest_rate, CAST(term_months AS STRING) AS term_months, employer_name,
# MAGIC              CAST(CAST(annual_income AS DECIMAL(18, 2)) AS STRING) AS annual_income, CAST(CAST(appraised_value AS DECIMAL(18, 2)) AS STRING) AS appraised_value, loan_number AS loan_number_value
# MAGIC       FROM sor_loans)
# MAGIC UNPIVOT (system_value FOR system_column IN (borrower_name, property_address, loan_amount, interest_rate, term_months, employer_name,
# MAGIC                                             annual_income, appraised_value, loan_number_value AS loan_number))
# MAGIC UNION ALL
# MAGIC SELECT 'Investments', upper(investor_account), system_column, system_value
# MAGIC FROM (SELECT investor_account, investor_name, fund_name, CAST(CAST(commitment_amount AS DECIMAL(18, 2)) AS STRING) AS commitment_amount,
# MAGIC              CAST(management_fee_pct AS STRING) AS management_fee_pct, tax_id_last4, CAST(call_number AS STRING) AS call_number,
# MAGIC              CAST(CAST(call_amount AS DECIMAL(18, 2)) AS STRING) AS call_amount, CAST(call_due_date AS STRING) AS call_due_date,
# MAGIC              wire_bank_name, wire_routing_number, wire_account_last4
# MAGIC       FROM sor_investor_commitments)
# MAGIC UNPIVOT (system_value FOR system_column IN (investor_name, fund_name, commitment_amount, management_fee_pct, tax_id_last4, call_number,
# MAGIC                                             call_amount, call_due_date, wire_bank_name, wire_routing_number, wire_account_last4))
# MAGIC UNION ALL
# MAGIC SELECT 'Claims', upper(claim_number), system_column, system_value
# MAGIC FROM (SELECT claim_number, policy_number, insured_name, CAST(date_of_loss AS STRING) AS date_of_loss,
# MAGIC              policy_effective_date || ' to ' || policy_expiration_date AS policy_period, vehicle_vin,
# MAGIC              CAST(CAST(reserve_amount AS DECIMAL(18, 2)) AS STRING) AS reserve_amount, CAST(CAST(coverage_limit AS DECIMAL(18, 2)) AS STRING) AS coverage_limit
# MAGIC       FROM sor_claims)
# MAGIC UNPIVOT (system_value FOR system_column IN (policy_number, insured_name, date_of_loss, policy_period, vehicle_vin, reserve_amount, coverage_limit));
# MAGIC
# MAGIC SELECT line_of_business, count(*) AS rules, count(DISTINCT doc_type) AS document_types FROM recon_rules GROUP BY ALL;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE recon_results COMMENT 'Every document field reconciled against the system of record' AS
# MAGIC WITH fields AS (
# MAGIC   SELECT f.document_id, f.packet_id, k.received_date, k.line_of_business, k.record_key, f.doc_type, f.field_name, f.field_value, f.confidence,
# MAGIC          r.system_column, r.compare_type, r.rule_description
# MAGIC   FROM extracted_fields f
# MAGIC   JOIN packet_index k USING (packet_id)
# MAGIC   JOIN recon_rules r ON r.line_of_business = k.line_of_business AND r.doc_type = f.doc_type AND r.field_name = f.field_name
# MAGIC ),
# MAGIC joined AS (
# MAGIC   SELECT f.*, s.system_value, rk.record_key IS NOT NULL AS record_found
# MAGIC   FROM fields f
# MAGIC   LEFT JOIN sor_values s ON s.line_of_business = f.line_of_business AND s.record_key = f.record_key AND s.system_column = f.system_column
# MAGIC   LEFT JOIN (SELECT DISTINCT line_of_business, record_key FROM sor_values) rk ON rk.line_of_business = f.line_of_business AND rk.record_key = f.record_key
# MAGIC ),
# MAGIC compared AS (
# MAGIC   SELECT *,
# MAGIC     CASE
# MAGIC       WHEN NOT record_found                         THEN 'NO_SYSTEM_RECORD'
# MAGIC       WHEN nullif(trim(field_value), '') IS NULL    THEN 'NOT_ON_DOCUMENT'
# MAGIC       WHEN CASE compare_type
# MAGIC              WHEN 'name'   THEN norm_name(field_value) = norm_name(system_value)
# MAGIC              WHEN 'text'   THEN norm_text(field_value) = norm_text(system_value)
# MAGIC              WHEN 'id'     THEN norm_text(field_value) = norm_text(system_value)
# MAGIC              WHEN 'last4'  THEN right(regexp_replace(field_value, '[^0-9]', ''), 4) = right(regexp_replace(system_value, '[^0-9]', ''), 4)
# MAGIC              WHEN 'number' THEN abs(to_amount(field_value) - to_amount(system_value)) < 0.001
# MAGIC              WHEN 'amount' THEN abs(to_amount(field_value) - to_amount(system_value)) <= abs(to_amount(system_value)) * amount_tolerance_pct / 100
# MAGIC              WHEN 'amount_not_above' THEN to_amount(field_value) <= to_amount(system_value) * (1 + amount_tolerance_pct / 100)
# MAGIC              WHEN 'date'   THEN to_date_any(field_value) = to_date_any(system_value)
# MAGIC              WHEN 'date_in_period' THEN to_date_any(field_value) BETWEEN to_date(split(system_value, ' to ')[0]) AND to_date(split(system_value, ' to ')[1])
# MAGIC            END                                      THEN 'MATCH'
# MAGIC       ELSE 'MISMATCH'
# MAGIC     END AS status,
# MAGIC     CASE WHEN compare_type IN ('amount', 'amount_not_above')
# MAGIC          THEN round((to_amount(field_value) - to_amount(system_value)) / nullif(to_amount(system_value), 0) * 100, 2) END AS pct_diff
# MAGIC   FROM joined
# MAGIC ),
# MAGIC missing AS (   -- required documents that aren't in the packet
# MAGIC   SELECT CAST(NULL AS STRING) AS document_id, k.packet_id, k.received_date, k.line_of_business, k.record_key, q.doc_type,
# MAGIC          CAST(NULL AS STRING) AS field_name, CAST(NULL AS STRING) AS field_value, CAST(NULL AS DOUBLE) AS confidence,
# MAGIC          CAST(NULL AS STRING) AS system_column, 'required_document' AS compare_type, 'Required document is in the packet' AS rule_description,
# MAGIC          CAST(NULL AS STRING) AS system_value, true AS record_found, 'MISSING_DOCUMENT' AS status, CAST(NULL AS DOUBLE) AS pct_diff
# MAGIC   FROM packet_index k
# MAGIC   JOIN required_documents q ON q.line_of_business = k.line_of_business
# MAGIC   LEFT ANTI JOIN packet_documents d ON d.packet_id = k.packet_id AND d.doc_type = q.doc_type
# MAGIC )
# MAGIC SELECT concat_ws(':', coalesce(document_id, packet_id), doc_type, coalesce(field_name, 'document'), coalesce(system_column, 'required')) AS exception_id, *,
# MAGIC        -- the facts ai_decide will see in step 5 (as JSON)
# MAGIC        to_json(named_struct(
# MAGIC          'line_of_business', line_of_business, 'document_type', doc_type, 'field', field_name, 'check', rule_description,
# MAGIC          'result', status, 'value_on_document', field_value, 'value_in_system_of_record', system_value,
# MAGIC          'percent_difference', pct_diff, 'extraction_confidence', confidence)) AS decision_state
# MAGIC FROM (SELECT * FROM compared UNION ALL SELECT * FROM missing);
# MAGIC
# MAGIC SELECT line_of_business, status, count(*) AS checks
# MAGIC FROM recon_results GROUP BY ALL ORDER BY line_of_business, status;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Strict rules flag real problems — and some that only look like problems (nicknames, legal-name variants, Ave vs Avenue).
# MAGIC -- Sorting the two apart is step 5's job.
# MAGIC SELECT packet_id, line_of_business, record_key, doc_type, field_name, field_value AS on_document, system_value AS in_system,
# MAGIC        pct_diff, status, rule_description
# MAGIC FROM recon_results
# MAGIC WHERE status <> 'MATCH'
# MAGIC ORDER BY line_of_business, packet_id, doc_type;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7.4 💬 Ask Genie — mismatches
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Which document types and fields have the most mismatches against the system of record?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7.5 🛠 Try it yourself
# MAGIC Set **`amount_tolerance_pct`** to `5`, re-run **S.2 and S.3**, then re-run **steps 7.2–7.3** (no AI cost — it's just SQL). Which mismatches disappear? Would your credit policy accept that tolerance for W-2 income? Set it back to `1` when you're done.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Decide what to do with each exception — `ai_decide`
# MAGIC
# MAGIC Rules can tell you two values differ. They can't tell you whether *"Jon E Novak"* and *"Jonathan E. Novak"* are the same borrower, or whether a **changed bank account on wire instructions** is a typo or a **business-email-compromise** attempt. That judgment is what `ai_decide` is for.
# MAGIC
# MAGIC You give it the **facts** of each exception (the *state*, as JSON) and your **policy as questions** — and it returns **calibrated probabilities**, not just a label:
# MAGIC
# MAGIC | Question | Type | Returns |
# MAGIC |---|---|---|
# MAGIC | `action` — auto-clear, analyst review, or **hold** the funding / wire / payment? | `choice` | the chosen action, a probability for each action, confidence |
# MAGIC | `fraud_signal` — could this be fraud or tampering? | `noul` (yes / no) | probability 0–1 |
# MAGIC | `materiality` — how much money is at stake if it's wrong? | `score` (4-point scale) | weighted score 0–3 |
# MAGIC
# MAGIC Decisions are cached by a **hash of the facts**: an exception is only sent to `ai_decide` again if its facts change. **8.2–8.4** roll the verdicts up into a packet decision (**straight-through / review / hold**) and a work queue; **8.5–8.6** score them against the answer key.
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the preview: profile menu → **Previews** → enable **AI Decide** (`ai_decide`, Beta). Then re-run the failed cell.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ai_decide output, cached by the hash of the facts it was given (incremental, like the Job's AI tables)
# MAGIC CREATE TABLE IF NOT EXISTS exception_decisions (
# MAGIC   state_hash STRING, exception_id STRING, packet_id STRING, state STRING COMMENT 'The facts ai_decide was given (JSON)',
# MAGIC   decision VARIANT COMMENT 'ai_decide output', decided_at TIMESTAMP
# MAGIC ) COMMENT 'ai_decide output per exception, cached by the hash of its facts (incremental)';
# MAGIC
# MAGIC -- The state = the facts of the exception, built in step 7.2 (decision_state), e.g.
# MAGIC -- {"line_of_business":"Investments","document_type":"wire_instructions","field":"account_last4","check":"Account = account on file",
# MAGIC --  "result":"MISMATCH","value_on_document":"5531","value_in_system_of_record":"4417", ...}
# MAGIC INSERT INTO exception_decisions
# MAGIC WITH new_exceptions AS (
# MAGIC   SELECT sha2(decision_state, 256) AS state_hash, any_value(exception_id) AS exception_id, any_value(packet_id) AS packet_id,
# MAGIC          decision_state AS state
# MAGIC   FROM recon_results r
# MAGIC   WHERE r.status <> 'MATCH'
# MAGIC     AND NOT EXISTS (SELECT 1 FROM exception_decisions d WHERE d.state_hash = sha2(r.decision_state, 256))   -- only facts we haven't decided before
# MAGIC   GROUP BY decision_state
# MAGIC )
# MAGIC SELECT state_hash, exception_id, packet_id, state,
# MAGIC        ai_decide(state,
# MAGIC          '{
# MAGIC            "action": {
# MAGIC              "type": "choice",
# MAGIC              "instructions": "You are a quality-control reviewer in a bank, fund administrator or insurer. A document in a loan file, capital call package or claim packet was reconciled against the system of record and did not match. What should happen to this exception?",
# MAGIC              "criteria": {
# MAGIC                "auto_clear":     "Same underlying fact written differently: nickname or legal-name variant of the same person or entity, abbreviation (Ave vs Avenue), ZIP+4, punctuation or rounding. No correction needed.",
# MAGIC                "analyst_review": "A real discrepancy a processor must correct or confirm before closing - an amount, income, value, rate, fee or date differs, a required document is missing, or a field could not be found - but no immediate risk of paying or wiring money to the wrong party.",
# MAGIC                "hold":           "Stop funding, the wire or the claim payment until resolved: bank, routing or account details differ from those on file, the vehicle (VIN) or identity does not match, the loss is outside the policy period, or the discrepancy could cause money to be sent to the wrong party."
# MAGIC              }
# MAGIC            },
# MAGIC            "fraud_signal": {
# MAGIC              "type": "noul",
# MAGIC              "instructions": "Could this discrepancy be a sign of fraud, document tampering or social engineering?",
# MAGIC              "criteria": {
# MAGIC                "true":  "Pattern commonly seen in fraud: changed bank or payment instructions, a different vehicle, account or identity, or a loss reported outside the coverage period",
# MAGIC                "false": "Ordinary data-entry, calculation, valuation, timing, rounding or formatting difference, or a missing document"
# MAGIC              }
# MAGIC            },
# MAGIC            "materiality": {
# MAGIC              "type": "score",
# MAGIC              "instructions": "How material is the financial impact if this discrepancy is not corrected?",
# MAGIC              "criteria": ["No financial impact", "Minor and easily corrected", "Affects pricing, fees, reserves or eligibility", "Could cause a wrong disbursement, a loss or a regulatory issue"]
# MAGIC            }
# MAGIC          }',
# MAGIC          map('version', '1.0')) AS decision,
# MAGIC        current_timestamp() AS decided_at
# MAGIC FROM new_exceptions;
# MAGIC
# MAGIC SELECT count(*) AS decisions_total, count_if(decided_at >= current_timestamp() - INTERVAL 1 HOUR) AS decided_this_run,
# MAGIC        count_if(decision:error_message::STRING IS NOT NULL) AS errors
# MAGIC FROM exception_decisions;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE exceptions COMMENT 'Every non-matching recon result with the ai_decide verdict' AS
# MAGIC SELECT r.exception_id, r.packet_id, r.received_date, r.line_of_business, r.record_key, r.document_id, r.doc_type, r.field_name,
# MAGIC        r.field_value, r.system_value, r.pct_diff, r.status, r.rule_description,
# MAGIC        d.decision:response:answers:action:choice::STRING           AS action,
# MAGIC        round(d.decision:response:answers:action:confidence::DOUBLE, 2)        AS action_confidence,
# MAGIC        round(d.decision:response:answers:fraud_signal:probability::DOUBLE, 2) AS fraud_probability,
# MAGIC        round(d.decision:response:answers:materiality:score::DOUBLE, 2)        AS materiality_score,
# MAGIC        d.decision:error_message::STRING                            AS decision_error
# MAGIC FROM recon_results r
# MAGIC JOIN (SELECT state_hash, any_value(decision) AS decision FROM exception_decisions GROUP BY state_hash) d
# MAGIC   ON d.state_hash = sha2(r.decision_state, 256)   -- same facts -> same (cached) decision
# MAGIC WHERE r.status <> 'MATCH';
# MAGIC
# MAGIC SELECT line_of_business, doc_type, coalesce(field_name, '(missing document)') AS field, on_document, in_system,
# MAGIC        action, action_confidence, fraud_probability, materiality_score
# MAGIC FROM (SELECT *, field_value AS on_document, system_value AS in_system FROM exceptions)
# MAGIC ORDER BY CASE action WHEN 'hold' THEN 1 WHEN 'analyst_review' THEN 2 ELSE 3 END, fraud_probability DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE packet_status COMMENT 'One row per packet: STRAIGHT_THROUGH / REVIEW / HOLD with reasons and exposure' AS
# MAGIC WITH checks AS (
# MAGIC   SELECT packet_id, count(*) AS checks, count_if(status = 'MATCH') AS matched FROM recon_results GROUP BY packet_id
# MAGIC ),
# MAGIC ex AS (
# MAGIC   SELECT packet_id,
# MAGIC          count(*)                         AS exceptions,
# MAGIC          count_if(action = 'auto_clear')     AS auto_cleared,
# MAGIC          count_if(action = 'analyst_review') AS for_review,
# MAGIC          count_if(action = 'hold')           AS holds,
# MAGIC          max(fraud_probability)              AS max_fraud_probability,
# MAGIC          array_join(array_distinct(collect_list(CASE WHEN action <> 'auto_clear' THEN
# MAGIC            upper(action) || ': ' || doc_type || IF(field_name IS NULL, ' missing', '.' || field_name || ' (doc ' || coalesce(field_value, '-') || ' vs system ' || coalesce(system_value, '-') || ')') END)), '; ') AS open_reasons
# MAGIC   FROM exceptions GROUP BY packet_id
# MAGIC ),
# MAGIC exposure AS (
# MAGIC   SELECT line_of_business, record_key, to_amount(system_value) AS exposure_usd
# MAGIC   FROM sor_values WHERE system_column IN ('loan_amount', 'call_amount', 'reserve_amount')
# MAGIC )
# MAGIC SELECT k.packet_id, k.received_date, k.line_of_business, k.record_key, k.documents, c.checks, c.matched,
# MAGIC        coalesce(e.exceptions, 0) AS exceptions, coalesce(e.auto_cleared, 0) AS auto_cleared,
# MAGIC        coalesce(e.for_review, 0) AS for_review, coalesce(e.holds, 0) AS holds,
# MAGIC        CASE WHEN coalesce(e.holds, 0) > 0 THEN 'HOLD' WHEN coalesce(e.for_review, 0) > 0 THEN 'REVIEW' ELSE 'STRAIGHT_THROUGH' END AS packet_decision,
# MAGIC        coalesce(e.max_fraud_probability, 0) AS max_fraud_probability,
# MAGIC        x.exposure_usd, e.open_reasons
# MAGIC FROM packet_index k
# MAGIC LEFT JOIN checks c   USING (packet_id)
# MAGIC LEFT JOIN ex e       USING (packet_id)
# MAGIC LEFT JOIN exposure x ON x.line_of_business = k.line_of_business AND x.record_key = k.record_key;
# MAGIC
# MAGIC SELECT line_of_business, count(*) AS packets,
# MAGIC        count_if(packet_decision = 'STRAIGHT_THROUGH') AS straight_through,
# MAGIC        count_if(packet_decision = 'REVIEW') AS review, count_if(packet_decision = 'HOLD') AS hold,
# MAGIC        round(100 * avg(CAST(packet_decision = 'STRAIGHT_THROUGH' AS INT)), 0) AS straight_through_pct,
# MAGIC        sum(auto_cleared) AS false_alarms_auto_cleared,
# MAGIC        format_number(sum(CASE WHEN packet_decision = 'HOLD' THEN exposure_usd ELSE 0 END), 0) AS usd_on_hold
# MAGIC FROM packet_status GROUP BY ALL ORDER BY line_of_business;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT packet_decision, packet_id, received_date, line_of_business, record_key,
# MAGIC        format_number(exposure_usd, 0) AS exposure_usd, round(max_fraud_probability, 2) AS fraud_probability,
# MAGIC        open_reasons,
# MAGIC        CASE WHEN packet_decision = 'HOLD' AND line_of_business = 'Investments' THEN 'Call investor services on the number ON FILE to verify wire instructions; do not release the wire'
# MAGIC             WHEN packet_decision = 'HOLD' AND line_of_business = 'Claims'      THEN 'Refer to SIU / coverage review before any payment'
# MAGIC             WHEN packet_decision = 'HOLD'                                      THEN 'Suspend funding; escalate to underwriting manager'
# MAGIC             WHEN line_of_business = 'Lending'     THEN 'Underwriter: condition the file / request corrected document'
# MAGIC             WHEN line_of_business = 'Investments' THEN 'Investor services: confirm with fund accounting'
# MAGIC             ELSE 'Adjuster: update reserve / confirm facts of loss' END AS next_step
# MAGIC FROM packet_status
# MAGIC WHERE packet_decision <> 'STRAIGHT_THROUGH'
# MAGIC ORDER BY CASE packet_decision WHEN 'HOLD' THEN 1 ELSE 2 END, exposure_usd DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Synthetic data only: did we catch every planted problem, and did we wave through the harmless look-alikes?
# MAGIC WITH k AS (SELECT * FROM answer_key_issues WHERE packet_id IN (SELECT packet_id FROM packet_status)),
# MAGIC matched AS (
# MAGIC   SELECT k.*, e.action
# MAGIC   FROM k LEFT JOIN exceptions e
# MAGIC     ON e.packet_id = k.packet_id AND e.doc_type = k.doc_type AND (k.field_name IS NULL OR e.field_name = k.field_name)
# MAGIC ),
# MAGIC per_issue AS (
# MAGIC   SELECT packet_id, issue, expected_action,
# MAGIC          CASE max(CASE action WHEN 'hold' THEN 3 WHEN 'analyst_review' THEN 2 WHEN 'auto_clear' THEN 1 ELSE 0 END)
# MAGIC               WHEN 3 THEN 'hold' WHEN 2 THEN 'analyst_review' WHEN 1 THEN 'auto_clear' ELSE 'not flagged' END AS ai_action
# MAGIC   FROM matched GROUP BY ALL
# MAGIC )
# MAGIC SELECT issue, expected_action, count(*) AS planted,
# MAGIC        count_if(CASE expected_action WHEN 'hold'       THEN ai_action = 'hold'
# MAGIC                                      WHEN 'review'     THEN ai_action IN ('analyst_review', 'hold')
# MAGIC                                      WHEN 'auto_clear' THEN ai_action IN ('auto_clear', 'not flagged') END) AS handled_correctly,
# MAGIC        array_join(collect_list(ai_action), ', ') AS ai_actions
# MAGIC FROM per_issue
# MAGIC GROUP BY ALL
# MAGIC ORDER BY CASE expected_action WHEN 'hold' THEN 1 WHEN 'review' THEN 2 ELSE 3 END, issue;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT e.packet_id, e.doc_type, e.field_name, e.field_value, e.system_value, e.action, e.action_confidence
# MAGIC FROM exceptions e
# MAGIC LEFT ANTI JOIN answer_key_issues k
# MAGIC   ON k.packet_id = e.packet_id AND k.doc_type = e.doc_type AND (k.field_name IS NULL OR k.field_name = e.field_name)
# MAGIC WHERE e.action <> 'auto_clear'
# MAGIC ORDER BY e.packet_id;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8.7 💬 Ask Genie — holds
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Which packets are on hold and why?", before=("The Job tasks plus sections 7–8 (parse → classify → extract → reconcile → decide)", 400))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8.8 🛠 Try it yourself
# MAGIC Your **credit / claims policy is the prompt.** In **step 8.1**, edit the `hold` criterion — for example add *"or a W-2 shows income more than 10% below the stated income"* — then re-run **8.1–8.6**. Because the questions changed but the *facts* didn't, first clear the cache with `DELETE FROM exception_decisions`. Does the W-2 issue move from REVIEW to HOLD?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Executive summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT area, metric, value FROM (
# MAGIC             SELECT 1 AS n, 'Throughput' AS area, 'Packets processed' AS metric, format_number(count(*), 0) AS value FROM packet_status
# MAGIC   UNION ALL SELECT 2, 'Throughput', 'Pages read by ai_parse_document', format_number(count(*), 0) FROM packet_pages
# MAGIC   UNION ALL SELECT 3, 'Throughput', 'Documents split & classified', format_number(count(*), 0) FROM packet_documents WHERE doc_family <> 'Other'
# MAGIC   UNION ALL SELECT 4, 'Throughput', 'Fields extracted', format_number(count(*), 0) FROM extracted_fields
# MAGIC   UNION ALL SELECT 5, 'Throughput', 'Fields reconciled to a system of record', format_number(count(*), 0) FROM recon_results WHERE status <> 'MISSING_DOCUMENT'
# MAGIC   UNION ALL SELECT 6, 'Automation', 'Straight-through packets (no human touch)',
# MAGIC          count_if(packet_decision = 'STRAIGHT_THROUGH') || ' of ' || count(*) || ' (' || CAST(round(100 * avg(CAST(packet_decision = 'STRAIGHT_THROUGH' AS INT))) AS INT) || '%)' FROM packet_status
# MAGIC   UNION ALL SELECT 7, 'Automation', 'Rule mismatches auto-cleared by ai_decide (false alarms avoided)', format_number(count_if(action = 'auto_clear'), 0) FROM exceptions
# MAGIC   UNION ALL SELECT 8, 'Automation', 'Analyst hours saved (illustrative: 30 min per straight-through packet, 5 min per auto-cleared exception)',
# MAGIC          format_number((SELECT count_if(packet_decision = 'STRAIGHT_THROUGH') FROM packet_status) * 0.5 + (SELECT count_if(action = 'auto_clear') FROM exceptions) * 5 / 60.0, 1)
# MAGIC   UNION ALL SELECT 9, 'Risk', 'Packets on HOLD', format_number(count_if(packet_decision = 'HOLD'), 0) FROM packet_status
# MAGIC   UNION ALL SELECT 10, 'Risk', 'Exposure on HOLD (USD)', format_number(sum(CASE WHEN packet_decision = 'HOLD' THEN exposure_usd ELSE 0 END), 0) FROM packet_status
# MAGIC   UNION ALL SELECT 11, 'Risk', 'Exceptions with a fraud signal (probability > 0.5)', format_number(count_if(fraud_probability > 0.5), 0) FROM exceptions
# MAGIC   UNION ALL SELECT 12, 'Risk', 'Packets missing a required document', format_number(count(DISTINCT packet_id), 0) FROM recon_results WHERE status = 'MISSING_DOCUMENT'
# MAGIC ) ORDER BY n;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.2 Everything you built, in Unity Catalog
# MAGIC Every table below — written by a Job task or by this notebook — has **lineage** back to the PDFs in the inbox volume — click *lineage* on `packet_status` to trace a HOLD decision all the way back to the document that caused it.

# COMMAND ----------

show_uc_cards(title="Document intelligence pipeline — from inbox to decision")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Your turn — ask anything, then build more
# MAGIC **Ask your own question:** pick one in the **`question`** widget — or type anything in **`custom_question`** — and re-run **step 10.1**. Click **Continue this conversation in Genie** to drill down with follow-ups and charts.

# COMMAND ----------

question = dbutils.widgets.get("custom_question").strip() or dbutils.widgets.get("question")
ask(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10.2 🚀 Build more — ideas to take this further
# MAGIC | Idea | How | Effort |
# MAGIC |---|---|---|
# MAGIC | **Add a document type** (closing disclosure, flood certificate, K-1, side letter, total-loss valuation) | Add a label in step 1.5, a schema in step 1.7, rules in step 7.1 — then re-run 2.1 | 20 min |
# MAGIC | **Show the evidence** — highlight *where* on the page a value came from | `enable_citations` on the `extract_*` tasks + the bounding boxes in `parsed_packets` | 30 min |
# MAGIC | **Write the condition letter / adjuster note** for each REVIEW packet | `ai_gen` or `ai_query` over `packet_status.open_reasons` | 15 min |
# MAGIC | **Process packets minutes after they land** | Job **file-arrival trigger** on the `inbox` volume (step 6.4) | 5 min |
# MAGIC | **Extend the DAG** with reconciliation and decisions | Turn the SQL of steps 7.1–8.3 into SQL tasks after `assemble_extractions` (same `define_sql_task` pattern) | 30 min |
# MAGIC | **Harder forms** | Set `mode: MODE_PRECISION` on an `extract_*` task for complex fields and long documents | 2 min |
# MAGIC # MAGIC | **Bundle it** | Define the same Job in YAML with **Declarative Automation Bundles** (`ai_function_task` / `sql_task`) to version and promote it across workspaces | 1 hr |
# MAGIC | **Human-in-the-loop review screen** | **Databricks App** over `packet_status` + page images, writing reviewer verdicts back to a Delta table | 1 day |
# MAGIC | **Ops dashboard** | **AI/BI dashboard** on `packet_status` and `exceptions` (STP rate, holds, aging), with Genie built in | 30 min |
# MAGIC | **Share Genie** with the business | Open the Genie agent → **Share** → add your underwriting, investor-services and claims groups | 2 min |
# MAGIC | **Your real documents** | Drop PDFs into the inbox volume and point the views in *Bring your own documents* (step 0.11) at your LOS, fund-admin and claims tables | 1 day |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. What you just saw — and where to take it next
# MAGIC
# MAGIC * **A real document intelligence workflow, not a script.** `ai_parse_document`, `ai_classify` and `ai_extract` run as **native AI Function tasks in a Lakeflow Job DAG** — two-level classification, parallel extraction per line of business, incremental processing, retries and repair runs per task, row-level success counts, a schedule — with a few SQL tasks as glue, all governed by Unity Catalog.
# MAGIC * **Then judgment on top of the workflow's output.** Cheap SQL rules reconcile every field against the systems of record, and `ai_decide` applies your policy to every exception — separating *"Jon vs Jonathan"* from *"the wire account changed"*, so analysts only see what matters.
# MAGIC * **Backfill and daily are the same DAG.** Incremental AI tables mean you pay for each page, document and decision exactly once; the daily run processed only the new packets.
# MAGIC * **Genie turns the results into self-service.** Underwriters, investor-services analysts and claims supervisors ask in plain English; Genie writes the SQL against the same governed tables.
# MAGIC
# MAGIC **Production path**
# MAGIC 1. **Land real documents** — scanner, email, SFTP or core-system exports into the `inbox` volume (or any cloud storage path governed by Unity Catalog).
# MAGIC 2. **Point at your systems of record** — LOS, fund administration and claims tables via Lakeflow Connect / partner connectors; swap the views in step 0.11.
# MAGIC 3. **Run it** — unpause the Job's schedule or switch to a file-arrival trigger (step 6.4), add failure alerts, and extend the DAG with reconcile / decide tasks (step 10.2).
# MAGIC 4. **Close the loop** — a review app for analysts, their verdicts written back to Delta, and **MLflow evaluation** on a labelled sample to track extraction and decision accuracy over time.
# MAGIC 5. **Graduate where needed** — **Agent Bricks Information Extraction** for harder forms, or custom models with Model Serving.
# MAGIC
# MAGIC **Docs:** [ai_parse_document](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document) · [ai_classify](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_classify) · [ai_extract](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_extract) · [ai_decide](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_decide) · [AI Functions overview](https://docs.databricks.com/aws/en/large-language-models/ai-functions) · [Lakeflow Jobs](https://docs.databricks.com/aws/en/jobs/) · [Genie](https://docs.databricks.com/aws/en/genie/)
# MAGIC
# MAGIC <sub>`ai_decide` is in Beta. Review your organization's compliance requirements (e.g. GLBA, NYDFS Part 500, SOC 2, model-risk management / SR 11-7) before using AI Functions on customer data, and keep a human in the loop for credit, investment and claim decisions. All data in this notebook is synthetic.</sub>
