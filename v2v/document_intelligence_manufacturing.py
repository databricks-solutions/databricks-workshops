# Databricks notebook source

# MAGIC %md
# MAGIC # Order, Invoice & Engineering Document Intelligence for Manufacturing
# MAGIC **For manufacturers:** turn the purchase orders, invoices, product data sheets and quality / engineering documents that land in shared inboxes every day into clean, validated data — line items, prices, specs and measurements — and flag the exceptions before they cost you. Then let anyone ask Genie. Powered by Databricks AI Functions.
# MAGIC
# MAGIC **Industry:** Manufacturing — Order management (customer POs), Procure-to-pay (supplier invoices), Quality & Engineering (data sheets, certificates of conformance, inspection reports, engineering change notices)
# MAGIC
# MAGIC Every plant has the same shared inboxes — `orders@`, `ap@`, `quality@`, `engineering@` — and a team retyping attachments into the ERP: *Which emails need action? What's on this PO? Does the invoice match what we ordered and received? Is the supplier's part in spec? Did engineering just obsolete the part the customer ordered?*
# MAGIC This notebook answers all of them **with no OCR templates, ML models or endpoints to manage** — using Databricks AI Functions. You design each AI step on a few documents, then run them as native **AI Function tasks** in a **Lakeflow Job**: a **one-time backfill** of the last 45 days, then a **daily run** that processes only new mail. Finally the results go to the business through **Genie**, so buyers, AP clerks and quality engineers can ask in plain English.
# MAGIC
# MAGIC | Part | Section | Question | Databricks capability |
# MAGIC |---|---|---|---|
# MAGIC | **1 · Design** | 1 | Which emails need action — and which workflow does each belong to? | `ai_classify` |
# MAGIC | | 2 | What's inside each attachment (text, tables, layout) — and what kind of document is it? | `ai_parse_document`, `ai_classify` |
# MAGIC | | 3 | What are the line items, prices, specs and measurements? | `ai_extract` (with a confidence score per field) |
# MAGIC | **2 · Build & run** | 4 | How do these run in production — once for history, then every day? | **Lakeflow Job** with native **AI Function tasks** (parse · classify · extract) + notebook tasks; one-time backfill |
# MAGIC | **3 · Explore & ask** | 5–6 | What did the backfill find in the inbox, the documents and their line items? | the job's output tables |
# MAGIC | | 7 | Which documents don't match our POs, receipts, prices, specs or engineering changes? | SQL rules against ERP master data |
# MAGIC | | 8 | What should we do about each exception — and what do we tell the supplier or customer? | `ai_decide`, `ai_gen` |
# MAGIC | | 9 | What happens tomorrow morning? | the daily run — only new mail |
# MAGIC | 💬 | | After every section in Part 3: **ask Genie** the same question in plain English — no SQL | **Genie** agent on Unity Catalog |
# MAGIC | 🛠 | | After every section: a **Try it yourself** exercise, and a *Build more* list at the end | |
# MAGIC
# MAGIC ---
# MAGIC ### How to run
# MAGIC 1. Keep the default **Serverless** compute. Genie runs on a **Serverless or Pro SQL warehouse** in your workspace (the notebook finds one for you).
# MAGIC 2. A **workspace admin** enables these on the **Previews** page (profile menu → *Previews*): **Lakeflow AI Function Task** (Beta — the job's parse / classify / extract tasks), **`ai_decide`** (Beta) and **`ai_gen`** (Public Preview). **If any AI Function cell fails, this is the first thing to check** — ask your admin to turn these previews on.
# MAGIC 3. Run the **Setup** steps (**S.1–S.3**) and set the widgets (`catalog`, `schema`, …) to a location where you can create tables and a volume. Then either **Run all** (about 50 minutes, mostly the job runs in steps 4.2 and 9.2), or — better for learning — run section by section: sections end with a **🛠 Try it yourself** exercise, and in Part 3 with a **💬 Ask Genie** cell. *(Step 0.9 starts parsing in the background; Run all then builds the full job — schedule paused — and runs it twice: the backfill in step 4.2 and a daily run in step 9.2.)*
# MAGIC 4. **Every section and cell is numbered** (S.1, 0.1, 1.1 …) — look for the number in each cell's title.
# MAGIC 5. **New to some of the code?** Ask **Genie Code** to explain any cell: click into it, open Genie Code and type `/explain` (or ask *"what does this cell do?"*).
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → **Lakeflow AI Function Task** and the **AI Functions** previews, including **`ai_decide`** (Beta) and **`ai_gen`**. Then re-run the failed cell.
# MAGIC
# MAGIC > **No documents? No problem.** Section 0 (steps 0.1–0.2) generates a realistic, fully synthetic inbox: ~85 emails and ~70 PDF attachments from fictitious customers and suppliers, plus a small batch of *today's mail* delivered in step 9.1. **Have your own?** See *"Bring your own data"* (step 0.8) — land your attachments in a volume and every downstream cell keeps working.
# MAGIC
# MAGIC **Terms used in this notebook:** a **volume** is a Unity Catalog folder for files (here, the PDFs) governed like a table · a **session variable** holds a widget value for SQL cells · **VARIANT** is a column type for semi-structured JSON (what the AI Functions return) · **3-way match** compares an invoice with the purchase order and the goods receipt · **CoC** = certificate of conformance (a supplier's quality certificate for a shipped lot) · **ECN** = engineering change notice.
# MAGIC
# MAGIC *All companies, people, part numbers and documents in this notebook are fictitious.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## S. Setup — choose where to write the demo tables
# MAGIC Three short setup steps. The **widgets** (the notebook's parameters) drive everything below.
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | **S.1** | Creates the widgets |
# MAGIC | **S.2** | Turns the widget values into SQL session variables and creates / selects the schema and the `inbox` volume |
# MAGIC | **S.3** | Loads the Python helper functions used later (Unity Catalog links, Genie) |
# MAGIC
# MAGIC | Widget | Meaning | Default |
# MAGIC |---|---|---|
# MAGIC | `catalog` / `schema` | Where tables and the `inbox` volume are created in **Unity Catalog** (created if missing) | your current catalog / `mfg_doc_intel_demo` |
# MAGIC | `price_tolerance_pct` | A price more than this % away from contract / PO price is an exception | `2` |
# MAGIC | `confidence_threshold` | Extracted fields with a confidence score below this go to a human review queue | `0.90` |
# MAGIC | `question` / `custom_question` | What to ask Genie in step 11.1 | starter questions |
# MAGIC | `run_label` | Stamped on every row the notebook tasks process. The job sets it to `one-time backfill` or `daily job` | `notebook` |
# MAGIC
# MAGIC > 🧭 **Where are the widgets?** They show up once you run **S.1**, in one of two places depending on your layout:
# MAGIC > * **A bar across the top** of the notebook (the classic layout), or
# MAGIC > * **A panel on the left-hand side** — click the **Parameters** icon (the parameters symbol in the left sidebar) to open it.
# MAGIC >
# MAGIC > Same widgets either way — use whichever you prefer.
# MAGIC
# MAGIC > 🔁 **Changed a widget?** Re-run **S.2 and S.3**, then the cells you want to refresh (or just **Run all**). With the default widget settings, changing a value does **not** re-run SQL cells, so the new value isn't picked up until S.2 and S.3 run again.
# MAGIC
# MAGIC > **▶ DAILY JOB** — cells whose first line is `▶ DAILY JOB · task: <name>` are the job's **notebook tasks**; S.1 and S.2 (`▶ DAILY JOB · every task`) run at the start of each one. Step 4.1 copies exactly those cells into one small notebook per task. The parse / classify / extract steps are native **AI Function tasks** you define in sections 1–3.

# COMMAND ----------

# ▶ DAILY JOB · every task — this cell runs at the start of every notebook task of the Lakeflow Job built in step 4.1
# Widgets — they appear in the bar at the top of the notebook, or in the left-hand Parameters panel
GENIE_QUESTIONS = [
    "Which supplier invoices should we hold, and why?",
    "What is the total value of customer orders we received by customer?",
    "Which parts failed incoming inspection, and what did the supplier report?",
    "Which customer orders reference obsolete part numbers, and what replaces them?",
    "Where were we billed more than we received or more than our PO price?",
    "What did the daily job process today?",
]
dbutils.widgets.text("catalog", "", "1. Catalog (blank = current)")
dbutils.widgets.text("schema", "mfg_doc_intel_demo", "2. Schema")
dbutils.widgets.dropdown("price_tolerance_pct", "2", ["1", "2", "5"], "3. Price tolerance (%)")
dbutils.widgets.dropdown("confidence_threshold", "0.90", ["0.80", "0.90", "0.95"], "4. Review if confidence below")
dbutils.widgets.dropdown("question", GENIE_QUESTIONS[0], GENIE_QUESTIONS, "5. Ask Genie (any time)")
dbutils.widgets.text("custom_question", "", "6. ...or type your own Genie question")
dbutils.widgets.text("run_label", "notebook", "7. Run label (set by the job)")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ▶ DAILY JOB · every task
# MAGIC -- Turn the widgets into session variables (with safe defaults if a widget is left blank).
# MAGIC -- Changed a widget? Re-run this cell (S.2) and the helpers (S.3), then the cells you want to refresh.
# MAGIC DECLARE OR REPLACE VARIABLE target_catalog       STRING DEFAULT coalesce(nullif(:catalog, ''), current_catalog());
# MAGIC DECLARE OR REPLACE VARIABLE target_schema        STRING DEFAULT coalesce(nullif(:schema, ''), 'mfg_doc_intel_demo');
# MAGIC DECLARE OR REPLACE VARIABLE price_tolerance_pct  DOUBLE DEFAULT CAST(coalesce(nullif(:price_tolerance_pct, ''), '2') AS DOUBLE);
# MAGIC DECLARE OR REPLACE VARIABLE confidence_threshold DOUBLE DEFAULT CAST(coalesce(nullif(:confidence_threshold, ''), '0.90') AS DOUBLE);
# MAGIC DECLARE OR REPLACE VARIABLE run_label            STRING DEFAULT coalesce(nullif(:run_label, ''), 'notebook');
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS IDENTIFIER(target_catalog || '.' || target_schema)
# MAGIC COMMENT 'MFG AI Functions industry play: order, invoice, product-sheet and QA / engineering document extraction (synthetic data)';
# MAGIC
# MAGIC USE IDENTIFIER(target_catalog || '.' || target_schema);
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS inbox
# MAGIC COMMENT 'Landing zone for email attachments: customer POs, supplier invoices, data sheets, CoCs / inspection reports, ECNs';
# MAGIC
# MAGIC SELECT current_catalog() AS catalog, current_schema() AS schema,
# MAGIC        '/Volumes/' || current_catalog() || '/' || current_schema() || '/inbox' AS inbox_volume,
# MAGIC        price_tolerance_pct, confidence_threshold, run_label;

# COMMAND ----------

# MAGIC %md
# MAGIC ### S.3 Load the helper functions
# MAGIC Step S.3 **doesn't create any tables or change any data** — it only defines Python helpers the rest of the notebook calls, so later cells stay short and readable:
# MAGIC
# MAGIC | Helper | Used in | What it does |
# MAGIC |---|---|---|
# MAGIC | `WorkspaceClient()`, `CATALOG`, `SCHEMA`, `INBOX` | everywhere | Connects to this workspace with **your** identity and remembers where the demo tables and the `inbox` volume live (from S.2) |
# MAGIC | `pick_warehouse()` | S.3, used by step 0.5 | Finds a **Serverless or Pro SQL warehouse** for the Genie agent |
# MAGIC | `preview_sql()`, `show_task()`, `JOB_NAME` | step 0.9 and sections 1–4 | Runs an AI Function on a few rows with the same settings as its job task (and shows the SQL), and renders a task definition |
# MAGIC | `uc_url()`, `button()`, `show_uc_cards()` | step 0.4 and later | Builds clickable **Unity Catalog** links and the table / volume cards (schema · sample · lineage) |
# MAGIC | `ask_genie()`, `show_genie_answer()` — and `ask()`, defined with the agent in step 0.5 | every 💬 Ask Genie step | Asks the **Genie agent** a question through the Genie API and renders the answer, the SQL Genie wrote and a *Continue in Genie* link |
# MAGIC
# MAGIC When it finishes it prints the catalog / schema, the inbox volume and the SQL warehouse it will use. **Re-run this cell whenever you change the `catalog` or `schema` widget** (after S.2).
# MAGIC
# MAGIC > 💡 **Don't fully understand a cell?** Ask **Genie Code** to explain it: click into the cell, open Genie Code and type `/explain` — or just ask *"what does this cell do?"* in plain English.

# COMMAND ----------

import json, time, uuid, html, re
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
CATALOG, SCHEMA = spark.sql("SELECT current_catalog(), current_schema()").first()
INBOX = f"/Volumes/{CATALOG}/{SCHEMA}/inbox"
HOST = w.config.host.rstrip("/")
try:
    ORG = f"?o={w.get_workspace_id()}"
except Exception:
    ORG = ""

def pick_warehouse():
    # Serverless or Pro SQL warehouses only (skip Classic and real-time / special-purpose warehouses)
    whs = [x for x in w.warehouses.list() if str(x.warehouse_type).endswith("PRO")]
    if not whs:
        raise Exception("Create a Serverless or Pro SQL warehouse — the Genie agent runs on Databricks SQL.")
    whs.sort(key=lambda x: ("RUNNING" not in str(x.state), not x.enable_serverless_compute))
    return whs[0]
WAREHOUSE = pick_warehouse()

def uc_url(*parts, tab=None):
    q = ORG + ((("&" if ORG else "?") + f"activeTab={tab}") if tab else "")
    return f"{HOST}/explore/data/" + "/".join(parts) + q

def volume_url():
    return f"{HOST}/explore/data/volumes/{CATALOG}/{SCHEMA}/inbox{ORG}"

FONT = "font-family:DM Sans,Helvetica,Arial,sans-serif"
def button(label, url, color="#FF3621"):
    return (f'<a href="{url}" target="_blank" style="display:inline-block;margin:4px 8px 4px 0;padding:8px 14px;'
            f'background:{color};color:white;border-radius:6px;text-decoration:none;font-weight:600;{FONT};font-size:13px">'
            f'{html.escape(label)} ↗</a>')

# table -> (role shown on the card, description, include in the Genie agent?)
TABLE_INFO = {
    "emails": ("Source", "Shared-mailbox emails: sender, subject, body, attachment path", True),
    "item_master": ("Source", "ERP item master: finished goods and purchased components", True),
    "component_specs": ("Source", "Engineering spec: nominal and tolerance per characteristic", True),
    "customers": ("Source", "Customers, email domains and contract discounts", True),
    "suppliers": ("Source", "Suppliers, email domains and bank account on file", True),
    "purchase_orders": ("Source", "Our purchase orders to suppliers (PO lines)", True),
    "goods_receipts": ("Source", "What we actually received against each PO line", True),
    "email_classifications": ("AI Function task", "classify_emails output: workflow + confidence per email", False),
    "parsed_documents": ("AI Function task", "parse_attachments output: pages, elements, tables per PDF", False),
    "document_classifications": ("AI Function task", "classify_documents output: type + confidence per document", False),
    "extracted_purchase_orders": ("AI Function task", "extract_purchase_orders output", False),
    "extracted_invoices": ("AI Function task", "extract_invoices output", False),
    "extracted_data_sheets": ("AI Function task", "extract_data_sheets output", False),
    "extracted_quality_docs": ("AI Function task", "extract_quality_docs output", False),
    "extracted_ecns": ("AI Function task", "extract_ecns output", False),
    "email_routing": ("Insight", "Every email with its workflow", True),
    "documents": ("Insight", "Every parsed attachment with its document type", True),
    "customer_order_lines": ("Insight", "Customer PO line items with confidence", True),
    "supplier_invoice_lines": ("Insight", "Supplier invoice line items with remit-to details", True),
    "product_specs": ("Insight", "Material, temperature rating and compliance per data sheet", True),
    "inspection_results": ("Insight", "Measured min / max per characteristic per lot", True),
    "engineering_changes": ("Insight", "Part supersessions and revisions from ECNs", True),
    "extraction_review_queue": ("Insight", "Extracted values below the confidence threshold", True),
    "document_exceptions": ("Insight", "Every exception found, with $ at risk", True),
    "exception_decisions": ("ai_decide", "Route, hold probability and business impact per exception", True),
    "draft_replies": ("ai_gen", "Drafted email for every exception that needs a person", True),
}
ROLE_COLOR = {"Source": "#5A6F77", "Insight": "#00A972", "AI Function task": "#1B3139"}

def show_uc_cards(roles=None, title=None, volume=False):
    existing = {r.tableName for r in spark.sql(f"SHOW TABLES IN `{CATALOG}`.`{SCHEMA}`").collect()}
    cards = []
    if volume:
        cards.append(f"""<div style="border:1px solid #e0e0e0;border-radius:10px;padding:12px 14px;width:280px;{FONT}">
          <div style="font-size:11px;font-weight:700;color:#FF3621;text-transform:uppercase;letter-spacing:.04em">Volume · unstructured files</div>
          <div style="font-size:15px;font-weight:700;margin:4px 0"><a href="{volume_url()}" target="_blank" style="color:#1B3139">inbox</a></div>
          <div style="font-size:12px;color:#555;min-height:32px">Email attachments (PDF) organized by received date</div>
          <div style="font-size:12px;margin-top:6px"><a href="{volume_url()}" target="_blank">browse &amp; preview the PDFs</a></div></div>""")
    for t, (role, desc, _) in TABLE_INFO.items():
        if t not in existing or (roles and role not in roles):
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
    tables = sorted(set(re.findall(r"\b" + re.escape(SCHEMA) + r"\.`?(\w+)", sql))) if sql else []
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

JOB_NAME = f"Document intelligence pipeline ({CATALOG}.{SCHEMA})"

def preview_sql(sql, **params):
    """Shows the SQL (click to expand) and runs it; :name markers in the SQL are bound to the keyword arguments."""
    displayHTML(f'<details style="{FONT};font-size:13px;margin-bottom:4px"><summary style="cursor:pointer;font-weight:600;color:#1B3139">'
                f'Show the SQL this preview runs</summary><pre style="background:#f6f6f6;padding:10px;border-radius:6px;'
                f'white-space:pre-wrap;font-size:12px">{html.escape(sql.strip())}</pre></details>')
    return spark.sql(sql, args=params)

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

print(f"Writing to {CATALOG}.{SCHEMA} · inbox volume {INBOX} · SQL warehouse for Genie: '{WAREHOUSE.name}'")
displayHTML(button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139") + button("Open the inbox volume", volume_url(), "#5A6F77"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Generate a synthetic inbox *(skip 0.1–0.2 if you bring your own)*
# MAGIC
# MAGIC **Ridgeline Precision Components** (fictitious) makes bearing housings, shafts, gearboxes and fittings, sells to six OEM customers and buys components from six suppliers. Step 0.2 writes 45 days of email to four shared mailboxes — with real **PDF attachments** in a Unity Catalog **volume** — that behave like the real thing:
# MAGIC
# MAGIC * **Customer purchase orders** in three different layouts (classic, SAP-style, compact) — no two customers format a PO the same way, which is why template-based OCR breaks
# MAGIC * **Supplier invoices** (including a bilingual Spanish / English one) against **our purchase orders and goods receipts**
# MAGIC * **Product data sheets**, **certificates of conformance with dimensional inspection reports**, and **engineering change notices**
# MAGIC * **Noise**: newsletters, out-of-office replies, IT notices — and a few emails sent to the **wrong mailbox**, plus order changes written only in the email body
# MAGIC
# MAGIC …and **planted problems** for the AI Functions and rules to find:
# MAGIC
# MAGIC | Planted problem | Where you'll see it |
# MAGIC |---|---|
# MAGIC | Two customer POs priced below contract, one PO for an **obsolete** part, one with a delivery date inside lead time, one PO sent twice | section 7 |
# MAGIC | Invoices 4.5% and 7% over PO price, an invoice for **150 more** units than we received, an invoice sent twice, an invoice with **no PO** | section 7 |
# MAGIC | An invoice from a **look-alike domain** announcing **new bank details** (classic payment fraud) | sections 7–8 |
# MAGIC | A data sheet whose new material is **not RoHS compliant**, one whose temperature rating dropped below our spec | section 7 |
# MAGIC | A lot measured **out of tolerance that the supplier marked PASS**, a lot with no lot number | section 7 |
# MAGIC | An ECN that obsoletes the part a customer later orders — the link only shows up when you read *both* documents | step 7.3 |
# MAGIC
# MAGIC > ⚠️ **Step 0.2 starts fresh:** it empties the `inbox` volume, drops the demo tables in the schema from S.2 and deletes this notebook's job (if it exists), so the backfill in step 4.2 re-processes everything. **Don't run it against a schema that holds your own data** — skip to step 0.8 instead.
# MAGIC
# MAGIC All randomness is seeded, so results are reproducible, and dates are relative to today, so the inbox always looks current. A second batch — **today's mail** — is held back until step 9.1, so the daily run has something new to process.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Item master: what we sell (finished goods) and what we buy (purchased components), as it lives in the ERP
# MAGIC CREATE OR REPLACE TABLE item_master
# MAGIC COMMENT 'ERP item master: finished goods we sell and purchased components we buy' AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('RPC-1001', 'Pillow block bearing housing, 1-1/4 in bore', 'finished_good', 'ACTIVE',   'EA',   48.50, NULL, 10, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1002', 'Flange bearing unit, 2-bolt, 1 in',           'finished_good', 'ACTIVE',   'EA',   36.20, NULL, 10, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1003', 'Precision drive shaft, 25 mm x 600 mm, 4140', 'finished_good', 'ACTIVE',   'EA',  112.00, NULL, 21, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1004', 'Planetary gearbox, 10:1, NEMA 23',            'finished_good', 'ACTIVE',   'EA',  389.00, NULL, 28, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1005', 'Hydraulic fitting kit, JIC-8',                'finished_good', 'ACTIVE',   'KIT',  22.75, NULL,  7, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1006', 'Linear guide rail, 20 mm x 1000 mm',          'finished_good', 'ACTIVE',   'EA',  141.00, NULL, 14, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1007', 'Timing pulley, HTD 5M, 30 teeth',             'finished_good', 'ACTIVE',   'EA',   18.40, NULL,  7, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1008', 'Jaw coupling, 19 mm',                         'finished_good', 'ACTIVE',   'EA',   27.90, NULL,  7, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1009', 'Gearbox output shaft, rev B',                 'finished_good', 'OBSOLETE', 'EA',   64.00, NULL, 14, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('RPC-1012', 'Gearbox output shaft, rev C (hardened)',      'finished_good', 'ACTIVE',   'EA',   68.50, NULL, 14, NULL,     NULL,             NULL,                    NULL,  NULL),
# MAGIC   ('CMP-2001', 'Deep-groove ball bearing 6205-2RS',           'purchased_component', 'ACTIVE', 'EA', NULL,  4.10, 21, 'SUP-01', '6205-2RS',       '52100 chrome steel',    TRUE,  120),
# MAGIC   ('CMP-2007', 'Deep-groove ball bearing 6204-2RS',           'purchased_component', 'ACTIVE', 'EA', NULL,  3.35, 21, 'SUP-01', '6204-2RS',       '52100 chrome steel',    TRUE,  120),
# MAGIC   ('CMP-2002', 'Shaft blank, AISI 4140, 25 mm x 610 mm',      'purchased_component', 'ACTIVE', 'EA', NULL, 19.80, 28, 'SUP-02', 'GL-4140-25-610', 'AISI 4140 Q&T',         FALSE, NULL),
# MAGIC   ('CMP-2003', 'Housing casting, ductile iron 65-45-12',      'purchased_component', 'ACTIVE', 'EA', NULL, 11.40, 35, 'SUP-03', 'MC-HSG-114',     'Ductile iron 65-45-12', TRUE,  NULL),
# MAGIC   ('CMP-2004', 'Planetary gear set 10:1',                     'purchased_component', 'ACTIVE', 'EA', NULL, 96.00, 42, 'SUP-04', 'PGW-PS10-23',    '8620 carburized',       TRUE,  100),
# MAGIC   ('CMP-2005', 'O-ring kit, FKM (Viton), JIC-8',              'purchased_component', 'ACTIVE', 'KIT', NULL, 2.15, 14, 'SUP-05', 'ST-ORK-J8-FKM',  'FKM 75 Shore A',        TRUE,  200),
# MAGIC   ('CMP-2008', 'Seal kit, NBR, JIC-8',                        'purchased_component', 'ACTIVE', 'KIT', NULL, 1.10, 14, 'SUP-05', 'ST-SK-J8-NBR',   'NBR 70 Shore A',        TRUE,  100),
# MAGIC   ('CMP-2006', 'Pulley blank, aluminum 6061-T6',              'purchased_component', 'ACTIVE', 'EA', NULL,  3.60, 30, 'SUP-06', 'AEM-PB-6061-30', 'Aluminum 6061-T6',      TRUE,  NULL)
# MAGIC AS t(part_number, description, item_type, status, uom, list_price, unit_cost, lead_time_days, supplier_id, supplier_item_code,
# MAGIC      material_spec, rohs_required, min_temp_rating_c);
# MAGIC
# MAGIC -- Our engineering specification for each purchased component (what incoming inspection must meet)
# MAGIC CREATE OR REPLACE TABLE component_specs
# MAGIC COMMENT 'Engineering specification: nominal and +/- tolerance (mm) per characteristic of each purchased component' AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('CMP-2001', 'Bore diameter', 25.000, 0.008), ('CMP-2001', 'Outside diameter', 52.000, 0.011), ('CMP-2001', 'Width', 15.000, 0.060),
# MAGIC   ('CMP-2007', 'Bore diameter', 20.000, 0.008), ('CMP-2007', 'Outside diameter', 47.000, 0.011), ('CMP-2007', 'Width', 14.000, 0.060),
# MAGIC   ('CMP-2002', 'Diameter', 25.000, 0.020), ('CMP-2002', 'Length', 610.000, 0.500),
# MAGIC   ('CMP-2003', 'Bore diameter', 62.000, 0.025), ('CMP-2003', 'Base width', 38.000, 0.300), ('CMP-2003', 'Center height', 33.300, 0.100),
# MAGIC   ('CMP-2004', 'Output bore', 14.000, 0.012), ('CMP-2004', 'Pilot diameter', 38.000, 0.020),
# MAGIC   ('CMP-2005', 'Cross-section', 1.780, 0.080), ('CMP-2005', 'Inside diameter', 17.170, 0.150),
# MAGIC   ('CMP-2006', 'Outside diameter', 47.750, 0.050), ('CMP-2006', 'Bore', 12.000, 0.018)
# MAGIC AS t(part_number, characteristic, nominal_mm, tol_mm);
# MAGIC
# MAGIC -- Customers (who send us purchase orders) and their contract discount off list price
# MAGIC CREATE OR REPLACE TABLE customers
# MAGIC COMMENT 'Customer master: email domain (used to identify the sender), contract discount off list price, terms' AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('CUS-01', 'Atlas Ag Equipment',          'atlasag.com',        8.0, 'Net 45', 'Midwest US'),
# MAGIC   ('CUS-02', 'Corvus Packaging Systems',    'corvuspack.com',     5.0, 'Net 30', 'Northeast US'),
# MAGIC   ('CUS-03', 'Helix Robotics',              'helixrobotics.io',  10.0, 'Net 60', 'West US'),
# MAGIC   ('CUS-04', 'Summit Conveyor Co.',         'summitconveyor.com', 3.0, 'Net 30', 'Southeast US'),
# MAGIC   ('CUS-05', 'Bluewater Marine Hydraulics', 'bluewatermh.com',    0.0, 'Net 30', 'Gulf Coast US'),
# MAGIC   ('CUS-06', 'Pioneer Food Machinery',      'pioneerfm.com',      6.0, 'Net 45', 'Ontario CA')
# MAGIC AS t(customer_id, customer_name, email_domain, discount_pct, payment_terms, region);
# MAGIC
# MAGIC -- Suppliers (who send us invoices, data sheets and quality certificates) with the bank account on file
# MAGIC CREATE OR REPLACE TABLE suppliers
# MAGIC COMMENT 'Supplier master: email domain and bank account (last 4 digits) on file, terms' AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('SUP-01', 'Keystone Bearing Supply',          'keystonebearing.com', '4471', 'Net 45', 'US'),
# MAGIC   ('SUP-02', 'Great Lakes Steel & Bar',          'glsteelbar.com',      '8820', 'Net 30', 'US'),
# MAGIC   ('SUP-03', 'Midwest Castings Co.',             'midwestcastings.com', '3105', 'Net 45', 'US'),
# MAGIC   ('SUP-04', 'Precision Gear Works',             'pgw-gears.com',       '6642', 'Net 60', 'US'),
# MAGIC   ('SUP-05', 'SealTech Industries',              'sealtech-ind.com',    '1958', 'Net 30', 'US'),
# MAGIC   ('SUP-06', 'Aluminios Extruidos de Monterrey', 'aluextmx.com.mx',     '2290', 'Net 60', 'MX')
# MAGIC AS t(supplier_id, supplier_name, email_domain, bank_account_last4, payment_terms, country);

# COMMAND ----------

# Synthetic data generator. It writes real PDF files to the inbox volume and one row per email to the `emails` table,
# plus our purchase orders and goods receipts (what the 3-way match in section 7 compares invoices against).
# Nothing here uses AI — it only creates realistic test documents. Skip it if you bring your own documents (step 0.8).
from pyspark.sql import types as T
import os, datetime as dt

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

import random, datetime as dt, hashlib

OUR = "Ridgeline Precision Components"
OUR_ADDR = "800 Forge Road, Dayton, OH 45414"
MAILBOX = {"orders": "orders@ridgelinepc.com", "ap": "ap@ridgelinepc.com",
           "quality": "quality@ridgelinepc.com", "engineering": "engineering@ridgelinepc.com"}
NAVY, GRAY, WHITE = (0.11, 0.19, 0.22), (0.45, 0.45, 0.45), (1, 1, 1)
money = lambda v: f"{v:,.2f}"
qty_s = lambda v: f"{v:,.0f}"


def build_inbox(items, customers, suppliers, specs, today=None, history_days=45, seed=2026):
    """Deterministically builds a synthetic shared-inbox history plus 'today's mail'.
    Returns dict(emails=[...], files={relative_path: bytes}, purchase_orders=[...], goods_receipts=[...]).
    Every email has batch = 'history' or 'today'."""
    today = today or dt.date.today()
    rng = random.Random(seed)
    item = {i["part_number"]: i for i in items}
    cust = {c["customer_id"]: c for c in customers}
    sup = {s["supplier_id"]: s for s in suppliers}
    spec = {}
    for s in specs:
        spec.setdefault(s["part_number"], []).append(s)
    emails, files, pos, receipts = [], {}, [], []
    seq = {"history": 0, "today": 0}

    def day(offset):
        return today + dt.timedelta(days=offset)

    def add_email(offset, mailbox, from_name, from_addr, subject, body, attachment=None, hour=None, batch="history"):
        seq[batch] += 1
        d = day(offset)
        h = hour if hour is not None else (rng.choice([0, 1, 2, 3, 4, 5]) if batch == "today" else rng.choice([6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]))
        ts = dt.datetime.combine(d, dt.time(h, rng.randint(0, 59), rng.randint(0, 59)))
        eid = f"EM-{seq[batch]:04d}" if batch == "history" else f"EM-{today:%Y%m%d}-{seq[batch]:02d}"
        e = dict(email_id=eid, received_ts=ts, mailbox=MAILBOX[mailbox], from_name=from_name, from_address=from_addr,
                 subject=subject, body=body, attachment_name=None, attachment_path=None, batch=batch)
        if attachment:
            fname, data = attachment
            rel = f"{d:%Y/%m/%d}/{eid}_{fname}"
            files[rel] = data
            e["attachment_name"], e["attachment_path"] = fname, rel
        emails.append(e)
        return e

    # ------------------------------------------------------------------ renderers
    def header(doc, company, lines, title):
        doc.box(0, 0, 612, 74, fill=NAVY, stroke=None)
        doc.text(40, 32, company.upper(), 15 if len(company) <= 24 else 11.5, True, WHITE)
        for i, l in enumerate(lines):
            doc.text(40, 48 + i * 11, l, 8, False, (0.85, 0.87, 0.9))
        doc.text(572, 44, title, 20, True, WHITE, "right")

    def kv(doc, x, y, pairs, kw=95):
        for k, v in pairs:
            doc.text(x, y, k, 8.5, True)
            doc.text(x + kw, y, v, 8.5)
            y += 13
        return y

    def render_customer_po(c, po):
        from_layout = int(c["customer_id"][-1]) % 3
        d = PDFDoc()
        total = sum(l["qty"] * l["price"] for l in po["lines"])
        if from_layout == 0:
            header(d, c["customer_name"], [c["address"], f"Purchasing - {c['buyer']}"], "PURCHASE ORDER")
            kv(d, 380, 100, [("PO Number", po["po_number"]), ("PO Date", po["po_date"].strftime("%m/%d/%Y")),
                             ("Required By", po["requested"].strftime("%m/%d/%Y")), ("Terms", c["payment_terms"])], 80)
            d.text(40, 100, "Supplier", 8.5, True); d.text(40, 113, OUR); d.text(40, 126, OUR_ADDR)
            d.text(40, 146, "Ship To", 8.5, True); d.text(40, 159, f"{c['customer_name']} - Receiving Dock 3"); d.text(40, 172, c["address"])
            rows = [[str(i + 1), l["part"], l["cust_part"], l["desc"], qty_s(l["qty"]), item[l["part"]]["uom"], money(l["price"]),
                     money(l["qty"] * l["price"])] for i, l in enumerate(po["lines"])]
            y = d.table(40, 200, [("Ln", 24, "left"), ("Supplier Part #", 70, "left"), ("Our Item #", 62, "left"),
                                  ("Description", 168, "left"), ("Qty", 40, "right"), ("UOM", 32, "left"),
                                  ("Unit Price", 58, "right"), ("Extended", 78, "right")], rows, size=7.8)
            d.text(470, y + 18, "Order Total (USD)", 9, True, align="right"); d.text(572, y + 18, money(total), 9, True, align="right")
            note_y = y + 50
        elif from_layout == 1:
            header(d, c["customer_name"], [c["address"]], "Purchase Order")
            kv(d, 40, 100, [("Purchase order", po["po_number"]), ("Order date", po["po_date"].strftime("%d.%m.%Y")),
                            ("Vendor", f"{OUR} (vendor no. 100{c['customer_id'][-1]}87)"), ("Payment terms", c["payment_terms"]),
                            ("Incoterms", "FCA Dayton OH")], 95)
            rows = []
            for i, l in enumerate(po["lines"]):
                rows.append([f"{(i + 1) * 10:05d}", l["cust_part"], f"{l['desc']} / Vendor mat. {l['part']}", qty_s(l["qty"]),
                             item[l["part"]]["uom"], money(l["price"]), money(l["qty"] * l["price"]), po["requested"].strftime("%d.%m.%Y")])
            y = d.table(40, 185, [("Item", 34, "left"), ("Material", 58, "left"), ("Short text", 196, "left"), ("Order qty", 44, "right"),
                                  ("Unit", 28, "left"), ("Net price", 52, "right"), ("Net value", 56, "right"), ("Deliv. date", 64, "left")],
                        rows, size=7.4)
            d.text(470, y + 18, "Total net value USD", 9, True, align="right"); d.text(572, y + 18, money(total), 9, True, align="right")
            note_y = y + 50
        else:
            header(d, c["customer_name"], [c["address"]], "ORDER")
            d.text(40, 104, f"PO No. {po['po_number']}    Date: {po['po_date']:%B %d, %Y}    Please deliver by: {po['requested']:%B %d, %Y}", 9, True)
            d.text(40, 120, f"To: {OUR}, {OUR_ADDR}    Terms: {c['payment_terms']}", 8.5)
            rows = [[qty_s(l["qty"]), l["part"], l["desc"], money(l["price"]), money(l["qty"] * l["price"])] for l in po["lines"]]
            y = d.table(40, 140, [("Qty", 50, "right"), ("Part No.", 80, "left"), ("Description", 250, "left"),
                                  ("Price Each", 70, "right"), ("Total", 82, "right")], rows)
            d.text(470, y + 18, "TOTAL", 9, True, align="right"); d.text(572, y + 18, money(total), 9, True, align="right")
            note_y = y + 50
        for i, n in enumerate(po.get("notes", [])):
            d.text(40, note_y + i * 12, n, 8)
        d.text(40, 760, f"Please confirm receipt of this order to {c['buyer_email']}. Reference the PO number on all documents.", 7.5, color=GRAY)
        return d.bytes()

    def render_invoice(s, inv):
        d = PDFDoc()
        sub = sum(l["qty"] * l["price"] for l in inv["lines"])
        total = sub + inv["freight"]
        mx = s["country"] == "MX"
        header(d, s["supplier_name"], [s["address"]], "FACTURA / INVOICE" if mx else "INVOICE")
        kv(d, 360, 100, [("Invoice No." if not mx else "Factura / Invoice", inv["invoice_number"]),
                         ("Invoice Date", inv["date"].strftime("%Y-%m-%d")),
                         ("Customer PO", inv["po_number"] or ""), ("Terms", s["payment_terms"])], 100)
        d.text(40, 100, "Bill To", 8.5, True); d.text(40, 113, OUR); d.text(40, 126, f"Accounts Payable, {OUR_ADDR}")
        cols = ([("Clave", 80, "left"), ("Descripción / Description", 222, "left"), ("Cant./Qty", 60, "right"),
                 ("P.Unit/Price", 80, "right"), ("Importe/Amount", 90, "right")] if mx else
                [("Line", 32, "left"), ("Item", 88, "left"), ("Description", 212, "left"), ("Qty", 50, "right"),
                 ("Unit Price", 68, "right"), ("Amount", 82, "right")])
        rows = []
        for i, l in enumerate(inv["lines"]):
            r = [l["code"], l["desc"], qty_s(l["qty"]), money(l["price"]), money(l["qty"] * l["price"])]
            rows.append(r if mx else [str(i + 1)] + r)
        y = d.table(40, 170, cols, rows)
        for k, v in [("Subtotal", money(sub)), ("Freight", money(inv["freight"])), ("Total Due (USD)", money(total))]:
            y += 15; d.text(470, y, k, 9, True, align="right"); d.text(572, y, v, 9, align="right")
        y += 36
        if inv.get("bank_change"):
            d.box(40, y - 12, 532, 40, fill=(1, 0.95, 0.85), stroke=0.6)
            d.text(48, y + 2, "IMPORTANT: Our banking details have changed. Please update your records and remit all", 8.5, True)
            d.text(48, y + 15, "payments, including open invoices, to the new account below effective immediately.", 8.5, True)
            y += 44
        d.text(40, y, f"Remit to: {s['supplier_name']}, {inv['bank']}, account ending {inv['last4']}", 8.5)
        for i, n in enumerate(inv.get("notes", [])):
            d.text(40, y + 16 + i * 12, n, 8)
        d.text(40, 760, "Thank you for your business. Late payments are subject to 1.5% monthly service charge.", 7.5, color=GRAY)
        return d.bytes()

    def render_datasheet(s, ds):
        d = PDFDoc()
        header(d, s["supplier_name"], [s["address"], "Engineering & Technical Services"], "PRODUCT DATA SHEET")
        d.text(40, 104, ds["product"], 13, True)
        y = kv(d, 40, 124, [("Supplier part no.", ds["code"]), ("Customer part ref.", ds["part"]), ("Revision", ds["rev"]),
                             ("Date", ds["date"].strftime("%Y-%m-%d")), ("Material", ds["material"]), ("Hardness", ds["hardness"]),
                             ("Operating temp.", ds["temp"])], 105)
        d.text(40, y + 12, "Dimensions", 10, True)
        rows = [[s_["characteristic"], f"{s_['nominal_mm']:.3f}", f"± {s_['tol_mm']:.3f}"] for s_ in ds["dims"]]
        y = d.table(40, y + 20, [("Characteristic", 240, "left"), ("Nominal (mm)", 110, "right"), ("Tolerance (mm)", 110, "right")], rows)
        d.text(40, y + 22, "Compliance", 10, True)
        y = d.table(40, y + 30, [("Regulation", 240, "left"), ("Status", 220, "left")],
                    [["EU RoHS 2011/65/EU (2015/863)", ds["rohs"]], ["EU REACH SVHC (> 0.1% w/w)", ds["reach"]],
                     ["Conflict minerals (CMRT)", "Declaration available on request"]])
        for i, n in enumerate(ds.get("notes", [])):
            d.text(40, y + 22 + i * 12, n, 8.5)
        d.text(40, 760, "Data are typical values for guidance and do not constitute a specification unless agreed in writing.", 7.5, color=GRAY)
        return d.bytes()

    def render_coc(s, q):
        d = PDFDoc()
        header(d, s["supplier_name"], [s["address"], "Quality Assurance"], "CERTIFICATE OF CONFORMANCE")
        d.text(40, 102, "Certificate of Conformance & Dimensional Inspection Report", 11, True)
        kv(d, 40, 122, [("Customer", OUR), ("Customer PO", q["po_number"]), ("Part number", f"{q['part']} ({q['code']})"),
                        ("Description", item[q["part"]]["description"])], 85)
        kv(d, 360, 122, [("Lot no.", q["lot"] or "-"), ("Heat no.", q["heat"]), ("Qty shipped", qty_s(q["qty"])),
                         ("Inspection date", q["date"].strftime("%Y-%m-%d"))], 85)
        rows = [[m["characteristic"], f"{m['nominal']:.3f}", f"± {m['tol']:.3f}", f"{m['min']:.3f}", f"{m['max']:.3f}", m["result"]]
                for m in q["measurements"]]
        y = d.table(40, 190, [("Characteristic", 170, "left"), ("Nominal", 62, "right"), ("Tol.", 58, "right"),
                              ("Meas. min", 66, "right"), ("Meas. max", 66, "right"), ("Result", 70, "left")], rows)
        d.text(40, y + 20, f"Sample size: {q['sample']} pcs per characteristic (ANSI/ASQ Z1.4, level II).", 8.5)
        for i, n in enumerate(q.get("notes", [])):
            d.text(40, y + 34 + i * 12, n, 8.5)
        d.text(40, y + 70, "We certify that the material supplied conforms to the purchase order, drawing and specification requirements.", 8.5)
        d.text(40, y + 96, f"{q['inspector']}, Quality Manager", 9, True)
        return d.bytes()

    def render_ecn(e):
        d = PDFDoc()
        header(d, OUR, [OUR_ADDR, "Engineering"], "ENGINEERING CHANGE")
        d.text(40, 104, f"Engineering Change Notice {e['ecn']}", 13, True)
        y = kv(d, 40, 126, [("Effective date", e["effective"].strftime("%Y-%m-%d")), ("Originator", e["originator"]),
                             ("Change class", e["cls"]), ("Affected item", f"{e['old_part']} rev {e['old_rev']}"),
                             ("Replaced by", f"{e['new_part']} rev {e['new_rev']}"),
                             ("Affected customers", e["customers"])], 105)
        d.text(40, y + 14, "Reason for change", 10, True)
        for i, l in enumerate(e["reason"]):
            d.text(40, y + 30 + i * 12, l, 8.5)
        y += 40 + len(e["reason"]) * 12
        d.text(40, y + 10, "Disposition", 10, True)
        d.table(40, y + 18, [("Inventory / orders", 200, "left"), ("Action", 300, "left")], e["disposition"])
        return d.bytes()

    # ------------------------------------------------------------------ master-data helpers
    fg_by_cust = {"CUS-01": ["RPC-1001", "RPC-1002", "RPC-1003"], "CUS-02": ["RPC-1006", "RPC-1007", "RPC-1008"],
                  "CUS-03": ["RPC-1004", "RPC-1006", "RPC-1012"], "CUS-04": ["RPC-1001", "RPC-1002", "RPC-1007"],
                  "CUS-05": ["RPC-1005", "RPC-1008"], "CUS-06": ["RPC-1003", "RPC-1004", "RPC-1012"]}
    qty_range = {"RPC-1004": (4, 30), "RPC-1003": (10, 80), "RPC-1006": (10, 60), "RPC-1012": (10, 60), "RPC-1009": (10, 60)}
    po_fmt = {"CUS-01": "AAE-{n}", "CUS-02": "CPS-PO-{n}", "CUS-03": "HR-2026-{n}", "CUS-04": "45000{n}",
              "CUS-05": "BMH{n}", "CUS-06": "PFM/{n}"}
    for c in customers:
        n = c["customer_name"].split()[0]
        c.setdefault("address", {"CUS-01": "2200 Harvest Way, Peoria, IL 61602", "CUS-02": "41 Mill Street, Worcester, MA 01608",
                                 "CUS-03": "900 Innovation Dr, San Jose, CA 95134", "CUS-04": "77 Belt Line Rd, Greenville, SC 29601",
                                 "CUS-05": "15 Harbor Blvd, Houma, LA 70360", "CUS-06": "350 Queen St, Kitchener, ON N2G 1W1"}[c["customer_id"]])
        c.setdefault("buyer", {"CUS-01": "Megan Ortiz", "CUS-02": "Paul Brennan", "CUS-03": "Priya Raman", "CUS-04": "Luis Carter",
                               "CUS-05": "Jolene Thibodeaux", "CUS-06": "Graham Fischer"}[c["customer_id"]])
        c["buyer_email"] = c["buyer"].lower().replace(" ", ".") + "@" + c["email_domain"]
    for s in suppliers:
        s.setdefault("address", {"SUP-01": "1450 Industrial Pkwy, Allentown, PA 18109", "SUP-02": "3900 Lakeshore Ave, Cleveland, OH 44114",
                                 "SUP-03": "610 Foundry Rd, Muncie, IN 47302", "SUP-04": "88 Gear Ct, Rockford, IL 61109",
                                 "SUP-05": "2701 Polymer Way, Akron, OH 44311", "SUP-06": "Av. Industrias 1450, Apodaca, N.L. 66600, MX"}[s["supplier_id"]])
        s.setdefault("contact", {"SUP-01": "Rachel Kim", "SUP-02": "Tom Novak", "SUP-03": "Gary Lindqvist", "SUP-04": "Ana Duarte",
                                 "SUP-05": "Kevin Shah", "SUP-06": "Mariana Treviño"}[s["supplier_id"]])
        s.setdefault("bank", {"SUP-01": "First Commonwealth Bank", "SUP-02": "Fifth Lakes Bank", "SUP-03": "Hoosier National Bank",
                              "SUP-04": "Rock River Bank", "SUP-05": "Summit Trust", "SUP-06": "Banorte"}[s["supplier_id"]])
    comps_by_sup = {}
    for i in items:
        if i["item_type"] == "purchased_component":
            comps_by_sup.setdefault(i["supplier_id"], []).append(i)
    def cust_part(cid, part):
        return {"CUS-01": "ATL", "CUS-02": "CV", "CUS-03": "HX", "CUS-04": "SMT", "CUS-05": "BW", "CUS-06": "PF"}[cid] + "-" + \
               str(int(hashlib.md5((cid + part).encode()).hexdigest(), 16) % 90000 + 10000)
    def price_for(cid, part):
        return round(float(item[part]["list_price"]) * (1 - float(cust[cid]["discount_pct"]) / 100), 2)
    def sup_email(s, who="ar"):
        return f"{who}@{s['email_domain']}"

    # ------------------------------------------------------------------ 1. customer POs
    po_counter = {k: 0 for k in cust}
    def make_customer_po(cid, offset, parts=None, price_factor=None, lead_override=None, notes=None, batch="history"):
        po_counter[cid] += 1
        n = {"CUS-01": 77800, "CUS-02": 2150, "CUS-03": 400, "CUS-04": 12380, "CUS-05": 5520, "CUS-06": 3300}[cid] + po_counter[cid] * 7
        po_no = po_fmt[cid].format(n=n)
        po_date = day(offset - rng.randint(0, 1))
        parts = parts or rng.sample(fg_by_cust[cid], rng.randint(1, min(3, len(fg_by_cust[cid]))))
        lines = []
        for p in parts:
            lo, hi = qty_range.get(p, (25, 400))
            q = rng.randint(lo, hi)
            q = q if q < 50 else round(q, -1)
            price = price_for(cid, p)
            if price_factor and p in price_factor:
                price = round(price * price_factor[p], 2)
            lines.append(dict(part=p, cust_part=cust_part(cid, p), desc=item[p]["description"], qty=q, price=price))
        lead = max(int(item[p]["lead_time_days"]) for p in parts)
        requested = po_date + dt.timedelta(days=lead_override if lead_override is not None else lead + rng.randint(4, 21))
        po = dict(po_number=po_no, po_date=po_date, requested=requested, lines=lines, notes=notes or [])
        return po, render_customer_po(cust[cid], po)

    def send_customer_po(cid, offset, po, pdf, subject=None, body=None, mailbox="orders", batch="history"):
        c = cust[cid]
        subject = subject or rng.choice([f"PO {po['po_number']}", f"New purchase order {po['po_number']} - {c['customer_name']}",
                                         f"{c['customer_name']} PO# {po['po_number']}", f"Order {po['po_number']} attached"])
        body = body or rng.choice([
            f"Hello,\n\nPlease find attached our purchase order {po['po_number']}. Kindly confirm price and delivery date.\n\nThanks,\n{c['buyer']}\nPurchasing, {c['customer_name']}",
            f"Hi Ridgeline team,\n\nAttached is PO {po['po_number']}. Please send an order acknowledgment within 48 hours.\n\nBest regards,\n{c['buyer']}",
            f"Good morning,\n\nNew order attached. Ship complete please - partial shipments must be pre-approved.\n\n{c['buyer']} | Buyer\n{c['customer_name']}"])
        return add_email(offset, mailbox, c["buyer"], c["buyer_email"], subject, body,
                         (f"PO_{po['po_number'].replace('/', '-')}.pdf", pdf), batch=batch)

    # planted: offsets chosen so each lands in history
    plants_po = {
        ("CUS-01", -20): dict(parts=["RPC-1003", "RPC-1001"], price_factor={"RPC-1003": 0.94},
                              notes=["Pricing per 2025 annual agreement."]),
        ("CUS-04", -8): dict(parts=["RPC-1001"], price_factor={"RPC-1001": 0.93}),
        ("CUS-03", -12): dict(parts=["RPC-1009", "RPC-1006"]),
        ("CUS-06", -5): dict(parts=["RPC-1004"], lead_override=6, notes=["URGENT - line-down replacement, need ASAP."]),
        ("CUS-05", -15): dict(parts=["RPC-1005", "RPC-1008"],
                              notes=["Qty for RPC-1005 is 2 cartons of 50 kits. If cartons are 40 kits, ship 2 cartons anyway",
                                     "and invoice actual quantity. Hold RPC-1008 if lead time exceeds 2 weeks."]),
    }
    regular = [(rng.choice(list(cust)), -rng.randint(1, history_days)) for _ in range(27)]
    for cid, off in regular:
        po, pdf = make_customer_po(cid, off, parts=[p for p in rng.sample(fg_by_cust[cid], rng.randint(1, min(3, len(fg_by_cust[cid]))))])
        send_customer_po(cid, off, po, pdf)
    for (cid, off), kw in plants_po.items():
        po, pdf = make_customer_po(cid, off, **kw)
        send_customer_po(cid, off, po, pdf)
    # duplicate: Corvus re-sends the same PO two days later
    po, pdf = make_customer_po("CUS-02", -18, parts=["RPC-1006", "RPC-1007"])
    send_customer_po("CUS-02", -18, po, pdf)
    send_customer_po("CUS-02", -16, po, pdf, subject=f"RE: PO {po['po_number']} - resending, did you receive?",
                     body=f"Hi,\n\nWe have not received an acknowledgment for PO {po['po_number']}. Resending the PO attached.\n\nPaul Brennan\nCorvus Packaging Systems")
    # order change requests in the email body (no attachment)
    add_email(-10, "orders", "Luis Carter", cust["CUS-04"]["buyer_email"], "Change to our open order - add 40 pcs RPC-1007",
              "Hi team,\n\nPlease add 40 pcs of RPC-1007 (timing pulley HTD 5M 30T) to our open PO and keep the same ship date. "
              "Send a revised acknowledgment.\n\nThanks,\nLuis Carter\nSummit Conveyor Co.")
    add_email(-3, "orders", "Megan Ortiz", cust["CUS-01"]["buyer_email"], "Pull in delivery date?",
              "Hello,\n\nCan you pull in the delivery for our last order of RPC-1003 shafts by one week? Our harvest-line build moved up.\n\nMegan")

    # ------------------------------------------------------------------ 2. our POs, receipts, invoices, CoCs
    po_seq = [45100]
    our_pos = []
    for sid in sorted(sup):
        comps = comps_by_sup.get(sid, [])
        off = -88 + rng.randint(0, 6)
        while off < -8:
            po_seq[0] += rng.randint(3, 11)
            lines = []
            for ln, comp in enumerate(rng.sample(comps, rng.randint(1, len(comps)))):
                q = rng.choice([200, 300, 400, 500, 600, 800, 1000, 1200]) if comp["unit_cost"] < 10 else rng.choice([50, 80, 100, 150, 200, 250])
                lines.append(dict(line=ln + 1, part=comp["part_number"], code=comp["supplier_item_code"], desc=comp["description"],
                                  qty=q, price=float(comp["unit_cost"])))
            p = dict(po_number=f"PO-{po_seq[0]}", supplier_id=sid, po_date=day(off), lines=lines)
            rec_off = off + rng.randint(9, 17)
            p["received_offset"] = rec_off if rec_off <= -1 else None
            our_pos.append(p)
            off += rng.randint(12, 20)
    for p in our_pos:
        for l in p["lines"]:
            pos.append(dict(po_number=p["po_number"], supplier_id=p["supplier_id"], po_date=p["po_date"], line_number=l["line"],
                            part_number=l["part"], supplier_item_code=l["code"], qty_ordered=l["qty"], unit_price=l["price"]))
            if p["received_offset"] is not None:
                rq = l["qty"]
                p.setdefault("rq", {})[l["line"]] = rq
                receipts.append(dict(po_number=p["po_number"], line_number=l["line"], part_number=l["part"],
                                     received_date=day(p["received_offset"]), qty_received=rq))

    inv_counter = {}
    def make_invoice(p, offset, qty_override=None, price_factor=None, po_ref=True, bank_change=False, notes=None):
        s = sup[p["supplier_id"]]
        inv_counter[s["supplier_id"]] = inv_counter.get(s["supplier_id"], 0) + 1
        prefix = {"SUP-01": "KBS-", "SUP-02": "GLS", "SUP-03": "MC-INV-", "SUP-04": "PGW", "SUP-05": "ST-", "SUP-06": "AEM-F"}[s["supplier_id"]]
        base = {"SUP-01": 118200, "SUP-02": 60410, "SUP-03": 9120, "SUP-04": 33050, "SUP-05": 90210, "SUP-06": 4410}[s["supplier_id"]]
        lines = []
        for l in p["lines"]:
            q = (qty_override or {}).get(l["line"], p.get("rq", {}).get(l["line"], l["qty"]))
            price = round(l["price"] * (price_factor or {}).get(l["line"], 1.0), 2)
            lines.append(dict(code=l["code"], desc=l["desc"], qty=q, price=price))
        inv = dict(invoice_number=f"{prefix}{base + inv_counter[s['supplier_id']] * 3}", date=day(offset - rng.randint(0, 1)),
                   po_number=p["po_number"] if po_ref else None, lines=lines,
                   freight=rng.choice([0.0, 0.0, 95.0, 145.0, 185.0, 240.0]), bank=s["bank"], last4=s["bank_account_last4"],
                   bank_change=bank_change, notes=notes or [])
        if bank_change:
            inv["bank"], inv["last4"] = "Pinnacle Digital Bank", "9034"
        return inv, render_invoice(s, inv)

    def send_invoice(p, offset, inv, pdf, mailbox="ap", from_addr=None, subject=None, body=None, batch="history"):
        s = sup[p["supplier_id"]]
        subject = subject or rng.choice([f"Invoice {inv['invoice_number']}", f"{s['supplier_name']} - Invoice {inv['invoice_number']} for {inv['po_number'] or 'your order'}",
                                         f"Invoice {inv['invoice_number']} attached"])
        body = body or f"Dear Accounts Payable,\n\nPlease find attached invoice {inv['invoice_number']}. Payment terms {s['payment_terms']}.\n\nRegards,\nAccounts Receivable\n{s['supplier_name']}"
        return add_email(offset, mailbox, f"{s['supplier_name']} AR", from_addr or sup_email(s), subject, body,
                         (f"Invoice_{inv['invoice_number']}.pdf", pdf), batch=batch)

    def make_coc(p, offset, overrides=None, lot=True, result_override=None, notes=None):
        s = sup[p["supplier_id"]]
        l = next(l for l in p["lines"] if l["part"] in spec)
        meas = []
        for sp in spec[l["part"]]:
            nom, tol = float(sp["nominal_mm"]), float(sp["tol_mm"])
            lo = nom + tol * rng.uniform(-0.6, 0.1)
            hi = nom + tol * rng.uniform(0.0, 0.6)
            if overrides and sp["characteristic"] in overrides:
                f_lo, f_hi = overrides[sp["characteristic"]]   # in units of tolerance from nominal
                lo, hi = nom + f_lo * tol, nom + f_hi * tol
            res = "PASS" if (nom - tol <= lo and hi <= nom + tol) else "FAIL"
            if result_override and sp["characteristic"] in result_override:
                res = result_override[sp["characteristic"]]
            meas.append(dict(characteristic=sp["characteristic"], nominal=nom, tol=tol, min=lo, max=hi, result=res))
        q = dict(po_number=p["po_number"], part=l["part"], code=l["code"], qty=l["qty"], date=day(offset),
                 lot=(f"L{p['po_number'][-3:]}{rng.randint(10, 99)}-{day(offset):%y%m}" if lot else None),
                 heat=f"H{rng.randint(100000, 999999)}", measurements=meas, sample=rng.choice([13, 20, 32]),
                 inspector={"SUP-01": "Deborah Hale", "SUP-02": "Ivan Petrov", "SUP-03": "Sam Okafor", "SUP-04": "Lena Brooks",
                            "SUP-05": "Marcus Lee", "SUP-06": "Jorge Salinas"}[s["supplier_id"]], notes=notes or [])
        return q, render_coc(s, q)

    def send_coc(p, offset, q, pdf, mailbox="quality", batch="history"):
        s = sup[p["supplier_id"]]
        return add_email(offset, mailbox, f"{s['supplier_name']} Quality", sup_email(s, "quality"),
                         rng.choice([f"CoC and inspection report - {q['po_number']}", f"Certificate of Conformance {q['part']} lot {q['lot'] or ''}".strip(),
                                     f"Shipment docs for {q['po_number']}"]),
                         f"Hello,\n\nAttached is the certificate of conformance and dimensional inspection report for the shipment against {q['po_number']}.\n\n{q['inspector']}\nQuality, {s['supplier_name']}",
                         (f"CoC_{q['po_number']}_{q['part']}.pdf", pdf), batch=batch)

    received = [p for p in our_pos if p["received_offset"] is not None]
    rng.shuffle(received)
    # keep the 3 most recently received POs un-invoiced so they can be billed in "today's mail"
    received.sort(key=lambda p: p["received_offset"])
    today_pos = received[-3:]
    hist_pos = [p for p in received[:-3] if p["received_offset"] >= -history_days + 2]  # older ones were billed before the window
    by_sup = {}
    for p in hist_pos:
        by_sup.setdefault(p["supplier_id"], []).append(p)
    planted_inv = {}
    def first(sid):
        lst = by_sup[sid]
        return next(p for p in lst if p["received_offset"] + 4 <= -2 and id(p) not in planted_inv)
    planted_inv[id(first("SUP-04"))] = dict(price_factor={1: 1.045})
    planted_inv[id(first("SUP-03"))] = dict(price_factor={1: 1.07}, notes=["Price includes ductile iron alloy surcharge (Nov)."])
    p_kbs = first("SUP-01")
    planted_inv[id(p_kbs)] = dict(qty_override={1: p_kbs["lines"][0]["qty"] + 150})
    planted_inv[id(first("SUP-02"))] = dict(bank_change=True)
    planted_inv[id(first("SUP-06"))] = dict(po_ref=False)
    dup_sealtech = first("SUP-05")
    planted_inv[id(dup_sealtech)] = dict()
    misroute = [p for p in hist_pos if id(p) not in planted_inv][:2]
    invoice_docs = {}
    for p in hist_pos:
        off = min(p["received_offset"] + rng.randint(1, 5), -1)
        kw = planted_inv.get(id(p), {})
        inv, pdf = make_invoice(p, off, **kw)
        invoice_docs[p["po_number"]] = (inv, pdf, off)
        s = sup[p["supplier_id"]]
        if kw.get("bank_change"):
            send_invoice(p, off, inv, pdf, from_addr="billing@glsteelbar-payments.com",
                         subject=f"URGENT: Invoice {inv['invoice_number']} - updated remittance details",
                         body=f"Dear AP team,\n\nPlease note that our bank details have changed. Kindly process the attached invoice {inv['invoice_number']} "
                              f"and all open invoices to our new account today to avoid a credit hold.\n\nRegards,\nTom Novak\nGreat Lakes Steel & Bar")
        elif p in misroute:
            send_invoice(p, off, inv, pdf, mailbox="orders")
        else:
            send_invoice(p, off, inv, pdf)
        if p is dup_sealtech:
            send_invoice(p, min(off + 6, -1), inv, pdf, subject=f"2nd notice - invoice {inv['invoice_number']} past due",
                         body=f"Dear AP,\n\nOur records show invoice {inv['invoice_number']} is unpaid. Copy attached. Please remit at your earliest convenience.\n\nSealTech Industries AR")

    # CoCs for receipts of parts with a specification
    with_spec = [p for p in hist_pos if any(l["part"] in spec for l in p["lines"])]
    rng.shuffle(with_spec)
    coc_plants = {}
    kbs = next(p for p in with_spec if p["supplier_id"] == "SUP-01")
    coc_plants[id(kbs)] = dict(overrides={"Bore diameter": (0.1, 1.45)}, result_override={"Bore diameter": "PASS"})
    mc = next(p for p in with_spec if p["supplier_id"] == "SUP-03")
    coc_plants[id(mc)] = dict(overrides={"Bore diameter": (0.16, 1.24)},
                              notes=["Bore diameter exceeds upper limit on 3 of 20 samples - shipped under deviation request DR-118 (pending approval)."])
    gl = next(p for p in with_spec if p["supplier_id"] == "SUP-02")
    coc_plants[id(gl)] = dict(lot=False)
    coc_list = list(coc_plants) + [id(p) for p in with_spec if id(p) not in coc_plants][:11]
    by_id = {id(p): p for p in with_spec}
    for k, pid in enumerate(coc_list):
        p = by_id[pid]
        off = p["received_offset"]
        q, pdf = make_coc(p, off - 1, **coc_plants.get(pid, {}))
        send_coc(p, off - 1, q, pdf, mailbox="ap" if k == 5 else "quality")

    # ------------------------------------------------------------------ 3. product data sheets
    ds_plan = [("CMP-2001", "52100 chrome steel", "60-64 HRC", "-30 to 120 °C", "Compliant", "Compliant", "C", []),
               ("CMP-2002", "AISI 4140, quenched & tempered", "28-32 HRC", "n/a", "Compliant", "Compliant", "B", []),
               ("CMP-2003", "Ductile iron ASTM A536 65-45-12", "170-220 HB", "-40 to 250 °C", "Compliant", "Compliant", "D", []),
               ("CMP-2004", "8620 carburized, case 0.6-0.9 mm", "58-62 HRC case", "-20 to 100 °C", "Compliant", "Compliant", "A", []),
               ("CMP-2005", "FKM blend, 70 Shore A", "70 +/- 5 Shore A", "-20 to 150 °C", "Compliant", "Compliant", "F",
                ["Rev F: compound changed from FKM 75A to FKM blend 70A for improved availability."]),
               ("CMP-2006", "Aluminum 6262-T6 (replaces 6061-T6)", "95 HB", "n/a", "Not compliant - alloy contains Pb 0.4-0.7%",
                "Lead (Pb) present above 0.1% w/w", "E", ["Rev E: alloy change from 6061-T6 to free-machining 6262-T6 for better chip control."])]
    for k, (part, mat, hard, temp, rohs, reach, rev, notes) in enumerate(ds_plan):
        it = item[part]
        s = sup[it["supplier_id"]]
        off = -rng.randint(4, history_days - 2)
        ds = dict(product=it["description"], code=it["supplier_item_code"], part=part, rev=rev, date=day(off), material=mat, hardness=hard,
                  temp=temp, rohs=rohs, reach=reach, notes=notes,
                  dims=[dict(characteristic=x["characteristic"], nominal_mm=float(x["nominal_mm"]), tol_mm=float(x["tol_mm"])) for x in spec.get(part, [])])
        add_email(off, "engineering", s["contact"], sup_email(s, s["contact"].split()[0].lower()),
                  f"Updated product data sheet rev {rev} - {it['supplier_item_code']}",
                  f"Hi Ridgeline engineering,\n\nAttached is the latest product data sheet (rev {rev}) for {it['supplier_item_code']} ({it['description']}). "
                  f"Please update your approved supplier documentation.\n\nBest,\n{s['contact']}\n{s['supplier_name']}",
                  (f"Datasheet_{it['supplier_item_code']}_rev{rev}.pdf", render_datasheet(s, ds)))

    # ------------------------------------------------------------------ 4. engineering change notices
    ecns = [dict(ecn="ECN-2026-014", effective=day(-30), originator="Dana Whitfield, Product Engineering", cls="Class I - form/fit/function",
                 old_part="RPC-1009", old_rev="B", new_part="RPC-1012", new_rev="C", customers="Helix Robotics; Pioneer Food Machinery",
                 reason=["Field returns show fatigue cracking of the output shaft in high-cycle robotic applications.",
                         "Rev C adds induction hardening (50-55 HRC) at the bearing journals. RPC-1009 is obsolete and must not be",
                         "quoted or shipped; open and new customer orders for RPC-1009 are to be converted to RPC-1012."],
                 disposition=[["Finished goods RPC-1009 (on hand: 140)", "Scrap - do not ship"],
                              ["Open customer orders for RPC-1009", "Convert to RPC-1012; confirm new price with customer"],
                              ["Work in process", "Rework per routing R-1012-C"]], offset=-31),
            dict(ecn="ECN-2026-017", effective=day(-8), originator="Arjun Mehta, Manufacturing Engineering", cls="Class II - documentation",
                 old_part="RPC-1003", old_rev="D", new_part="RPC-1003", new_rev="E", customers="All",
                 reason=["Drawing update: 0.5 x 45 deg chamfer added at keyway ends to remove burr risk. No change to form, fit",
                         "or function; part number unchanged."],
                 disposition=[["Inventory and open orders", "Use as is"], ["Drawing", "Release rev E in PLM"]], offset=-9)]
    for e in ecns:
        add_email(e["offset"], "engineering", e["originator"].split(",")[0],
                  e["originator"].split(",")[0].lower().replace(" ", ".") + "@ridgelinepc.com",
                  f"{e['ecn']} released - {e['old_part']} rev {e['old_rev']} to {e['new_part']} rev {e['new_rev']}",
                  f"Team,\n\n{e['ecn']} has been released and is effective {e['effective']:%Y-%m-%d}. See the attached notice for disposition.\n\n{e['originator']}",
                  (f"{e['ecn']}.pdf", render_ecn(e)))

    # ------------------------------------------------------------------ 5. no-action noise
    noise = [("orders", "Fastener & Motion Weekly", "news@fmweekly.com", "This week: steel prices, port delays and the 2027 outlook",
              "Your weekly industry digest. Read online. Unsubscribe."),
             ("ap", "Priya Raman", cust["CUS-03"]["buyer_email"], "Automatic reply: out of office until Monday",
              "I am out of the office with limited access to email. For urgent matters contact purchasing@helixrobotics.io."),
             ("quality", "IT Service Desk", "it@ridgelinepc.com", "Planned maintenance: email archive Saturday 10pm",
              "The email archive will be unavailable Saturday from 10pm to 2am."),
             ("engineering", "Precision Gear Works", "events@pgw-gears.com", "Join us at IMTS booth 4421",
              "See our new low-backlash planetary sets. Book a meeting with our team."),
             ("orders", "Summit Conveyor Co.", "info@summitconveyor.com", "Holiday shipping & receiving schedule",
              "Our docks will be closed Nov 26-27 and Dec 24-26. Please plan deliveries accordingly."),
             ("ap", "Concur Notifications", "noreply@expense-notify.com", "Your expense report was approved",
              "Expense report ER-2291 was approved."),
             ("quality", "ASQ Section 0812", "section0812@asq-members.org", "Monthly meeting: lean audit techniques",
              "Join us Thursday 6pm for dinner and a talk on layered process audits."),
             ("engineering", "LinkedIn", "jobs-noreply@linkedin-mail.com", "5 new jobs for Design Engineer",
              "Mechanical Design Engineer - Dayton, OH and 4 more."),
             ("orders", "Atlas Ag Equipment", cust["CUS-01"]["buyer_email"], "Thanks for the quick turnaround",
              "Just a note to say thanks - the last shipment arrived early and the line is running. Megan"),
             ("ap", "Keystone Bearing Supply", "ar@keystonebearing.com", "Statement of account - October",
              "Your October statement shows no balance past due. No action needed."),
             ("quality", "Ridgeline HR", "hr@ridgelinepc.com", "Reminder: open enrollment ends Friday", "Complete your benefits elections by Friday."),
             ("engineering", "SolidWorks Admin", "cad-admin@ridgelinepc.com", "License server restart tonight", "Save your work by 7pm.")]
    for mb, name, addr, subj, body in noise:
        add_email(-rng.randint(1, history_days), mb, name, addr, subj, body)

    # ------------------------------------------------------------------ 6. TODAY'S MAIL (delivered by the daily-job step)
    T = dict(batch="today")
    po, pdf = make_customer_po("CUS-01", 0, parts=["RPC-1002", "RPC-1001"], price_factor={"RPC-1002": 0.95},
                               notes=["Pricing per 2025 annual agreement."])
    send_customer_po("CUS-01", 0, po, pdf, **T)
    po, pdf = make_customer_po("CUS-03", 0, parts=["RPC-1012", "RPC-1004"])
    send_customer_po("CUS-03", 0, po, pdf, **T)
    po, pdf = make_customer_po("CUS-04", 0, parts=["RPC-1007"])
    send_customer_po("CUS-04", 0, po, pdf, **T)
    for p in today_pos[:2]:
        inv, pdf = make_invoice(p, 0)
        send_invoice(p, 0, inv, pdf, **T)
    pgw = next(po_ for po_ in hist_pos if po_["supplier_id"] == "SUP-04")
    inv, pdf, _ = invoice_docs[pgw["po_number"]]
    send_invoice(pgw, 0, inv, pdf, subject=f"Invoice {inv['invoice_number']} - copy", **T)
    p = next((p for p in today_pos if any(l["part"] in spec for l in p["lines"])), None) or with_spec[0]
    first_part = next(l for l in p["lines"] if l["part"] in spec)["part"]
    ch = spec[first_part][-1]["characteristic"]
    nom, tol = float(spec[first_part][-1]["nominal_mm"]), float(spec[first_part][-1]["tol_mm"])
    q, pdf = make_coc(p, 0, overrides={ch: (-0.4, 1.35)}, result_override={ch: "PASS"})
    send_coc(p, 0, q, pdf, **T)
    add_email(0, "orders", "Fastener & Motion Weekly", "news@fmweekly.com", "This week: tariffs, lead times and robotics demand",
              "Your weekly industry digest. Read online. Unsubscribe.", **T)

    emails.sort(key=lambda e: e["received_ts"])
    return dict(emails=emails, files=files, purchase_orders=pos, goods_receipts=receipts)

rows = lambda t: [r.asDict() for r in spark.table(t).collect()]
INBOX_DATA = build_inbox(rows("item_master"), rows("customers"), rows("suppliers"), rows("component_specs"))

EMAIL_SCHEMA = T.StructType([T.StructField(c, t) for c, t in [
    ("email_id", T.StringType()), ("received_ts", T.TimestampType()), ("mailbox", T.StringType()), ("from_name", T.StringType()),
    ("from_address", T.StringType()), ("subject", T.StringType()), ("body", T.StringType()), ("attachment_name", T.StringType()),
    ("attachment_path", T.StringType()), ("email_text", T.StringType())]])

def email_text(e):
    """What the classify_emails task reads: sender, mailbox, subject, body and the attachment name, in one column."""
    att = f"\n\n[Attachment: {e['attachment_name']}]" if e["attachment_name"] else ""
    return f"From: {e['from_name']} <{e['from_address']}>\nTo: {e['mailbox']}\nSubject: {e['subject']}\n\n{e['body']}{att}"

def deliver_mail(batch):
    """Writes the PDFs of one batch ('history' or 'today') to the inbox volume and appends its emails to `emails`."""
    batch_emails = [e for e in INBOX_DATA["emails"] if e["batch"] == batch]
    for e in batch_emails:
        if e["attachment_path"]:
            full = f"{INBOX}/{e['attachment_path']}"
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(INBOX_DATA["files"][e["attachment_path"]])
    df = spark.createDataFrame([{**{k: e[k] for k in EMAIL_SCHEMA.fieldNames() if k != "email_text"},
                                 "attachment_path": f"{INBOX}/{e['attachment_path']}" if e["attachment_path"] else None,
                                 "email_text": email_text(e)}
                                for e in batch_emails], EMAIL_SCHEMA)
    df.createOrReplaceTempView("incoming_mail")
    spark.sql("CREATE TABLE IF NOT EXISTS emails (email_id STRING, received_ts TIMESTAMP, mailbox STRING, from_name STRING, "
              "from_address STRING, subject STRING, body STRING, attachment_name STRING, attachment_path STRING, email_text STRING) "
              "COMMENT 'Emails received by the orders@, ap@, quality@ and engineering@ shared mailboxes' "
              "TBLPROPERTIES (delta.enableChangeDataFeed = true)")   # the classify_emails task reads new rows via change data feed
    spark.sql("MERGE INTO emails t USING incoming_mail s ON t.email_id = s.email_id WHEN NOT MATCHED THEN INSERT *")
    return len(batch_emails), sum(1 for e in batch_emails if e["attachment_path"])

# Start fresh: delete this notebook's job, clear the inbox volume and drop the pipeline's tables, so the backfill re-processes everything
for j in _api("GET", "/api/2.2/jobs/list", query={"name": JOB_NAME}).get("jobs", []):
    _api("POST", "/api/2.2/jobs/delete", body={"job_id": j["job_id"]})
PIPELINE_TABLES = ["emails", "email_classifications", "parsed_documents", "document_classifications", "email_routing", "documents",
                   "extract_input_purchase_orders", "extract_input_invoices", "extract_input_data_sheets", "extract_input_quality_docs",
                   "extract_input_ecns", "extracted_purchase_orders", "extracted_invoices", "extracted_data_sheets", "extracted_quality_docs",
                   "extracted_ecns", "customer_order_lines", "supplier_invoice_lines", "product_specs", "inspection_results",
                   "engineering_changes", "extraction_review_queue", "document_exceptions", "exception_decisions", "draft_replies"]
views = {r.viewName for r in spark.sql("SHOW VIEWS").collect()}
assert not views & set(PIPELINE_TABLES), f"{views & set(PIPELINE_TABLES)} are views over your own data - step 0.2 would erase the demo; skip it (see step 0.8)"
for t in PIPELINE_TABLES:
    spark.sql(f"DROP TABLE IF EXISTS `{t}`")
for f in dbutils.fs.ls(INBOX):
    dbutils.fs.rm(f.path, True)

spark.createDataFrame(INBOX_DATA["purchase_orders"]).selectExpr(
    "po_number", "supplier_id", "CAST(po_date AS DATE) AS po_date", "CAST(line_number AS INT) AS line_number", "part_number",
    "supplier_item_code", "CAST(qty_ordered AS INT) AS qty_ordered", "CAST(unit_price AS DOUBLE) AS unit_price"
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("purchase_orders")
spark.createDataFrame(INBOX_DATA["goods_receipts"]).selectExpr(
    "po_number", "CAST(line_number AS INT) AS line_number", "part_number", "CAST(received_date AS DATE) AS received_date",
    "CAST(qty_received AS INT) AS qty_received"
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("goods_receipts")
spark.sql("COMMENT ON TABLE purchase_orders IS 'Our purchase orders to suppliers, one row per PO line (ERP)'")
spark.sql("COMMENT ON TABLE goods_receipts IS 'Goods received against our PO lines (ERP receiving)'")

n_emails, n_files = deliver_mail("history")
print(f"Delivered {n_emails} emails with {n_files} PDF attachments to {INBOX} (today's mail is held back for step 9.1)")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.3 Inbox at a glance
# MAGIC Four shared mailboxes; most emails carry a PDF, some don't. *Tip: every result below can be charted — click **+ → Visualization** on any result table.*

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Inbox at a glance: emails per shared mailbox, with and without a PDF attachment
# MAGIC SELECT split(mailbox, '@')[0] AS mailbox,
# MAGIC        CASE WHEN attachment_path IS NULL THEN 'No attachment' ELSE 'PDF attached' END AS attachment,
# MAGIC        count(*) AS emails
# MAGIC FROM emails
# MAGIC GROUP BY ALL
# MAGIC ORDER BY mailbox, attachment;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Inbox at a glance: emails per shared mailbox, with and without a PDF attachment
# MAGIC SELECT split(mailbox, '@')[0] AS mailbox,
# MAGIC        CASE WHEN attachment_path IS NULL THEN 'No attachment' ELSE 'PDF attached' END AS attachment,
# MAGIC        count(*) AS emails
# MAGIC FROM emails
# MAGIC GROUP BY ALL
# MAGIC ORDER BY mailbox, attachment) SELECT `mailbox`,SUM(`emails`) `column_8547166339`,`attachment` FROM q GROUP BY `mailbox`,`attachment`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.4 Your source data in Unity Catalog
# MAGIC Structured ERP data (**tables**) and unstructured files (the **`inbox` volume** of PDFs) are governed side by side in **Unity Catalog** — permissions, lineage and audit come for free. Click a card to see the schema, sample data or lineage — or open the volume and **preview any PDF** right in the browser.

# COMMAND ----------

show_uc_cards(roles={"Source"}, volume=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.5 Meet your Genie agent
# MAGIC Sections 5–9 each end with **💬 Ask Genie**: the same insight, asked in plain English instead of SQL. That's how buyers, AP clerks, quality engineers and customer service will use these results — no SQL, no waiting on the data team, and every answer respects **Unity Catalog** permissions.
# MAGIC
# MAGIC Step 0.5 creates (or reuses) the **Order, Invoice & Quality Documents Genie** agent over this schema. As later steps build new tables, the agent picks them up automatically before each question.

# COMMAND ----------

TABLE_COMMENTS = {t: d for t, (role, d, g) in TABLE_INFO.items()}
TABLE_COMMENTS.update({
    "email_routing": "Every email with the workflow from the classify_emails AI Function task (customer_purchase_order, supplier_invoice, product_data_sheet, quality_document, engineering_change, no_action).",
    "documents": "One row per parsed attachment: doc_type from the classify_documents AI Function task, the email it came from, its mailbox and processed_by (one-time backfill or daily job).",
    "customer_order_lines": "ai_extract output: one row per customer PO line (customer_po_number, part_number, quantity, unit_price, line_value) with min_confidence.",
    "supplier_invoice_lines": "ai_extract output: one row per supplier invoice line (invoice_number, po_number, item_code, quantity, unit_price, amount) plus remit_to_account_last4 and bank_change_notice.",
    "inspection_results": "ai_extract output: one row per measured characteristic on a certificate of conformance / inspection report, with the supplier's own result.",
    "document_exceptions": "Exceptions found by SQL rules: exception_type, category, counterparty, reference, detail and amount_at_risk_usd.",
    "exception_decisions": "ai_decide output per exception: route (who handles it), route_confidence, hold_probability (0-1) and business_impact (0 negligible - 3 critical).",
    "draft_replies": "ai_gen output: drafted email per exception that needs a person; recipient says who it is for.",
})
COLUMN_COMMENTS = {
    ("document_exceptions", "amount_at_risk_usd"): "Estimated dollar impact of the exception (price difference x qty, overbilled qty x price, lot or invoice value)",
    ("exception_decisions", "hold_probability"): "ai_decide probability (0-1) that payment or shipment should be held until resolved",
    ("exception_decisions", "business_impact"): "ai_decide score: 0 negligible, 1 minor, 2 significant, 3 critical",
    ("customer_order_lines", "min_confidence"): "Lowest ai_extract confidence score among the key fields of this line",
}
GENIE_TABLES = [t for t, (role, d, g) in TABLE_INFO.items() if g]
fq = lambda t: f"{CATALOG}.{SCHEMA}.{t}"
_q = lambda s: s.replace("'", "''")

def existing_tables():
    return {r.tableName for r in spark.sql(f"SHOW TABLES IN `{CATALOG}`.`{SCHEMA}`").collect()}

def describe_tables():
    """(Re)applies business descriptions — CREATE OR REPLACE resets them, and Genie relies on them."""
    have = existing_tables()
    for t, c in TABLE_COMMENTS.items():
        if t not in have:
            continue
        try:
            spark.sql(f"COMMENT ON TABLE {fq(t)} IS '{_q(c)}'")
        except Exception:
            continue   # e.g. a table managed by an AI Function task
        cols = [r.col_name for r in spark.sql(f"SHOW COLUMNS IN {fq(t)}").collect()]   # SQL, so VARIANT columns are fine
        for (tt, col), cc in COLUMN_COMMENTS.items():
            if tt == t and col in cols:
                spark.sql(f"ALTER TABLE {fq(t)} ALTER COLUMN `{col}` COMMENT '{_q(cc)}'")

hexid = lambda: uuid.uuid4().hex
instructions = f"""This agent answers questions about documents received by a manufacturer's shared mailboxes (orders@, ap@, quality@, engineering@) and the exceptions found in them.
- Customer orders: {fq('customer_order_lines')} (one row per PO line; line_value = quantity x unit_price). Customers in {fq('customers')}.
- Supplier invoices: {fq('supplier_invoice_lines')}; our POs in {fq('purchase_orders')}; receipts in {fq('goods_receipts')}.
- Quality: {fq('inspection_results')} (measured_min / measured_max vs spec in {fq('component_specs')}); data sheets in {fq('product_specs')}.
- Engineering changes: {fq('engineering_changes')} (old_part_number replaced by new_part_number).
- Exceptions: {fq('document_exceptions')}; what to do about them: {fq('exception_decisions')} joined on exception_id (hold if hold_probability >= 0.5); drafted emails in {fq('draft_replies')}.
- "Today" / "the daily job" means rows where processed_by = 'daily job' in {fq('documents')}.
- Always state the date range and filters used, and round money to 2 decimals."""
def genie_space(tables):
    return {
        "version": 2,
        "config": {"sample_questions": sorted([{"id": hexid(), "question": [x]} for x in GENIE_QUESTIONS], key=lambda x: x["id"])},
        "data_sources": {"tables": [{"identifier": fq(t)} for t in sorted(tables)]},
        "instructions": {
            "text_instructions": [{"id": hexid(), "content": [instructions]}],
            "example_question_sqls": sorted([
                {"id": hexid(), "question": ["Which supplier invoices should we hold?"],
                 "sql": [f"SELECT x.counterparty, x.reference, x.exception_type, x.detail, x.amount_at_risk_usd, round(d.hold_probability, 2) AS hold_probability\n"
                         f"FROM {fq('document_exceptions')} x JOIN {fq('exception_decisions')} d USING (exception_id)\n"
                         f"WHERE x.doc_type = 'supplier_invoice' AND d.hold_probability >= 0.5\nORDER BY d.hold_probability DESC"]},
                {"id": hexid(), "question": ["Customer order value by customer"],
                 "sql": [f"SELECT customer_name, count(DISTINCT customer_po_number) AS purchase_orders, round(sum(line_value), 2) AS order_value_usd\n"
                         f"FROM {fq('customer_order_lines')}\nGROUP BY customer_name ORDER BY order_value_usd DESC"]},
            ], key=lambda x: x["id"]),
        },
    }

def sync_genie():
    """Adds tables that later steps have built to the Genie agent (Genie can only use tables that exist)."""
    want = sorted(t for t in GENIE_TABLES if t in existing_tables())
    g = _api("GET", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}", query={"include_serialized_space": "true"})
    have = sorted(x["identifier"] for x in json.loads(g["serialized_space"])["data_sources"]["tables"])
    if have != [fq(t) for t in want]:
        _api("PATCH", f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}", body={"serialized_space": json.dumps(genie_space(want))})

describe_tables()
GENIE_TITLE = f"Order, Invoice & Quality Documents Genie ({SCHEMA})"
existing = next((s for s in _api("GET", "/api/2.0/genie/spaces").get("spaces", []) if s.get("title") == GENIE_TITLE), None)
if existing:
    GENIE_SPACE_ID = existing["space_id"]
    print(f"Reusing Genie agent '{GENIE_TITLE}'")
else:
    GENIE_SPACE_ID = _api("POST", "/api/2.0/genie/spaces", body={
        "title": GENIE_TITLE,
        "description": "Ask about customer orders, supplier invoices, data sheets, inspection results, engineering changes and the exceptions found in them — built on Databricks AI Functions outputs.",
        "warehouse_id": WAREHOUSE.id,
        "parent_path": f"/Workspace/Users/{w.current_user.me().user_name}",
        "serialized_space": json.dumps(genie_space([t for t in GENIE_TABLES if t in existing_tables()])),
    })["space_id"]
    print(f"Created Genie agent '{GENIE_TITLE}'")
sync_genie()
GENIE_URL = f"{HOST}/genie/rooms/{GENIE_SPACE_ID}{ORG}"

def ask(question, before=None):
    """Ask the Genie agent a question in plain English and show the answer here."""
    describe_tables()
    sync_genie()
    show_genie_answer(ask_genie(question), before=before)

displayHTML(button("Open the Genie agent", GENIE_URL) + button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.6 💬 Ask Genie — explore the inbox
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("How many emails did each shared mailbox receive per week over the last 45 days?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.7 🛠 Try it yourself
# MAGIC Open the **`inbox`** volume from **step 0.4** and preview a few PDFs. Find the **Great Lakes Steel & Bar** invoice with the yellow *banking details have changed* box, and a customer PO from **Helix Robotics**. Can you spot by eye which documents have problems? The pipeline you build in sections 1–4 does that for **every** document, every day.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.8 Bring your own data
# MAGIC Every downstream cell reads only the **`emails`** table, the **`inbox`** volume and the ERP master tables. To run on real documents, skip steps 0.1–0.2, land your attachments in the volume and point the names at your own tables with views. Minimum columns:
# MAGIC
# MAGIC | Table / location | Required columns | Typical source |
# MAGIC |---|---|---|
# MAGIC | `emails` (change data feed on) | `email_id, received_ts, mailbox, from_name, from_address, subject, body, attachment_name, attachment_path` (the full `/Volumes/<catalog>/<schema>/inbox/...` path), `email_text` (sender + subject + body — what the `classify_emails` task reads) | Microsoft Graph / Gmail API job, Power Automate or Logic Apps rule, AP-automation export |
# MAGIC | `inbox` volume | PDF, DOCX, PPTX, JPG / PNG / TIFF (scans and phone photos work too) | Same job — save each attachment, store its path in `emails.attachment_path` |
# MAGIC | `item_master` | `part_number, status, list_price, lead_time_days, unit_cost, supplier_id, supplier_item_code, rohs_required, min_temp_rating_c` | ERP (SAP, Oracle, NetSuite, Epicor, Plex, Infor) |
# MAGIC | `customers` / `suppliers` | `customer_id, customer_name, email_domain, discount_pct` / `supplier_id, supplier_name, email_domain, bank_account_last4` | ERP business partners / vendor master |
# MAGIC | `purchase_orders` / `goods_receipts` | `po_number, supplier_id, line_number, part_number, supplier_item_code, qty_ordered, unit_price` / `po_number, line_number, qty_received` | ERP purchasing & receiving |
# MAGIC | `component_specs` | `part_number, characteristic, nominal_mm, tol_mm` | PLM / QMS (drawing characteristics) |
# MAGIC
# MAGIC ```sql
# MAGIC -- Example: point the notebook at your ERP vendor master
# MAGIC CREATE OR REPLACE VIEW suppliers AS
# MAGIC SELECT vendor_id AS supplier_id, vendor_name AS supplier_name, lower(email_domain) AS email_domain,
# MAGIC        right(bank_account, 4) AS bank_account_last4, payment_terms, country
# MAGIC FROM my_catalog.erp.vendor_master;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.9 Start parsing the inbox now — in the background
# MAGIC Parsing is the one step worth starting early. A brand-new **Document Parsing** task (the `ai_parse_document` AI Function task you'll meet in section 2) has a **one-time start-up of about 15–20 minutes on its first run**, no matter how many files it parses — after that it's fast (the daily run in section 9 parses today's mail in about 2 minutes, and the function itself parses all ~70 PDFs in under a minute).
# MAGIC
# MAGIC So step 0.9 creates the job with just that one task and **starts it now, without waiting**. While you work through sections 1–3, it parses the whole inbox. In step 4.1 you add the other 12 tasks to **the same job**, and the backfill in step 4.2 picks up where this run left off.

# COMMAND ----------

# The parse task: every new or changed file in the inbox volume -> the parsed_documents table (managed by the task).
parse_task = {
    "task_key": "parse_attachments",
    "ai_function_task": {
        "volume": {"volume_path": INBOX},                         # all sub-folders
        "output": {"catalog": CATALOG, "schema": SCHEMA, "table": "parsed_documents"},
        "ai_parse_document": {"version": "2.0"},                  # no AI figure descriptions needed for business documents
    },
    "max_retries": 1, "min_retry_interval_millis": 60_000, "timeout_seconds": 7200,
}
early_settings = {"name": JOB_NAME, "max_concurrent_runs": 1, "queue": {"enabled": True},
                  "tags": {"industry_play": "mfg-document-intelligence"}, "tasks": [parse_task]}
existing_jobs = _api("GET", "/api/2.2/jobs/list", query={"name": JOB_NAME}).get("jobs", [])
if existing_jobs:
    JOB_ID = existing_jobs[0]["job_id"]      # already built (e.g. re-running the notebook): leave it as is
else:
    JOB_ID = _api("POST", "/api/2.2/jobs/create", body=early_settings)["job_id"]
active = _api("GET", "/api/2.2/jobs/runs/list", query={"job_id": JOB_ID, "active_only": "true"}).get("runs", [])
EARLY_RUN = active[0] if active else _api("GET", "/api/2.2/jobs/runs/get", query={
    "run_id": _api("POST", "/api/2.2/jobs/run-now", body={"job_id": JOB_ID})["run_id"]})
displayHTML(f"""<div style="{FONT};font-size:14px">Parsing started in the background — continue with section 1.
  {button("Watch the parsing run", EARLY_RUN["run_page_url"])}</div>""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1 — Design the AI steps (sections 1–3)
# MAGIC In production, the three **document-intelligence** functions — `ai_classify`, `ai_parse_document` and `ai_extract` — run as native **AI Function tasks** in a Lakeflow Job (*Document intelligence* in the Jobs UI). Each task runs its function on every new file in a volume or every new row of a table, writes the results to a Unity Catalog table it manages, and on later runs processes **only what's new or changed** — no checkpoints or `WHERE NOT EXISTS` to write.
# MAGIC
# MAGIC So in sections 1–3 you **design** each task: define its labels or schema once in Python, **preview** the exact same settings on a few emails or one document with the SQL function (seconds, a few cents), and look at the task definition the job will use. Section 4 builds the job and runs the **one-time backfill**; sections 5–9 explore the results and ask Genie.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Route the inbox — `ai_classify`
# MAGIC
# MAGIC Before anyone reads an attachment, decide **what each email is**. `ai_classify` takes the sender, subject and body and returns one label from a list you define — here, the six workflows the business runs. Labels come with **plain-English descriptions** and optional **instructions** (*a manufacturer's shared mailboxes*), and with **confidence scores** turned on every answer comes with a score from 0 to 1 — no training data, no model to maintain.
# MAGIC
# MAGIC This catches what manual triage misses: an invoice sent to `orders@`, a certificate sent to `ap@`, or an order change written only in the email body.

# COMMAND ----------

# One definition, used twice: by the preview in step 1.2 and by the classify_emails task of the job (step 4.1).
EMAIL_LABELS = {
    "customer_purchase_order": "A customer places, changes or confirms an order with us (PO attached or details in the email body)",
    "supplier_invoice": "A supplier bills us, sends a payment reminder or changes payment details",
    "product_data_sheet": "A supplier sends a technical data sheet / specification for a material or component",
    "quality_document": "Certificate of conformance, inspection or test report, or a quality notification about delivered parts",
    "engineering_change": "Engineering change notice, drawing revision or part supersession",
    "no_action": "Newsletters, marketing, out-of-office, statements with no balance due, IT or HR notices, thank-you notes",
}
EMAIL_INSTRUCTIONS = "Emails received by the shared mailboxes of Ridgeline Precision Components, a manufacturer of bearing housings, shafts and gearboxes."

classify_emails_task = {
    "task_key": "classify_emails",
    "ai_function_task": {
        "delta_table": {"table_name": f"{CATALOG}.{SCHEMA}.emails", "input_column": "email_text", "merge_key_columns": ["email_id"]},
        "output": {"catalog": CATALOG, "schema": SCHEMA, "table": "email_classifications"},
        "ai_classify": {"version": "2.1", "instructions": EMAIL_INSTRUCTIONS, "enable_confidence_scores": True,
                        "inline_labels": {"labels": [{"name": k, "description": v} for k, v in EMAIL_LABELS.items()]}},
    },
}
show_task(classify_emails_task, "Classification", "Classifies every new email in the `emails` table into a workflow.")

# COMMAND ----------

# The SQL function with the exact labels and instructions of the task (passed as :named parameters).
preview = preview_sql("""
WITH sample AS (SELECT * FROM emails ORDER BY email_id LIMIT 10)
SELECT email_id, split(mailbox, '@')[0] AS mailbox, subject,
       result:response[0]:value::STRING                    AS workflow,
       round(result:response[0]:confidence_score::DOUBLE, 2) AS confidence
FROM (SELECT *, ai_classify(email_text, :labels,
                            map('version', '2.1', 'enableConfidenceScores', 'true', 'instructions', :instructions)) AS result
      FROM sample)
""", labels=json.dumps(EMAIL_LABELS), instructions=EMAIL_INSTRUCTIONS)
display(preview)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3 🛠 Try it yourself
# MAGIC In **step 1.1**, add a label such as `"order_change_request": "A customer asks to change quantity or date on an existing order"` and re-run **steps 1.1–1.2**. Iterating here costs a few cents; the job picks up the new labels when you build it in step 4.1. *(Once the job has run, changed labels apply only to emails processed after the change — to re-classify everything, give the task's output a new table name.)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Parse every attachment — `ai_parse_document`
# MAGIC
# MAGIC `ai_parse_document` reads PDFs, Office files and images (scans and phone photos included) and returns a **layout-aware** structure: every page, every element (title, text, **table**, figure, header, footer) with its bounding box and a confidence score — tables come back as HTML with rows and columns intact. No OCR templates per supplier, no zones to draw.
# MAGIC
# MAGIC * **2.1** shows two tasks: **`parse_attachments`** (Document Parsing, defined and started in step 0.9) reads every new PDF in the `inbox` volume; **`classify_documents`** (Classification) reads the parsed output and decides what each **document** is — chained with `{{tasks.parse_attachments.output.table_name}}`
# MAGIC * **2.2** follows one real invoice through the parser so you can see exactly what it returns
# MAGIC * **2.3** classifies that same document

# COMMAND ----------

# parse_task was defined — and started — in step 0.9. Here you add the classification task that reads its output.
DOC_LABELS = {
    "customer_purchase_order": "Purchase order issued by a customer to us",
    "supplier_invoice": "Invoice from a supplier billing us",
    "product_data_sheet": "Technical data sheet / specification of a material or component",
    "quality_document": "Certificate of conformance and/or dimensional inspection report for a delivered lot",
    "engineering_change": "Engineering change notice: part supersession or drawing revision",
    "other": "Anything else",
}
classify_documents_task = {
    "task_key": "classify_documents",
    "depends_on": [{"task_key": "parse_attachments"}],
    "ai_function_task": {
        "delta_table": {"table_name": "{{tasks.parse_attachments.output.table_name}}", "input_column": "response", "merge_key_columns": ["id"]},
        "output": {"catalog": CATALOG, "schema": SCHEMA, "table": "document_classifications"},
        "ai_classify": {"version": "2.1", "enable_confidence_scores": True,
                        "inline_labels": {"labels": [{"name": k, "description": v} for k, v in DOC_LABELS.items()]}},
    },
}
show_task(parse_task, "Document Parsing", "Parses every new PDF in the inbox volume — started in step 0.9, it may still be running.")
show_task(classify_documents_task, "Classification", "Classifies every newly parsed document — reads the parsing task's output table.")

# COMMAND ----------

# Parse one attachment with the SQL function and show every element it found — tables come back as HTML.
SAMPLE_PATH = spark.sql("SELECT attachment_path FROM emails WHERE attachment_name LIKE 'Invoice_GLS%' ORDER BY received_ts LIMIT 1").first()[0]
SAMPLE_PARSED = preview_sql(f"""
SELECT to_json(ai_parse_document(content, map('version', '2.0'))) AS parsed
FROM read_files('{SAMPLE_PATH}', format => 'binaryFile')
""").first().parsed
els = json.loads(SAMPLE_PARSED)["document"]["elements"]
TYPE_COLOR = {"table": "#FF3621", "title": "#1B3139", "page_header": "#5A6F77", "text": "#2272B4", "section_header": "#00A972"}
items_html = "".join(
    f"""<div style="display:flex;gap:10px;margin:6px 0;align-items:flex-start">
          <div style="min-width:96px;font-size:11px;font-weight:700;color:white;background:{TYPE_COLOR.get(e['type'], '#888')};border-radius:4px;padding:2px 6px;text-align:center">{e['type']}</div>
          <div style="font-size:12px">{e['content'] if e['type'] == 'table' else html.escape(e['content'] or '').replace(chr(10), '<br>')}</div></div>"""
    for e in els)
displayHTML(f"""<style>.pd table{{border-collapse:collapse}} .pd td,.pd th{{border:1px solid #ccc;padding:3px 8px;font-size:12px}}</style>
<div class="pd" style="{FONT};border:1px solid #e0e0e0;border-radius:10px;padding:14px;max-width:1000px">
  <div style="font-size:15px;font-weight:700">{html.escape(SAMPLE_PATH.rsplit('/', 1)[-1])} — {len(els)} elements</div>{items_html}</div>""")

# COMMAND ----------

display(preview_sql("""
SELECT result:response[0]:value::STRING AS doc_type, round(result:response[0]:confidence_score::DOUBLE, 2) AS confidence
FROM (SELECT ai_classify(parse_json(:doc), :labels, map('version', '2.1', 'enableConfidenceScores', 'true')) AS result)
""", doc=SAMPLE_PARSED, labels=json.dumps(DOC_LABELS)))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4 🛠 Try it yourself
# MAGIC In **step 2.2**, change `'Invoice_GLS%'` to `'Datasheet%'` or `'CoC%'` and re-run **steps 2.2–2.3**. Notice how the dimension and inspection tables come back as rows and columns — that's what makes extraction reliable. *(Going further: add `"image_output_path": "/Volumes/…/inbox_pages"` to the parse task to save an image of every page.)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Extract line items, specs & measurements — `ai_extract`
# MAGIC
# MAGIC `ai_extract` turns each parsed document into **typed, structured fields** from a JSON schema you write: strings, numbers, booleans, **enums** and **nested arrays** (one entry per PO line, invoice line or measured characteristic), each with a plain-English **description** the model follows. It reads the `ai_parse_document` output directly, so table structure isn't lost. With **confidence scores** on, every extracted value comes back with a score from 0 to 1 — the basis for a human **review queue** (step 6.4) instead of re-keying everything.
# MAGIC
# MAGIC There is **one extraction task per document type**, each with its own schema. Each task reads a small input table holding just its documents — `prepare_documents` (step 5.1) fills those five tables after the two classification tasks, so an invoice never runs through the PO schema.
# MAGIC
# MAGIC | Task | Document type | Extracted |
# MAGIC |---|---|---|
# MAGIC | `extract_purchase_orders` | Customer purchase orders | PO number, dates, ship-to, terms + **lines**: our part #, customer part #, qty, unit price |
# MAGIC | `extract_invoices` | Supplier invoices | Invoice & PO number, remit-to account, bank-change notice, freight, total + **lines** |
# MAGIC | `extract_data_sheets` | Product data sheets | Material, temperature rating, RoHS / REACH, revision + **dimensions** |
# MAGIC | `extract_quality_docs` | Certificates of conformance / inspection reports | PO, part, lot & heat number + **measurements** (min / max / supplier result) |
# MAGIC | `extract_ecns` | Engineering change notices | Old → new part number and revision, effective date, reason |

# COMMAND ----------

EXTRACT_SPECS = {
    "purchase_orders": {
        "doc_type": "customer_purchase_order",
        "schema": {
            "customer_name": {
                "type": "string",
                "description": "Company that issued the purchase order (the buyer)"
            },
            "customer_po_number": {
                "type": "string",
                "description": "The buyer purchase order number"
            },
            "po_date": {
                "type": "string",
                "description": "Order date as YYYY-MM-DD"
            },
            "requested_delivery_date": {
                "type": "string",
                "description": "Required / delivery date as YYYY-MM-DD (earliest if several)"
            },
            "ship_to": {
                "type": "string"
            },
            "payment_terms": {
                "type": "string"
            },
            "currency": {
                "type": "enum",
                "labels": [
                    "USD",
                    "EUR",
                    "CAD",
                    "MXN"
                ]
            },
            "lines": {
                "type": "array",
                "description": "One entry per ordered item. Do not include subtotal, total, tax or freight rows.",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_number": {
                            "type": "integer"
                        },
                        "our_part_number": {
                            "type": "string",
                            "description": "Ridgeline (supplier / vendor) part number, format RPC-####. It may be in its own column or inside the description text, e.g. Vendor mat. RPC-1001"
                        },
                        "customer_part_number": {
                            "type": "string",
                            "description": "The buyer own item / material number"
                        },
                        "description": {
                            "type": "string"
                        },
                        "quantity": {
                            "type": "number"
                        },
                        "unit_price": {
                            "type": "number",
                            "description": "Price per unit"
                        }
                    }
                }
            },
            "order_total": {
                "type": "number"
            },
            "special_instructions": {
                "type": "string",
                "description": "Notes or conditions written on the PO, if any"
            }
        },
        "instructions": "Customer purchase orders sent to Ridgeline Precision Components, the supplier. Return dates as YYYY-MM-DD. Dates written with dots (DD.MM.YYYY, e.g. 05.11.2026 = 5 November 2026) are day-first; dates with slashes (MM/DD/YYYY) are month-first; some are written in words."
    },
    "invoices": {
        "doc_type": "supplier_invoice",
        "schema": {
            "supplier_name": {
                "type": "string",
                "description": "Company issuing the invoice"
            },
            "invoice_number": {
                "type": "string"
            },
            "invoice_date": {
                "type": "string",
                "description": "Invoice date as YYYY-MM-DD"
            },
            "po_number": {
                "type": "string",
                "description": "Buyer (Ridgeline) purchase order number referenced, format PO-#####; empty if none"
            },
            "remit_to_account_last4": {
                "type": "string",
                "description": "Last 4 digits of the bank account in the remittance instructions"
            },
            "bank_change_notice": {
                "type": "boolean",
                "description": "True only if the invoice says bank or remittance details have changed"
            },
            "freight": {
                "type": "number"
            },
            "invoice_total": {
                "type": "number",
                "description": "Total amount due"
            },
            "lines": {
                "type": "array",
                "description": "One entry per billed item. Do not include subtotal, freight, tax or total rows.",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_code": {
                            "type": "string",
                            "description": "Supplier item code / clave"
                        },
                        "description": {
                            "type": "string"
                        },
                        "quantity": {
                            "type": "number"
                        },
                        "unit_price": {
                            "type": "number"
                        },
                        "amount": {
                            "type": "number"
                        }
                    }
                }
            }
        },
        "instructions": "Supplier invoices billed to Ridgeline Precision Components. Some are bilingual Spanish / English."
    },
    "data_sheets": {
        "doc_type": "product_data_sheet",
        "schema": {
            "supplier_name": {
                "type": "string"
            },
            "product_name": {
                "type": "string"
            },
            "supplier_part_number": {
                "type": "string"
            },
            "for_part_number": {
                "type": "string",
                "description": "Customer (Ridgeline) part reference, format CMP-####"
            },
            "revision": {
                "type": "string"
            },
            "material": {
                "type": "string"
            },
            "hardness": {
                "type": "string"
            },
            "max_operating_temp_c": {
                "type": "number",
                "description": "Upper end of the operating temperature range in deg C; empty if n/a"
            },
            "rohs_compliant": {
                "type": "boolean",
                "description": "False if the sheet says not compliant with RoHS"
            },
            "reach_compliant": {
                "type": "boolean",
                "description": "False if a REACH SVHC is present above 0.1%"
            },
            "change_note": {
                "type": "string",
                "description": "What changed in this revision, if stated"
            },
            "dimensions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "characteristic": {
                            "type": "string"
                        },
                        "nominal_mm": {
                            "type": "number"
                        },
                        "tolerance_mm": {
                            "type": "number"
                        }
                    }
                }
            }
        },
        "instructions": None
    },
    "quality_docs": {
        "doc_type": "quality_document",
        "schema": {
            "supplier_name": {
                "type": "string"
            },
            "po_number": {
                "type": "string",
                "description": "Customer (Ridgeline) PO number, format PO-#####"
            },
            "part_number": {
                "type": "string",
                "description": "Ridgeline part number, format CMP-####"
            },
            "lot_number": {
                "type": "string",
                "description": "Lot number; empty if missing or shown as a dash"
            },
            "heat_number": {
                "type": "string"
            },
            "quantity_shipped": {
                "type": "number"
            },
            "inspection_date": {
                "type": "string",
                "description": "YYYY-MM-DD"
            },
            "supplier_notes": {
                "type": "string",
                "description": "Any deviation or remark written by the supplier"
            },
            "measurements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "characteristic": {
                            "type": "string"
                        },
                        "nominal_mm": {
                            "type": "number"
                        },
                        "tolerance_mm": {
                            "type": "number",
                            "description": "Plus/minus tolerance"
                        },
                        "measured_min_mm": {
                            "type": "number"
                        },
                        "measured_max_mm": {
                            "type": "number"
                        },
                        "supplier_result": {
                            "type": "enum",
                            "labels": [
                                "PASS",
                                "FAIL"
                            ]
                        }
                    }
                }
            }
        },
        "instructions": None
    },
    "ecns": {
        "doc_type": "engineering_change",
        "schema": {
            "ecn_number": {
                "type": "string"
            },
            "effective_date": {
                "type": "string",
                "description": "YYYY-MM-DD"
            },
            "change_class": {
                "type": "string"
            },
            "old_part_number": {
                "type": "string"
            },
            "old_revision": {
                "type": "string"
            },
            "new_part_number": {
                "type": "string"
            },
            "new_revision": {
                "type": "string"
            },
            "affected_customers": {
                "type": "string"
            },
            "reason": {
                "type": "string",
                "description": "One-sentence summary of the reason for change"
            },
            "open_order_action": {
                "type": "string",
                "description": "What to do with open customer orders for the old part"
            }
        },
        "instructions": None
    }
}

extract_tasks = [{
    "task_key": f"extract_{name}",
    "depends_on": [{"task_key": "prepare_documents"}],
    "ai_function_task": {
        "delta_table": {"table_name": f"{CATALOG}.{SCHEMA}.extract_input_{name}", "input_column": "parsed", "merge_key_columns": ["doc_id"]},
        "output": {"catalog": CATALOG, "schema": SCHEMA, "table": f"extracted_{name}"},
        "ai_extract": {"version": "2.1", "schema": json.dumps(spec["schema"]), "enable_confidence_scores": True,
                       **({"instructions": spec["instructions"]} if spec["instructions"] else {})},
    },
} for name, spec in EXTRACT_SPECS.items()]
for t in extract_tasks:
    show_task(t, "Information Extraction", f"Extracts {EXTRACT_SPECS[t['task_key'][8:]]['doc_type'].replace('_', ' ')}s: {', '.join(EXTRACT_SPECS[t['task_key'][8:]]['schema'])}.")

# COMMAND ----------

spec = EXTRACT_SPECS["invoices"]
extracted = json.loads(preview_sql("""
SELECT to_json(ai_extract(parse_json(:doc), :schema,
                          map('version', '2.1', 'enableConfidenceScores', 'true', 'instructions', :instructions))) AS result
""", doc=SAMPLE_PARSED, schema=json.dumps(spec["schema"]), instructions=spec["instructions"] or "").first().result)["response"]

def flat(obj, prefix=""):
    # {field: {value, confidence_score}} (and arrays of those) -> rows of (field, value, confidence)
    if isinstance(obj, dict) and "value" in obj and "confidence_score" in obj:
        yield (prefix, obj["value"], obj["confidence_score"])
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj, 1):
            yield from flat(v, f"{prefix}[{i}]")
display(spark.createDataFrame([(f, None if v is None else str(v), float(c) if c is not None else None) for f, v, c in flat(extracted)],
                              "field string, value string, confidence double"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3 🛠 Try it yourself
# MAGIC In **step 3.1**, add a field to the `invoices` schema — for example `"payment_terms": {"type": "string"}` or `"due_date": {"type": "string", "description": "YYYY-MM-DD"}` — and re-run **steps 3.1–3.2**. No retraining, no template — just a new line of JSON. *(Precision mode: add `"mode": "MODE_PRECISION"` to a task's `ai_extract` settings for complex fields and long documents.)*

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2 — Build the pipeline and run the one-time backfill (section 4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. One job: AI Function tasks + notebook tasks
# MAGIC
# MAGIC Step 4.1 assembles everything into one **Lakeflow Job** — **8 AI Function tasks** (the ones you designed in sections 1–3) and **5 notebook tasks** for the steps that are plain SQL or use `ai_decide` / `ai_gen`, which aren't AI Function task types:
# MAGIC
# MAGIC ```
# MAGIC classify_emails ──────────────┐ (AI Function · Classification)
# MAGIC parse_attachments ─> classify_documents ─┴─> prepare_documents ─┬─> extract_purchase_orders ─┐
# MAGIC  (Document Parsing)  (Classification)        (notebook, 5.1)    ├─> extract_invoices ────────┤
# MAGIC                                                                 ├─> extract_data_sheets ─────┼─> flatten_extractions ─> validate_exceptions ─> decide_exceptions ─> draft_replies
# MAGIC                                                                 ├─> extract_quality_docs ────┤    (notebook, 6.1)       (notebook, 7.1)       (ai_decide, 8.1)    (ai_gen, 8.3)
# MAGIC                                                                 └─> extract_ecns ────────────┘
# MAGIC                                                                      (Information Extraction ×5)
# MAGIC ```
# MAGIC
# MAGIC | | AI Function tasks (parse · classify · extract) | Notebook tasks (prepare · flatten · validate · decide · draft) |
# MAGIC |---|---|---|
# MAGIC | **What runs** | The task you defined in sections 1–3 | Cells of this notebook whose first line is `▶ DAILY JOB · task: <name>` (plus S.1–S.2) |
# MAGIC | **Incremental** | Built in: only new or changed files / rows; deleted inputs remove their results | `MERGE` / `WHERE NOT EXISTS`, so AI calls never repeat |
# MAGIC | **Output** | A managed table: `id` or your ID column, `response` (VARIANT), `error` | Regular Delta tables |
# MAGIC | **Monitoring** | Progress %, rows succeeded / failed and ETA in the run graph; *Sample Data* tab | Notebook output per task |
# MAGIC
# MAGIC Every task retries once, a failed run can be **repaired** (only the failed task and what depends on it re-run), and the job runs on **serverless** compute. It's created with a **6:00 AM schedule, paused**.
# MAGIC
# MAGIC * **4.1** generates one small notebook per notebook task (in a subfolder next to this notebook) and creates or updates the job. Re-running 4.1 overwrites those notebooks and the job settings; the current `price_tolerance_pct` and `confidence_threshold` widget values become the job's parameter defaults.
# MAGIC * **4.2** runs the job **now** as the **one-time backfill** — the first run of each AI Function task processes **everything**: all 45 days of mail. If the parsing run from step 0.9 is still going, this run queues behind it, then the parse task only picks up anything new. *Expect about 10 minutes once the parsing from step 0.9 has finished (with Run all, nothing overlaps, so it waits up to ~20 more minutes for that run); later daily runs process only new mail in about 10 minutes.*
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → the **AI Functions** previews (including **`ai_decide`** and **`ai_gen`**) and **Lakeflow AI Function Task**. The AI Function tasks need **Lakeflow AI Function Task** (Beta); without it, step 4.1 fails with an *unknown task type* error.

# COMMAND ----------

import base64
from databricks.sdk.service.workspace import ExportFormat, ImportFormat

# 1) Notebook tasks: copy this notebook's ▶ DAILY JOB cells, grouped by task, into one small notebook per task
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
this_nb = json.loads(base64.b64decode(w.workspace.export(nb_path, format=ExportFormat.JUPYTER).content))
TAG = re.compile(r"^(?:%sql\s*\n)?(?:--|#) ▶ DAILY JOB · (every task|task: (\w+))")
every_task, task_cells = [], {}
for c in this_nb["cells"]:
    m = TAG.match("".join(c["source"]))
    if m:
        cell = dict(c, outputs=[], execution_count=None)
        (every_task if m.group(1) == "every task" else task_cells.setdefault(m.group(2), [])).append(cell)
NOTEBOOK_TASKS = {   # task -> the tasks it waits for
    "prepare_documents": ["classify_emails", "classify_documents"],
    "flatten_extractions": [t["task_key"] for t in extract_tasks],
    "validate_exceptions": ["flatten_extractions"],
    "decide_exceptions": ["validate_exceptions"],
    "draft_replies": ["decide_exceptions"],
}
assert set(NOTEBOOK_TASKS) == set(task_cells), f"tagged cells {sorted(task_cells)} don't match {sorted(NOTEBOOK_TASKS)}"
JOB_FOLDER = nb_path.rsplit("/", 1)[0] + f"/Document intelligence pipeline - {SCHEMA}"
w.workspace.mkdirs(JOB_FOLDER)
meta = this_nb["metadata"].get("application/vnd.databricks.v1+notebook", {})
task_paths = {}
for i, task in enumerate(NOTEBOOK_TASKS, 1):
    header = {"cell_type": "markdown", "metadata": {}, "source": [
        f"# Notebook task `{task}`\n",
        f"Generated by step 4.1 of `{nb_path}` from its **▶ DAILY JOB** cells. It only processes rows it hasn't seen before. ",
        "To change the logic, edit the source notebook and re-run step 4.1."]}
    nb = dict(this_nb, cells=[header] + every_task + task_cells[task])
    nb["metadata"] = dict(this_nb["metadata"], **{"application/vnd.databricks.v1+notebook": {**meta, "notebookName": f"{i:02d} {task}", "widgets": {}}})
    task_paths[task] = f"{JOB_FOLDER}/{i:02d} {task}"
    w.workspace.import_(task_paths[task], content=base64.b64encode(json.dumps(nb).encode()).decode(), format=ImportFormat.JUPYTER, overwrite=True)

# 2) All 13 tasks: the AI Function tasks from sections 1-3 + the notebook tasks
notebook_tasks = [{"task_key": t, "depends_on": [{"task_key": d} for d in deps], "notebook_task": {"notebook_path": task_paths[t]}}
                  for t, deps in NOTEBOOK_TASKS.items()]
TASKS = [classify_emails_task, parse_task, classify_documents_task] + extract_tasks + notebook_tasks
for t in TASKS:
    t.update(max_retries=1, min_retry_interval_millis=60_000, timeout_seconds=7200)

# 3) Create or update the job (Jobs API 2.2): serverless, job parameters flow to every notebook task, 6:00 AM schedule PAUSED
job_settings = {
    "name": JOB_NAME, "max_concurrent_runs": 1, "queue": {"enabled": True},   # a new run waits for the parsing run from step 0.9
    "tags": {"industry_play": "mfg-document-intelligence"},
    "parameters": [{"name": k, "default": v} for k, v in {
        "catalog": CATALOG, "schema": SCHEMA, "run_label": "daily job",
        "price_tolerance_pct": dbutils.widgets.get("price_tolerance_pct"),
        "confidence_threshold": dbutils.widgets.get("confidence_threshold")}.items()],
    "schedule": {"quartz_cron_expression": "0 0 6 * * ?", "timezone_id": "America/New_York", "pause_status": "PAUSED"},
    "tasks": TASKS,
}
existing_jobs = _api("GET", "/api/2.2/jobs/list", query={"name": JOB_NAME}).get("jobs", [])
if existing_jobs:
    JOB_ID = existing_jobs[0]["job_id"]
    _api("POST", "/api/2.2/jobs/reset", body={"job_id": JOB_ID, "new_settings": job_settings})
else:
    JOB_ID = _api("POST", "/api/2.2/jobs/create", body=job_settings)["job_id"]
JOB_URL = f"{HOST}/jobs/{JOB_ID}{ORG}"

kind = lambda t: ("AI Function · " + {"ai_parse_document": "Document Parsing", "ai_classify": "Classification", "ai_extract": "Information Extraction"}[
                   next(k for k in t["ai_function_task"] if k.startswith("ai_"))]) if "ai_function_task" in t else "Notebook"
rows_html = "".join(f"<tr><td style='padding:2px 10px'><b>{t['task_key']}</b></td><td style='padding:2px 10px'>{kind(t)}</td>"
                    f"<td style='padding:2px 10px'>{', '.join(d['task_key'] for d in t.get('depends_on', [])) or '—'}</td></tr>" for t in TASKS)
displayHTML(f"""<div style="{FONT}"><div style="font-size:15px;font-weight:700;margin-bottom:6px">Job ready — {len(TASKS)} tasks, serverless, 6:00 AM (paused)</div>
  <table style="font-size:12px;border-collapse:collapse"><tr><th align=left style='padding:2px 10px'>Task</th><th align=left style='padding:2px 10px'>Type</th>
  <th align=left style='padding:2px 10px'>Waits for</th></tr>{rows_html}</table>
  {button("Open the job", JOB_URL)}{button("Open the notebook tasks", f"{HOST}/#workspace{JOB_FOLDER}", "#1B3139")}</div>""")

# COMMAND ----------

def run_job(run_label):
    """Runs the job now with the given run_label, waits for it, and shows every task's result and duration."""
    run_id = _api("POST", "/api/2.2/jobs/run-now", body={"job_id": JOB_ID, "job_parameters": {"run_label": run_label}})["run_id"]
    t0, last = time.time(), ""
    while True:
        r = _api("GET", "/api/2.2/jobs/runs/get", query={"run_id": run_id})
        if r["state"]["life_cycle_state"] in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        running = ("queued — waiting for the parsing run from step 0.9 to finish" if r["state"]["life_cycle_state"] == "QUEUED" else
                   ", ".join(t["task_key"] for t in r.get("tasks", []) if t.get("state", {}).get("life_cycle_state") == "RUNNING"))
        if running != last:
            print(f"{round((time.time() - t0) / 60, 1):>5} min  running: {running or '(starting)'}")
            last = running
        time.sleep(20)
    tasks_html = "".join(
        f"<tr><td style='padding:2px 10px'>{t['task_key']}</td><td style='padding:2px 10px'>{t['state'].get('result_state', '')}</td>"
        f"<td style='padding:2px 10px' align=right>{round((t.get('end_time', 0) - t.get('start_time', 0)) / 60000, 1)} min</td></tr>"
        for t in sorted(r.get("tasks", []), key=lambda t: t.get("start_time", 0)))
    displayHTML(f"""<div style="{FONT};font-size:14px">Run <b>{run_label}</b> finished: <b>{r['state'].get('result_state')}</b> in {round((time.time() - t0) / 60, 1)} minutes
      <table style="font-size:12px;margin:6px 0">{tasks_html}</table>{button("Open the job run (see the graph)", r["run_page_url"])}</div>""")
    assert r["state"].get("result_state") == "SUCCESS", f"Job run did not succeed — open {r['run_page_url']} and use Repair run"
    return r

backfill_run = run_job("one-time backfill")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3 — Explore the results and ask Genie (sections 5–9)
# MAGIC The backfill wrote every table below. Each **▶ DAILY JOB** cell in this part is the exact code of a notebook task — you can re-run it here safely (it skips anything already processed) or change it and re-run step 4.1 to update the job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The inbox and its documents
# MAGIC
# MAGIC ### 5.1 What the `prepare_documents` task does
# MAGIC After the two classification tasks, this notebook task joins their outputs back to the emails and parsed files — the **`email_routing`** and **`documents`** tables everyone queries — and fills the five small input tables the `ai_extract` tasks read (change data feed on, so each extraction task only sees new documents). Plain SQL, no AI.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ▶ DAILY JOB · task: prepare_documents
# MAGIC -- Every email with its ai_classify workflow (from the classify_emails task)
# MAGIC CREATE OR REPLACE TABLE email_routing COMMENT 'Every email with the workflow assigned by the classify_emails AI Function task' AS
# MAGIC SELECT e.email_id, e.received_ts, e.mailbox, e.from_address, e.subject, e.attachment_path IS NOT NULL AS has_attachment,
# MAGIC        c.response:response[0]:value::STRING                 AS workflow,
# MAGIC        c.response:response[0]:confidence_score::DOUBLE      AS workflow_confidence
# MAGIC FROM emails e JOIN email_classifications c USING (email_id)
# MAGIC WHERE c.error IS NULL;
# MAGIC
# MAGIC -- One row per parsed attachment with its document type (from the parse_attachments and classify_documents tasks).
# MAGIC -- MERGE keeps the first run that saw each document in processed_by ('one-time backfill' or 'daily job').
# MAGIC CREATE TABLE IF NOT EXISTS documents (
# MAGIC   doc_id STRING, email_id STRING, file_name STRING, mailbox STRING, sender STRING, received_ts TIMESTAMP,
# MAGIC   email_workflow STRING, doc_type STRING, doc_type_confidence DOUBLE, page_count INT, processed_by STRING, processed_at TIMESTAMP)
# MAGIC COMMENT 'One row per parsed attachment with the doc_type assigned by the classify_documents AI Function task';
# MAGIC
# MAGIC MERGE INTO documents t
# MAGIC USING (
# MAGIC   SELECT p.id AS doc_id, e.email_id, regexp_extract(p.id, '([^/]+)$', 1) AS file_name, e.mailbox, e.from_address AS sender,
# MAGIC          e.received_ts, r.workflow AS email_workflow,
# MAGIC          c.response:response[0]:value::STRING AS doc_type, c.response:response[0]:confidence_score::DOUBLE AS doc_type_confidence,
# MAGIC          size(CAST(p.response:document:pages AS ARRAY<VARIANT>)) AS page_count, run_label AS processed_by, current_timestamp() AS processed_at
# MAGIC   FROM parsed_documents p
# MAGIC   JOIN document_classifications c ON c.id = p.id
# MAGIC   LEFT JOIN emails e ON e.attachment_path = replace(p.id, 'dbfs:', '')
# MAGIC   LEFT JOIN email_routing r ON r.email_id = e.email_id
# MAGIC   WHERE p.error IS NULL AND c.error IS NULL
# MAGIC ) s ON t.doc_id = s.doc_id
# MAGIC WHEN NOT MATCHED THEN INSERT *;
# MAGIC
# MAGIC -- The five inputs of the ai_extract tasks: one table per document type, with change data feed on
# MAGIC CREATE TABLE IF NOT EXISTS extract_input_purchase_orders (doc_id STRING, parsed VARIANT) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC CREATE TABLE IF NOT EXISTS extract_input_invoices        (doc_id STRING, parsed VARIANT) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC CREATE TABLE IF NOT EXISTS extract_input_data_sheets     (doc_id STRING, parsed VARIANT) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC CREATE TABLE IF NOT EXISTS extract_input_quality_docs    (doc_id STRING, parsed VARIANT) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC CREATE TABLE IF NOT EXISTS extract_input_ecns            (doc_id STRING, parsed VARIANT) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC MERGE INTO extract_input_purchase_orders t USING (SELECT d.doc_id, p.response AS parsed FROM documents d JOIN parsed_documents p ON p.id = d.doc_id WHERE d.doc_type = 'customer_purchase_order') s
# MAGIC ON t.doc_id = s.doc_id WHEN NOT MATCHED THEN INSERT *;
# MAGIC MERGE INTO extract_input_invoices t USING (SELECT d.doc_id, p.response AS parsed FROM documents d JOIN parsed_documents p ON p.id = d.doc_id WHERE d.doc_type = 'supplier_invoice') s
# MAGIC ON t.doc_id = s.doc_id WHEN NOT MATCHED THEN INSERT *;
# MAGIC MERGE INTO extract_input_data_sheets t USING (SELECT d.doc_id, p.response AS parsed FROM documents d JOIN parsed_documents p ON p.id = d.doc_id WHERE d.doc_type = 'product_data_sheet') s
# MAGIC ON t.doc_id = s.doc_id WHEN NOT MATCHED THEN INSERT *;
# MAGIC MERGE INTO extract_input_quality_docs t USING (SELECT d.doc_id, p.response AS parsed FROM documents d JOIN parsed_documents p ON p.id = d.doc_id WHERE d.doc_type = 'quality_document') s
# MAGIC ON t.doc_id = s.doc_id WHEN NOT MATCHED THEN INSERT *;
# MAGIC MERGE INTO extract_input_ecns t USING (SELECT d.doc_id, p.response AS parsed FROM documents d JOIN parsed_documents p ON p.id = d.doc_id WHERE d.doc_type = 'engineering_change') s
# MAGIC ON t.doc_id = s.doc_id WHEN NOT MATCHED THEN INSERT *;
# MAGIC
# MAGIC SELECT doc_type, count(*) AS documents FROM documents GROUP BY ALL ORDER BY documents DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 What each mailbox actually received
# MAGIC Each bar is a mailbox, split by the workflow the `classify_emails` task assigned. Look for an invoice color in the `orders` bar and a quality color in the `ap` bar — those are misrouted.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Routing results: what each mailbox actually received, by AI-assigned workflow
# MAGIC SELECT split(mailbox, '@')[0] AS mailbox, workflow, count(*) AS emails
# MAGIC FROM email_routing
# MAGIC GROUP BY ALL
# MAGIC ORDER BY mailbox, emails DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Routing results: what each mailbox actually received, by AI-assigned workflow
# MAGIC SELECT split(mailbox, '@')[0] AS mailbox, workflow, count(*) AS emails
# MAGIC FROM email_routing
# MAGIC GROUP BY ALL
# MAGIC ORDER BY mailbox, emails DESC) SELECT `mailbox`,SUM(`emails`) `column_2604123945`,`workflow` FROM q GROUP BY `mailbox`,`workflow`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.3 The emails manual triage misses
# MAGIC Misrouted emails and requests written only in the email body — each one would otherwise sit in the wrong queue.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The emails manual triage misses: sent to the wrong mailbox, or an actionable request with no attachment
# MAGIC WITH expected AS (
# MAGIC   SELECT *, CASE workflow WHEN 'customer_purchase_order' THEN 'orders' WHEN 'supplier_invoice' THEN 'ap'
# MAGIC                           WHEN 'quality_document' THEN 'quality' ELSE 'engineering' END AS should_go_to
# MAGIC   FROM email_routing WHERE workflow <> 'no_action'
# MAGIC )
# MAGIC SELECT received_ts, split(mailbox, '@')[0] AS sent_to, should_go_to, workflow AS ai_workflow,
# MAGIC        CASE WHEN split(mailbox, '@')[0] <> should_go_to THEN 'Wrong mailbox' ELSE 'No attachment — request is in the email body' END AS why_it_matters,
# MAGIC        from_address, subject
# MAGIC FROM expected
# MAGIC WHERE split(mailbox, '@')[0] <> should_go_to OR NOT has_attachment
# MAGIC ORDER BY received_ts DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4 Documents by type
# MAGIC The `classify_documents` task reads the **parsed PDF**; it should agree with the routing of its **email** almost everywhere — where it doesn't, trust the document.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Documents by type, and whether the document type agrees with how its email was routed
# MAGIC SELECT doc_type,
# MAGIC        CASE WHEN doc_type = email_workflow THEN 'Agrees with the email' ELSE 'Differs from the email' END AS routing_check,
# MAGIC        count(*) AS documents, sum(page_count) AS pages
# MAGIC FROM documents
# MAGIC GROUP BY ALL
# MAGIC ORDER BY documents DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Documents by type, and whether the document type agrees with how its email was routed
# MAGIC SELECT doc_type,
# MAGIC        CASE WHEN doc_type = email_workflow THEN 'Agrees with the email' ELSE 'Differs from the email' END AS routing_check,
# MAGIC        count(*) AS documents, sum(page_count) AS pages
# MAGIC FROM documents
# MAGIC GROUP BY ALL
# MAGIC ORDER BY documents DESC) SELECT `doc_type`,SUM(`documents`) `column_2750825948`,`routing_check` FROM q GROUP BY `doc_type`,`routing_check`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.5 💬 Ask Genie — the inbox
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("How many emails in each mailbox needed action, and which ones were sent to the wrong mailbox?", before=('Section 5 (routing)', 15))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.6 🛠 Try it yourself
# MAGIC Open the job run from **step 4.2**, click the **`classify_emails`** task and look at its output: rows **succeeded / failed** and the **Sample Data** tab. Then open the **`parse_attachments`** task's output table in Unity Catalog and compare one row's `response` with what you saw in step 2.2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Extracted line items, specs & measurements
# MAGIC
# MAGIC ### 6.1 What the `flatten_extractions` task does
# MAGIC The five `ai_extract` tasks write their results as nested VARIANT (`response`). This notebook task turns them into **one row per PO line, invoice line or measured characteristic** — the tables Genie, dashboards and the exception rules use — and builds the **human review queue** from the confidence scores. Plain SQL, no AI.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ▶ DAILY JOB · task: flatten_extractions
# MAGIC -- Plain SQL, no AI: turn the nested ai_extract task outputs into one row per line / measurement.
# MAGIC -- Each extracted field is {value, confidence_score}.
# MAGIC CREATE OR REPLACE TABLE customer_order_lines COMMENT 'Customer PO line items extracted by ai_extract' AS
# MAGIC SELECT x.doc_id, d.email_id, d.received_ts, d.sender, d.processed_by,
# MAGIC        r:customer_name:value::STRING                          AS customer_name,
# MAGIC        r:customer_po_number:value::STRING                     AS customer_po_number,
# MAGIC        try_cast(r:po_date:value::STRING AS DATE)              AS po_date,
# MAGIC        try_cast(r:requested_delivery_date:value::STRING AS DATE) AS requested_delivery_date,
# MAGIC        r:payment_terms:value::STRING                          AS payment_terms,
# MAGIC        r:special_instructions:value::STRING                   AS special_instructions,
# MAGIC        coalesce(l:line_number:value::INT, pos + 1)            AS line_number,
# MAGIC        coalesce(nullif(regexp_extract(upper(l:our_part_number:value::STRING), '([A-Z]{3}-[0-9]{4})', 1), ''), upper(trim(l:our_part_number:value::STRING))) AS part_number,  -- keep just the RPC-#### code
# MAGIC        l:customer_part_number:value::STRING                   AS customer_part_number,
# MAGIC        l:description:value::STRING                            AS description,
# MAGIC        l:quantity:value::DOUBLE                               AS quantity,
# MAGIC        l:unit_price:value::DOUBLE                             AS unit_price,
# MAGIC        round(l:quantity:value::DOUBLE * l:unit_price:value::DOUBLE, 2) AS line_value,
# MAGIC        least(r:customer_po_number:confidence_score::DOUBLE, l:our_part_number:confidence_score::DOUBLE,
# MAGIC              l:quantity:confidence_score::DOUBLE, l:unit_price:confidence_score::DOUBLE) AS min_confidence
# MAGIC FROM (SELECT doc_id, response:response AS r FROM extracted_purchase_orders WHERE error IS NULL) x
# MAGIC JOIN documents d USING (doc_id)
# MAGIC LATERAL VIEW OUTER posexplode(CAST(r:lines AS ARRAY<VARIANT>)) t AS pos, l
# MAGIC WHERE l IS NULL OR l:quantity:value::DOUBLE IS NOT NULL;   -- skip subtotal / total rows if any slip through
# MAGIC
# MAGIC CREATE OR REPLACE TABLE supplier_invoice_lines COMMENT 'Supplier invoice line items extracted by ai_extract' AS
# MAGIC SELECT x.doc_id, d.email_id, d.received_ts, d.sender, d.processed_by,
# MAGIC        r:supplier_name:value::STRING                          AS supplier_name,
# MAGIC        r:invoice_number:value::STRING                         AS invoice_number,
# MAGIC        try_cast(r:invoice_date:value::STRING AS DATE)         AS invoice_date,
# MAGIC        nullif(trim(r:po_number:value::STRING), '')            AS po_number,
# MAGIC        r:remit_to_account_last4:value::STRING                 AS remit_to_account_last4,
# MAGIC        coalesce(r:bank_change_notice:value::BOOLEAN, false)   AS bank_change_notice,
# MAGIC        r:freight:value::DOUBLE                                AS freight,
# MAGIC        r:invoice_total:value::DOUBLE                          AS invoice_total,
# MAGIC        pos + 1                                                AS line_number,
# MAGIC        l:item_code:value::STRING                              AS item_code,
# MAGIC        l:description:value::STRING                            AS description,
# MAGIC        l:quantity:value::DOUBLE                               AS quantity,
# MAGIC        l:unit_price:value::DOUBLE                             AS unit_price,
# MAGIC        l:amount:value::DOUBLE                                 AS amount,
# MAGIC        least(r:invoice_number:confidence_score::DOUBLE, r:invoice_total:confidence_score::DOUBLE,
# MAGIC              l:quantity:confidence_score::DOUBLE, l:unit_price:confidence_score::DOUBLE) AS min_confidence
# MAGIC FROM (SELECT doc_id, response:response AS r FROM extracted_invoices WHERE error IS NULL) x
# MAGIC JOIN documents d USING (doc_id)
# MAGIC LATERAL VIEW OUTER posexplode(CAST(r:lines AS ARRAY<VARIANT>)) t AS pos, l
# MAGIC WHERE l IS NULL OR l:quantity:value::DOUBLE IS NOT NULL;   -- skip subtotal / freight / total rows if any slip through
# MAGIC
# MAGIC CREATE OR REPLACE TABLE product_specs COMMENT 'Product data sheet attributes extracted by ai_extract' AS
# MAGIC SELECT x.doc_id, d.email_id, d.received_ts, d.sender, d.processed_by,
# MAGIC        r:supplier_name:value::STRING AS supplier_name, r:product_name:value::STRING AS product_name,
# MAGIC        r:supplier_part_number:value::STRING AS supplier_part_number, coalesce(nullif(regexp_extract(upper(r:for_part_number:value::STRING), '([A-Z]{3}-[0-9]{4})', 1), ''), upper(trim(r:for_part_number:value::STRING))) AS part_number,
# MAGIC        r:revision:value::STRING AS revision, r:material:value::STRING AS material, r:hardness:value::STRING AS hardness,
# MAGIC        r:max_operating_temp_c:value::DOUBLE AS max_operating_temp_c,
# MAGIC        r:rohs_compliant:value::BOOLEAN AS rohs_compliant, r:reach_compliant:value::BOOLEAN AS reach_compliant,
# MAGIC        r:change_note:value::STRING AS change_note,
# MAGIC        least(r:for_part_number:confidence_score::DOUBLE, r:rohs_compliant:confidence_score::DOUBLE,
# MAGIC              r:max_operating_temp_c:confidence_score::DOUBLE) AS min_confidence
# MAGIC FROM (SELECT doc_id, response:response AS r FROM extracted_data_sheets WHERE error IS NULL) x
# MAGIC JOIN documents d USING (doc_id);
# MAGIC
# MAGIC CREATE OR REPLACE TABLE inspection_results COMMENT 'Measured characteristics from certificates of conformance extracted by ai_extract' AS
# MAGIC SELECT x.doc_id, d.email_id, d.received_ts, d.sender, d.processed_by,
# MAGIC        r:supplier_name:value::STRING AS supplier_name, r:po_number:value::STRING AS po_number,
# MAGIC        coalesce(nullif(regexp_extract(upper(r:part_number:value::STRING), '([A-Z]{3}-[0-9]{4})', 1), ''), upper(trim(r:part_number:value::STRING))) AS part_number,  -- e.g. 'CMP-2001 (6205-2RS)' -> 'CMP-2001'
# MAGIC        nullif(nullif(trim(r:lot_number:value::STRING), ''), '-') AS lot_number, r:heat_number:value::STRING AS heat_number,
# MAGIC        r:quantity_shipped:value::DOUBLE AS quantity_shipped, try_cast(r:inspection_date:value::STRING AS DATE) AS inspection_date,
# MAGIC        r:supplier_notes:value::STRING AS supplier_notes,
# MAGIC        m:characteristic:value::STRING AS characteristic, m:nominal_mm:value::DOUBLE AS nominal_mm,
# MAGIC        m:tolerance_mm:value::DOUBLE AS tolerance_mm, m:measured_min_mm:value::DOUBLE AS measured_min_mm,
# MAGIC        m:measured_max_mm:value::DOUBLE AS measured_max_mm, m:supplier_result:value::STRING AS supplier_result,
# MAGIC        least(m:measured_min_mm:confidence_score::DOUBLE, m:measured_max_mm:confidence_score::DOUBLE,
# MAGIC              r:part_number:confidence_score::DOUBLE) AS min_confidence
# MAGIC FROM (SELECT doc_id, response:response AS r FROM extracted_quality_docs WHERE error IS NULL) x
# MAGIC JOIN documents d USING (doc_id)
# MAGIC LATERAL VIEW OUTER explode(CAST(r:measurements AS ARRAY<VARIANT>)) t AS m;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE engineering_changes COMMENT 'Engineering change notices extracted by ai_extract' AS
# MAGIC SELECT x.doc_id, d.received_ts, d.processed_by,
# MAGIC        r:ecn_number:value::STRING AS ecn_number, try_cast(r:effective_date:value::STRING AS DATE) AS effective_date,
# MAGIC        r:change_class:value::STRING AS change_class,
# MAGIC        coalesce(nullif(regexp_extract(upper(r:old_part_number:value::STRING), '([A-Z]{3}-[0-9]{4})', 1), ''), upper(trim(r:old_part_number:value::STRING))) AS old_part_number, r:old_revision:value::STRING AS old_revision,
# MAGIC        coalesce(nullif(regexp_extract(upper(r:new_part_number:value::STRING), '([A-Z]{3}-[0-9]{4})', 1), ''), upper(trim(r:new_part_number:value::STRING))) AS new_part_number, r:new_revision:value::STRING AS new_revision,
# MAGIC        r:affected_customers:value::STRING AS affected_customers, r:reason:value::STRING AS reason,
# MAGIC        r:open_order_action:value::STRING AS open_order_action
# MAGIC FROM (SELECT doc_id, response:response AS r FROM extracted_ecns WHERE error IS NULL) x
# MAGIC JOIN documents d USING (doc_id);
# MAGIC
# MAGIC -- Human review queue: any extracted line / measurement whose key fields fall below the confidence threshold (widget)
# MAGIC CREATE OR REPLACE TABLE extraction_review_queue COMMENT 'Extracted values below the confidence threshold — a person should verify them' AS
# MAGIC SELECT 'customer_purchase_order' AS doc_type, doc_id, received_ts, customer_po_number AS reference,
# MAGIC        concat('Line ', line_number, ': ', coalesce(part_number, '?'), ' x ', coalesce(CAST(quantity AS STRING), '?')) AS item, min_confidence
# MAGIC FROM customer_order_lines WHERE min_confidence < confidence_threshold
# MAGIC UNION ALL
# MAGIC SELECT 'supplier_invoice', doc_id, received_ts, invoice_number, concat('Line ', line_number, ': ', coalesce(item_code, '?')), min_confidence
# MAGIC FROM supplier_invoice_lines WHERE min_confidence < confidence_threshold
# MAGIC UNION ALL
# MAGIC SELECT 'product_data_sheet', doc_id, received_ts, supplier_part_number, coalesce(product_name, '?'), min_confidence
# MAGIC FROM product_specs WHERE min_confidence < confidence_threshold
# MAGIC UNION ALL
# MAGIC SELECT 'quality_document', doc_id, received_ts, po_number, coalesce(characteristic, '?'), min_confidence
# MAGIC FROM inspection_results WHERE min_confidence < confidence_threshold;
# MAGIC
# MAGIC -- CREATE OR REPLACE resets column comments; re-apply the ones Genie relies on
# MAGIC ALTER TABLE customer_order_lines ALTER COLUMN min_confidence COMMENT 'Lowest ai_extract confidence score among the key fields of this line';
# MAGIC ALTER TABLE customer_order_lines ALTER COLUMN line_value COMMENT 'quantity x unit_price in USD';
# MAGIC
# MAGIC SELECT 'customer_order_lines' AS table_name, count(*) AS row_count FROM customer_order_lines
# MAGIC UNION ALL SELECT 'supplier_invoice_lines', count(*) FROM supplier_invoice_lines
# MAGIC UNION ALL SELECT 'product_specs', count(*) FROM product_specs
# MAGIC UNION ALL SELECT 'inspection_results', count(*) FROM inspection_results
# MAGIC UNION ALL SELECT 'engineering_changes', count(*) FROM engineering_changes
# MAGIC UNION ALL SELECT 'extraction_review_queue', count(*) FROM extraction_review_queue;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.2 Customer order value by customer
# MAGIC Every dollar here was typed by nobody — it came straight out of PDFs in three different PO layouts.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Customer order value captured from PDFs, by customer (the whole 45-day backfill)
# MAGIC SELECT coalesce(c.customer_name, l.customer_name) AS customer,
# MAGIC        count(DISTINCT l.customer_po_number) AS purchase_orders,
# MAGIC        count(*) AS order_lines,
# MAGIC        round(sum(l.line_value), 2) AS order_value_usd
# MAGIC FROM customer_order_lines l
# MAGIC LEFT JOIN customers c ON lower(substring_index(l.sender, '@', -1)) = c.email_domain
# MAGIC GROUP BY ALL
# MAGIC ORDER BY order_value_usd DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Customer order value captured from PDFs, by customer (the whole 45-day backfill)
# MAGIC SELECT coalesce(c.customer_name, l.customer_name) AS customer,
# MAGIC        count(DISTINCT l.customer_po_number) AS purchase_orders,
# MAGIC        count(*) AS order_lines,
# MAGIC        round(sum(l.line_value), 2) AS order_value_usd
# MAGIC FROM customer_order_lines l
# MAGIC LEFT JOIN customers c ON lower(substring_index(l.sender, '@', -1)) = c.email_domain
# MAGIC GROUP BY ALL
# MAGIC ORDER BY order_value_usd DESC) SELECT `customer`,SUM(`order_value_usd`) `column_7159299840` FROM q GROUP BY `customer`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.3 How confident is the extraction?
# MAGIC Most extracted rows score at or near 1.0. Anything below the `confidence_threshold` widget lands in the review queue (step 6.4).

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Extraction confidence by document type: most values come back at or near 1.0; the rest go to review
# MAGIC SELECT doc_type,
# MAGIC        CASE WHEN min_confidence IS NULL THEN 'no score' WHEN min_confidence >= 0.95 THEN '0.95 – 1.00' WHEN min_confidence >= 0.80 THEN '0.80 – 0.95' ELSE 'below 0.80' END AS confidence_band,
# MAGIC        count(*) AS extracted_rows
# MAGIC FROM (
# MAGIC   SELECT 'customer_purchase_order' AS doc_type, min_confidence FROM customer_order_lines
# MAGIC   UNION ALL SELECT 'supplier_invoice', min_confidence FROM supplier_invoice_lines
# MAGIC   UNION ALL SELECT 'product_data_sheet', min_confidence FROM product_specs
# MAGIC   UNION ALL SELECT 'quality_document', min_confidence FROM inspection_results
# MAGIC )
# MAGIC GROUP BY ALL
# MAGIC ORDER BY doc_type, confidence_band;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Extraction confidence by document type: most values come back at or near 1.0; the rest go to review
# MAGIC SELECT doc_type,
# MAGIC        CASE WHEN min_confidence IS NULL THEN 'no score' WHEN min_confidence >= 0.95 THEN '0.95 – 1.00' WHEN min_confidence >= 0.80 THEN '0.80 – 0.95' ELSE 'below 0.80' END AS confidence_band,
# MAGIC        count(*) AS extracted_rows
# MAGIC FROM (
# MAGIC   SELECT 'customer_purchase_order' AS doc_type, min_confidence FROM customer_order_lines
# MAGIC   UNION ALL SELECT 'supplier_invoice', min_confidence FROM supplier_invoice_lines
# MAGIC   UNION ALL SELECT 'product_data_sheet', min_confidence FROM product_specs
# MAGIC   UNION ALL SELECT 'quality_document', min_confidence FROM inspection_results
# MAGIC )
# MAGIC GROUP BY ALL
# MAGIC ORDER BY doc_type, confidence_band) SELECT `doc_type`,SUM(`extracted_rows`) `column_6564123983`,`confidence_band` FROM q GROUP BY `doc_type`,`confidence_band`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.4 Human review queue
# MAGIC Only these values need a person to look — usually ambiguous wording or an unusual layout on the document. Scores can vary slightly from run to run, so the exact rows here may differ; the rest are accepted automatically.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Values below the confidence threshold: a person checks these instead of re-keying every document
# MAGIC SELECT q.doc_type, q.received_ts, q.reference, q.item, round(q.min_confidence, 2) AS min_confidence, d.file_name
# MAGIC FROM extraction_review_queue q JOIN documents d USING (doc_id)
# MAGIC ORDER BY q.min_confidence;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.5 💬 Ask Genie — extracted orders
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("What is the total value of customer orders we received by customer, and how many POs each?", before=('Section 6 (extraction)', 109))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.6 🛠 Try it yourself
# MAGIC Add `payment_terms` to the `invoices` schema in **step 3.1**, re-run **step 4.1** and give the `extract_invoices` task's output a new name (e.g. `"table": "extracted_invoices_v2"`) so every invoice is re-extracted; then add `r:payment_terms:value::STRING AS payment_terms` to `supplier_invoice_lines` in **step 6.1** (reading the new table) and re-run the job. That's the full change cycle: schema → task → table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Flag exceptions — AI extracts, SQL validates
# MAGIC
# MAGIC The AI Functions turn documents into data; **plain, auditable SQL rules** then compare that data to the ERP. Every rule is one block in step 7.1 — the `validate_exceptions` task — easy to read, change and explain to an auditor:
# MAGIC
# MAGIC | Area | Rule | Compares against |
# MAGIC |---|---|---|
# MAGIC | **Orders** | Price differs from contract price by more than `price_tolerance_pct` · part unknown or **obsolete** (and the ECN that replaced it) · delivery date inside lead time · same PO received twice | `item_master`, `customers`, `engineering_changes` |
# MAGIC | **Payables** | No / unknown PO · price above PO price · billed more than **received** (3-way match) · same invoice twice · **bank details changed or sent from a look-alike domain** | `purchase_orders`, `goods_receipts`, `suppliers` |
# MAGIC | **Engineering** | Data sheet not RoHS compliant where we require it · temperature rating below our spec | `item_master` |
# MAGIC | **Quality** | Measured **out of our tolerance** (regardless of what the supplier reported) · lot number missing | `component_specs` |
# MAGIC | **Data quality** | A date that can't be right (delivery requested before the PO arrived) · an extracted value below the confidence threshold — it goes to a person for review **instead of** raising a business exception, so a misread line never triggers a false price or tolerance alarm | `extraction_review_queue` |

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ▶ DAILY JOB · task: validate_exceptions
# MAGIC CREATE OR REPLACE TABLE document_exceptions
# MAGIC COMMENT 'One row per exception found in an extracted document; amount_at_risk_usd is the estimated dollar impact' AS
# MAGIC WITH po AS (
# MAGIC   SELECT l.*, c.customer_id, c.customer_name AS master_customer, i.status AS item_status, i.lead_time_days,
# MAGIC          CAST(round(i.list_price * (1 - c.discount_pct / 100), 2) AS DOUBLE) AS contract_price
# MAGIC   FROM customer_order_lines l
# MAGIC   LEFT JOIN customers c ON lower(substring_index(l.sender, '@', -1)) = c.email_domain
# MAGIC   LEFT JOIN item_master i ON i.part_number = l.part_number
# MAGIC ),
# MAGIC po_docs AS (
# MAGIC   SELECT doc_id, customer_id, master_customer, customer_po_number, received_ts, sum(line_value) AS order_value,
# MAGIC          dense_rank() OVER (PARTITION BY customer_id, customer_po_number ORDER BY received_ts) AS copy_no
# MAGIC   FROM po GROUP BY doc_id, customer_id, master_customer, customer_po_number, received_ts
# MAGIC ),
# MAGIC inv AS (
# MAGIC   SELECT l.*, coalesce(h.supplier_id, sd.supplier_id, sn.supplier_id) AS supplier_id, h.po_number IS NOT NULL AS po_found,
# MAGIC          o.unit_price AS po_price, g.qty_received
# MAGIC   FROM supplier_invoice_lines l
# MAGIC   LEFT JOIN (SELECT DISTINCT po_number, supplier_id FROM purchase_orders) h ON h.po_number = l.po_number
# MAGIC   LEFT JOIN suppliers sd ON sd.email_domain = lower(substring_index(l.sender, '@', -1))
# MAGIC   LEFT JOIN suppliers sn ON lower(sn.supplier_name) = lower(l.supplier_name)
# MAGIC   LEFT JOIN purchase_orders o ON o.po_number = l.po_number AND o.supplier_item_code = l.item_code
# MAGIC   LEFT JOIN goods_receipts g ON g.po_number = o.po_number AND g.line_number = o.line_number
# MAGIC ),
# MAGIC inv_docs AS (
# MAGIC   SELECT i.doc_id, i.supplier_id, s.supplier_name, s.email_domain, s.bank_account_last4, i.invoice_number, i.po_number, i.po_found,
# MAGIC          i.remit_to_account_last4, i.bank_change_notice, i.invoice_total, i.received_ts,
# MAGIC          lower(substring_index(i.sender, '@', -1)) AS sender_domain,
# MAGIC          dense_rank() OVER (PARTITION BY i.supplier_id, i.invoice_number ORDER BY i.received_ts) AS copy_no
# MAGIC   FROM (SELECT DISTINCT doc_id, supplier_id, invoice_number, po_number, po_found, remit_to_account_last4, bank_change_notice,
# MAGIC                         invoice_total, received_ts, sender FROM inv) i
# MAGIC   LEFT JOIN suppliers s USING (supplier_id)
# MAGIC ),
# MAGIC found AS (
# MAGIC   -- Orders ---------------------------------------------------------------------------------------------
# MAGIC   SELECT doc_id, 'customer_purchase_order' AS doc_type, 'Orders' AS category, master_customer AS counterparty, customer_po_number AS reference,
# MAGIC          'PRICE_MISMATCH' AS exception_type,
# MAGIC          format_string('Line %d %s: PO price $%.2f vs contract price $%.2f (%+.1f%%)', line_number, part_number, unit_price, contract_price,
# MAGIC                        (unit_price - contract_price) / contract_price * 100) AS detail,
# MAGIC          round(abs(contract_price - unit_price) * quantity, 2) AS amount_at_risk_usd
# MAGIC   FROM po WHERE coalesce(min_confidence, 1) >= confidence_threshold   -- untrusted values go to the review queue instead
# MAGIC     AND contract_price > 0 AND abs(unit_price - contract_price) / contract_price * 100 > price_tolerance_pct
# MAGIC   UNION ALL
# MAGIC   SELECT po.doc_id, 'customer_purchase_order', 'Orders', master_customer, customer_po_number,
# MAGIC          CASE WHEN item_status IS NULL THEN 'UNKNOWN_PART' ELSE 'OBSOLETE_PART' END,
# MAGIC          concat('Line ', line_number, ' orders ', part_number, CASE WHEN item_status IS NULL THEN ', which is not in the item master'
# MAGIC                 ELSE ' (OBSOLETE)' END,
# MAGIC                 coalesce(concat(' - superseded by ', e.new_part_number, ' per ', e.ecn_number, ' effective ', e.effective_date), '')),
# MAGIC          line_value
# MAGIC   FROM po LEFT JOIN (SELECT old_part_number, max_by(new_part_number, effective_date) AS new_part_number,
# MAGIC                            max_by(ecn_number, effective_date) AS ecn_number, max(effective_date) AS effective_date
# MAGIC                     FROM engineering_changes WHERE new_part_number <> old_part_number GROUP BY old_part_number) e
# MAGIC          ON e.old_part_number = po.part_number
# MAGIC   WHERE coalesce(po.min_confidence, 1) >= confidence_threshold AND po.part_number IS NOT NULL AND (item_status IS NULL OR item_status = 'OBSOLETE')
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, 'customer_purchase_order', 'Orders', master_customer, customer_po_number, 'LEAD_TIME_RISK',
# MAGIC          format_string('Line %d %s: delivery requested %s, %d days after the PO arrived; standard lead time is %d days',
# MAGIC                        line_number, part_number, CAST(requested_delivery_date AS STRING), datediff(requested_delivery_date, to_date(received_ts)), lead_time_days),
# MAGIC          line_value
# MAGIC   FROM po WHERE coalesce(min_confidence, 1) >= confidence_threshold
# MAGIC     AND requested_delivery_date >= to_date(received_ts)              -- an earlier date is a misread, handled below
# MAGIC     AND requested_delivery_date < date_add(to_date(received_ts), lead_time_days)
# MAGIC   UNION ALL
# MAGIC   SELECT DISTINCT doc_id, 'customer_purchase_order', 'Data quality', master_customer, customer_po_number, 'DATE_NOT_PLAUSIBLE',
# MAGIC          concat('Requested delivery date ', CAST(requested_delivery_date AS STRING), ' is before the PO arrived on ',
# MAGIC                 CAST(to_date(received_ts) AS STRING), ' - check the date format on the document'), NULL
# MAGIC   FROM po WHERE requested_delivery_date < to_date(received_ts)
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, 'customer_purchase_order', 'Orders', master_customer, customer_po_number, 'DUPLICATE_PO',
# MAGIC          concat('PO ', customer_po_number, ' was already received on ',
# MAGIC                 (SELECT min(to_date(b.received_ts)) FROM po_docs b WHERE b.customer_id = a.customer_id AND b.customer_po_number = a.customer_po_number)),
# MAGIC          order_value
# MAGIC   FROM po_docs a WHERE copy_no > 1
# MAGIC   -- Payables -------------------------------------------------------------------------------------------
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, 'supplier_invoice', 'Payables', supplier_name, invoice_number, 'NO_PO_REFERENCE',
# MAGIC          CASE WHEN po_number IS NULL THEN 'Invoice does not reference a purchase order' ELSE concat('PO ', po_number, ' not found in the ERP') END,
# MAGIC          invoice_total
# MAGIC   FROM inv_docs WHERE NOT po_found
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, 'supplier_invoice', 'Payables', s.supplier_name, invoice_number, 'PRICE_VARIANCE',
# MAGIC          format_string('%s: invoiced $%.2f vs PO price $%.2f (%+.1f%%) on %,.0f units', item_code, unit_price, po_price,
# MAGIC                        (unit_price - po_price) / po_price * 100, quantity),
# MAGIC          round((unit_price - po_price) * quantity, 2)
# MAGIC   FROM inv LEFT JOIN suppliers s USING (supplier_id)
# MAGIC   WHERE coalesce(min_confidence, 1) >= confidence_threshold AND po_price > 0 AND (unit_price - po_price) / po_price * 100 > price_tolerance_pct
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, 'supplier_invoice', 'Payables', s.supplier_name, invoice_number, 'BILLED_MORE_THAN_RECEIVED',
# MAGIC          format_string('%s: billed %,.0f units, received %,d against %s', item_code, quantity, qty_received, po_number),
# MAGIC          round((quantity - qty_received) * unit_price, 2)
# MAGIC   FROM inv LEFT JOIN suppliers s USING (supplier_id)
# MAGIC   WHERE coalesce(min_confidence, 1) >= confidence_threshold AND qty_received IS NOT NULL AND quantity > qty_received
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, 'supplier_invoice', 'Payables', supplier_name, invoice_number, 'DUPLICATE_INVOICE',
# MAGIC          concat('Invoice ', invoice_number, ' was already received on ',
# MAGIC                 (SELECT min(to_date(b.received_ts)) FROM inv_docs b WHERE b.supplier_id = a.supplier_id AND b.invoice_number = a.invoice_number)),
# MAGIC          invoice_total
# MAGIC   FROM inv_docs a WHERE copy_no > 1
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, 'supplier_invoice', 'Payables', supplier_name, invoice_number, 'BANK_DETAILS_CHANGED',
# MAGIC          concat_ws('; ',
# MAGIC            CASE WHEN remit_to_account_last4 <> bank_account_last4
# MAGIC                 THEN concat('remit-to account ending ', remit_to_account_last4, ' does not match account on file ending ', bank_account_last4) END,
# MAGIC            CASE WHEN bank_change_notice THEN 'invoice announces new bank details' END,
# MAGIC            CASE WHEN sender_domain <> email_domain THEN concat('sent from ', sender_domain, ', not ', email_domain) END),
# MAGIC          invoice_total
# MAGIC   FROM inv_docs
# MAGIC   WHERE remit_to_account_last4 <> bank_account_last4 OR bank_change_notice OR sender_domain <> email_domain
# MAGIC   -- Engineering ----------------------------------------------------------------------------------------
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, 'product_data_sheet', 'Engineering', coalesce(sp.supplier_name, p.supplier_name), concat(p.supplier_part_number, ' rev ', p.revision), 'SPEC_NONCOMPLIANT',
# MAGIC          concat_ws('; ',
# MAGIC            CASE WHEN i.rohs_required AND NOT p.rohs_compliant THEN concat('not RoHS compliant (', coalesce(p.material, ''), ') - RoHS is required for ', i.part_number) END,
# MAGIC            CASE WHEN p.max_operating_temp_c < i.min_temp_rating_c
# MAGIC                 THEN format_string('rated to %.0f deg C; our spec for %s requires %d deg C', p.max_operating_temp_c, i.part_number, i.min_temp_rating_c) END),
# MAGIC          NULL
# MAGIC   FROM product_specs p JOIN item_master i ON i.part_number = p.part_number LEFT JOIN suppliers sp ON sp.supplier_id = i.supplier_id
# MAGIC   WHERE (i.rohs_required AND NOT p.rohs_compliant) OR p.max_operating_temp_c < i.min_temp_rating_c
# MAGIC   -- Quality --------------------------------------------------------------------------------------------
# MAGIC   UNION ALL
# MAGIC   SELECT r.doc_id, 'quality_document', 'Quality', coalesce(sq.supplier_name, r.supplier_name), concat(r.po_number, ' / ', r.part_number), 'OUT_OF_TOLERANCE',
# MAGIC          format_string('%s measured %.3f - %.3f vs spec %.3f +/- %.3f mm; supplier reported %s', r.characteristic, r.measured_min_mm,
# MAGIC                        r.measured_max_mm, CAST(s.nominal_mm AS DOUBLE), CAST(s.tol_mm AS DOUBLE), coalesce(r.supplier_result, 'no result')),
# MAGIC          round(r.quantity_shipped * i.unit_cost, 2)
# MAGIC   FROM inspection_results r
# MAGIC   JOIN component_specs s ON s.part_number = r.part_number AND lower(s.characteristic) = lower(r.characteristic)
# MAGIC   LEFT JOIN item_master i ON i.part_number = r.part_number
# MAGIC   LEFT JOIN suppliers sq ON sq.supplier_id = i.supplier_id
# MAGIC   WHERE coalesce(r.min_confidence, 1) >= confidence_threshold
# MAGIC     AND (r.measured_max_mm > s.nominal_mm + s.tol_mm + 1e-9 OR r.measured_min_mm < s.nominal_mm - s.tol_mm - 1e-9)
# MAGIC   UNION ALL
# MAGIC   SELECT DISTINCT r.doc_id, 'quality_document', 'Quality', coalesce(sq.supplier_name, r.supplier_name), concat(r.po_number, ' / ', r.part_number),
# MAGIC          'MISSING_LOT_TRACE', 'Certificate has no lot number - shipment cannot be traced', NULL
# MAGIC   FROM inspection_results r LEFT JOIN item_master i ON i.part_number = r.part_number LEFT JOIN suppliers sq ON sq.supplier_id = i.supplier_id
# MAGIC   WHERE r.lot_number IS NULL
# MAGIC   -- Data quality ---------------------------------------------------------------------------------------
# MAGIC   UNION ALL
# MAGIC   SELECT doc_id, doc_type, 'Data quality', NULL, reference, 'LOW_CONFIDENCE_EXTRACTION',
# MAGIC          format_string('%s extracted with confidence %.2f - verify against the document', item, min_confidence), NULL
# MAGIC   FROM extraction_review_queue
# MAGIC )
# MAGIC SELECT md5(concat_ws('|', f.doc_id, f.exception_type, f.detail)) AS exception_id,
# MAGIC        f.doc_id, f.doc_type, f.category, coalesce(f.counterparty, c.customer_name, s.supplier_name) AS counterparty, f.reference,
# MAGIC        f.exception_type, f.detail, f.amount_at_risk_usd,
# MAGIC        d.email_id, d.received_ts, d.file_name, d.processed_by, current_timestamp() AS detected_at
# MAGIC FROM found f JOIN documents d USING (doc_id)
# MAGIC LEFT JOIN customers c ON c.email_domain = lower(substring_index(d.sender, '@', -1))
# MAGIC LEFT JOIN suppliers s ON s.email_domain = lower(substring_index(d.sender, '@', -1));
# MAGIC
# MAGIC -- CREATE OR REPLACE resets column comments; re-apply the one Genie relies on
# MAGIC ALTER TABLE document_exceptions ALTER COLUMN amount_at_risk_usd COMMENT 'Estimated dollar impact of the exception (price difference x qty, overbilled qty x price, lot or invoice value)';
# MAGIC SELECT count(*) AS exceptions_found FROM document_exceptions;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7.2 Exceptions by type
# MAGIC Each planted problem from section 0 should appear here — plus the low-confidence values from step 6.4.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Exceptions by type and area — what the rules found across the whole backfill
# MAGIC SELECT exception_type, category, count(*) AS exceptions, round(sum(coalesce(amount_at_risk_usd, 0)), 2) AS amount_at_risk_usd
# MAGIC FROM document_exceptions
# MAGIC GROUP BY ALL
# MAGIC ORDER BY exceptions DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Exceptions by type and area — what the rules found across the whole backfill
# MAGIC SELECT exception_type, category, count(*) AS exceptions, round(sum(coalesce(amount_at_risk_usd, 0)), 2) AS amount_at_risk_usd
# MAGIC FROM document_exceptions
# MAGIC GROUP BY ALL
# MAGIC ORDER BY exceptions DESC) SELECT `exception_type`,SUM(`exceptions`) `column_1456777582`,`category` FROM q GROUP BY `exception_type`,`category`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7.3 The exception worklist
# MAGIC Sorted by dollar impact. Find the `OBSOLETE_PART` row — the explanation (an ECN) arrived weeks earlier in a different mailbox — and the `OUT_OF_TOLERANCE` row the supplier marked **PASS**.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Every exception, biggest dollar impact first. Look for the OBSOLETE_PART row (explained by an ECN that arrived
# MAGIC -- weeks earlier in a different mailbox) and the OUT_OF_TOLERANCE row the supplier marked PASS.
# MAGIC SELECT category, exception_type, counterparty, reference, detail, amount_at_risk_usd, to_date(received_ts) AS received, file_name
# MAGIC FROM document_exceptions
# MAGIC ORDER BY amount_at_risk_usd DESC NULLS LAST;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7.4 💬 Ask Genie — exceptions
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Where were we billed more than we received or more than our PO price, and by how much?", before=('Section 7 (3-way match rules)', 145))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7.5 🛠 Try it yourself
# MAGIC Set the **`price_tolerance_pct`** widget to `5`, re-run **S.2 and S.3** so the new value takes effect, then re-run **steps 7.1–7.3**. Which price exceptions disappear? To make it the job's default, re-run **step 4.1**. The rule itself is one line in step 7.1.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Decide what to do — `ai_decide` — and draft the reply — `ai_gen`
# MAGIC
# MAGIC An exception list is only useful if someone acts on it. **`ai_decide`** *(Beta)* evaluates each exception against questions **you** define, in three styles — all in one call (step 8.1, the `decide_exceptions` task):
# MAGIC
# MAGIC | Question | Type | Returns |
# MAGIC |---|---|---|
# MAGIC | `route` — who should handle this? | **choice** (auto-resolve, buyer, customer service, quality engineer, AP fraud review, data review) | the chosen label + a probability for every label + confidence |
# MAGIC | `hold` — should we hold payment / shipment until it's resolved? | **noul** — `ai_decide`'s name for a yes / no likelihood question | a probability from 0 to 1 |
# MAGIC | `impact` — how serious is it? | **score** on your own 4-level rubric | a probability-weighted score from 0 to 3 |
# MAGIC
# MAGIC Then **`ai_gen`** (step 8.3, the `draft_replies` task) drafts the email for every exception that needs a person — to the supplier, the customer, or (for suspected payment fraud) to **internal AP security, never the sender**. People review and send; nobody starts from a blank page. *These run as notebook tasks because `ai_decide` and `ai_gen` aren't AI Function task types.*
# MAGIC
# MAGIC > ⚠️ `ai_decide` is in **Beta** and `ai_gen` in **Public Preview** — if these cells fail with *function not found*, ask a workspace admin to enable them on the **Previews** page.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ▶ DAILY JOB · task: decide_exceptions
# MAGIC CREATE TABLE IF NOT EXISTS exception_decisions (
# MAGIC   exception_id STRING, decision VARIANT, route STRING, route_confidence DOUBLE, hold_probability DOUBLE,
# MAGIC   business_impact DOUBLE, processed_by STRING, decided_at TIMESTAMP)
# MAGIC COMMENT 'ai_decide output per exception: route, hold probability and business impact';
# MAGIC
# MAGIC INSERT INTO exception_decisions
# MAGIC SELECT exception_id, decision,
# MAGIC        decision:response:answers:route:choice::STRING       AS route,
# MAGIC        decision:response:answers:route:confidence::DOUBLE   AS route_confidence,
# MAGIC        decision:response:answers:hold:probability::DOUBLE   AS hold_probability,
# MAGIC        decision:response:answers:impact:score::DOUBLE       AS business_impact,
# MAGIC        run_label, current_timestamp()
# MAGIC FROM (
# MAGIC   SELECT exception_id,
# MAGIC          ai_decide(
# MAGIC            to_json(named_struct('document_type', doc_type, 'area', category, 'counterparty', counterparty, 'reference', reference,
# MAGIC                                 'exception_type', exception_type, 'detail', detail, 'amount_at_risk_usd', amount_at_risk_usd)),
# MAGIC            '{"route": {"type": "choice", "instructions": "Who at a manufacturer should handle this document exception?",
# MAGIC                "criteria": {"auto_resolve": "Trivial, clearly explainable difference that can be accepted or corrected without a person",
# MAGIC                             "buyer": "Purchasing must contact a SUPPLIER: price, quantity, PO, receipt or duplicate problem on a supplier_invoice document",
# MAGIC                             "customer_service": "Customer service must contact a CUSTOMER: price, part number, duplicate order or delivery date problem on a customer_purchase_order document",
# MAGIC                             "quality_engineer": "Supplier quality / engineering must review: dimensions out of tolerance, traceability, material or compliance",
# MAGIC                             "ap_fraud_review": "Accounts payable security: possible payment fraud such as changed bank details or a look-alike sender domain",
# MAGIC                             "data_review": "The extracted data itself is uncertain; a clerk should verify the document"}},
# MAGIC              "hold": {"type": "noul", "instructions": "Should we hold payment (invoices), the shipment or production (orders and quality issues) until this is resolved?",
# MAGIC                "criteria": {"true": "Paying, shipping or building now risks losing money, shipping the wrong or non-conforming part, or paying a fraudster",
# MAGIC                             "false": "Work can continue while the issue is resolved"}},
# MAGIC              "impact": {"type": "score", "instructions": "How serious is this exception for the business?",
# MAGIC                "criteria": ["Negligible: informational or a few dollars", "Minor: small dollar amount or an easy correction",
# MAGIC                             "Significant: material dollars, schedule or customer impact", "Critical: fraud, safety, compliance or line-down risk"]}}',
# MAGIC            map('version', '1.0')) AS decision
# MAGIC   FROM document_exceptions x
# MAGIC   WHERE NOT EXISTS (SELECT 1 FROM exception_decisions d WHERE d.exception_id = x.exception_id)
# MAGIC );
# MAGIC
# MAGIC -- Sanity check: every exception should have a route (a NULL means ai_decide returned an error_message for that row)
# MAGIC SELECT count(*) AS decisions, count_if(route IS NULL) AS without_route,
# MAGIC        first(decision:error_message::STRING, true) AS example_error
# MAGIC FROM exception_decisions;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8.2 Who handles what — and how serious is it?
# MAGIC The look-alike-domain invoice should go to **AP fraud review** as critical; dimensional issues to the **quality engineer**; invoice problems to the **buyer**; order problems to **customer service**.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Decisions: exceptions by ai_decide route and impact band
# MAGIC SELECT d.route,
# MAGIC        CASE WHEN d.business_impact >= 2.5 THEN '3 Critical' WHEN d.business_impact >= 1.5 THEN '2 Significant'
# MAGIC             WHEN d.business_impact >= 0.5 THEN '1 Minor' ELSE '0 Negligible' END AS impact,
# MAGIC        count(*) AS exceptions,
# MAGIC        count_if(d.hold_probability >= 0.5) AS recommended_holds
# MAGIC FROM document_exceptions x JOIN exception_decisions d USING (exception_id)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY d.route, impact;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Decisions: exceptions by ai_decide route and impact band
# MAGIC SELECT d.route,
# MAGIC        CASE WHEN d.business_impact >= 2.5 THEN '3 Critical' WHEN d.business_impact >= 1.5 THEN '2 Significant'
# MAGIC             WHEN d.business_impact >= 0.5 THEN '1 Minor' ELSE '0 Negligible' END AS impact,
# MAGIC        count(*) AS exceptions,
# MAGIC        count_if(d.hold_probability >= 0.5) AS recommended_holds
# MAGIC FROM document_exceptions x JOIN exception_decisions d USING (exception_id)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY d.route, impact) SELECT `route`,SUM(`exceptions`) `column_1189037977`,`impact` FROM q GROUP BY `route`,`impact`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8.3 Draft the replies — `ai_gen`
# MAGIC One short email per exception that needs a person — addressed to the supplier or customer, or to internal AP security for suspected fraud. Only exceptions it hasn't drafted before are sent to `ai_gen`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ▶ DAILY JOB · task: draft_replies
# MAGIC CREATE TABLE IF NOT EXISTS draft_replies (exception_id STRING, recipient STRING, draft STRING, processed_by STRING, drafted_at TIMESTAMP)
# MAGIC COMMENT 'ai_gen output: drafted email per exception that needs a person';
# MAGIC
# MAGIC INSERT INTO draft_replies
# MAGIC SELECT x.exception_id,
# MAGIC        CASE d.route WHEN 'customer_service' THEN 'Customer' WHEN 'ap_fraud_review' THEN 'Internal: AP security'
# MAGIC                     WHEN 'data_review' THEN 'Internal: document review' ELSE 'Supplier' END AS recipient,
# MAGIC        ai_gen(concat(
# MAGIC          'You work at Ridgeline Precision Components, a manufacturer. Write a short, professional email (max 110 words, no subject line, no placeholders in brackets) ',
# MAGIC          CASE d.route
# MAGIC            WHEN 'customer_service' THEN concat('to our customer ', coalesce(x.counterparty, ''), ' ')
# MAGIC            WHEN 'ap_fraud_review' THEN 'to our internal AP security team - NOT to the sender. Explain why this looks like payment fraud and recommend verifying by phone using the supplier contact already on file before any payment or bank change '
# MAGIC            WHEN 'data_review' THEN 'to an internal document-review clerk asking them to verify the extracted value against the PDF '
# MAGIC            ELSE concat('to our supplier ', coalesce(x.counterparty, ''), ' ') END,
# MAGIC          'about ', replace(x.doc_type, '_', ' '), ' ', coalesce(x.reference, ''), '. Issue: ', x.detail,
# MAGIC          '. State the issue with the numbers given, use only these facts (do not invent any values), and ask for the specific correction or confirmation needed. Sign as "Ridgeline ',
# MAGIC          CASE d.route WHEN 'customer_service' THEN 'Customer Service' WHEN 'quality_engineer' THEN 'Supplier Quality'
# MAGIC                       WHEN 'buyer' THEN 'Purchasing' WHEN 'data_review' THEN 'Document Review' ELSE 'Accounts Payable' END, '".')) AS draft,
# MAGIC        run_label, current_timestamp()
# MAGIC FROM document_exceptions x JOIN exception_decisions d USING (exception_id)
# MAGIC WHERE coalesce(d.route, 'data_review') <> 'auto_resolve'
# MAGIC   AND NOT EXISTS (SELECT 1 FROM draft_replies r WHERE r.exception_id = x.exception_id);

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8.4 The action list
# MAGIC What each owner opens in the morning: the decision, the hold recommendation and a ready-to-edit email.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What a buyer, AP clerk or quality engineer opens in the morning: the decision and a ready-to-edit email
# MAGIC SELECT x.exception_type, x.counterparty, x.reference, d.route, round(d.hold_probability, 2) AS hold_probability,
# MAGIC        round(d.business_impact, 1) AS impact_0_to_3, x.amount_at_risk_usd, r.recipient, r.draft
# MAGIC FROM document_exceptions x
# MAGIC JOIN exception_decisions d USING (exception_id)
# MAGIC LEFT JOIN draft_replies r USING (exception_id)
# MAGIC ORDER BY d.business_impact DESC, x.amount_at_risk_usd DESC NULLS LAST;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8.5 💬 Ask Genie — what to hold
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Which supplier invoices should we hold, and why?", before=('Section 8 (decisions)', 14))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8.6 🛠 Try it yourself
# MAGIC In **step 8.1**, add a new question — for example `"supplier_relationship": {"type": "score", "instructions": "How much does this exception hurt the supplier relationship?", "criteria": ["None", "Some", "A lot"]}` — or change the tone of the email in **step 8.3** (*"friendly but firm"*, *"in Spanish for Mexican suppliers"*). Run `TRUNCATE TABLE exception_decisions` (or `draft_replies`) in a new cell and re-run **steps 8.1–8.4**. To see a new question's answer, query it directly: `SELECT decision:response:answers:supplier_relationship:score::DOUBLE FROM exception_decisions`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Every morning after that — the daily run
# MAGIC
# MAGIC The backfill is done. From here on the **same job** runs daily and processes **only new mail**: the AI Function tasks pick up only new files and rows, and the notebook tasks skip what they've seen. **Cost scales with new documents, not history.**
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | **9.1** | Delivers **today's mail** — new POs, invoices (one is a copy of an invoice we already have), a certificate with an out-of-tolerance dimension, and a newsletter |
# MAGIC | **9.2** | Runs the job again, as the daily run would at 6:00 AM — watch how much faster it is |
# MAGIC | **9.3–9.4** | Show exactly what each run processed and what's new today |

# COMMAND ----------

# Deliver the batch of mail held back in step 0.2 (dated today). In production this is your mailbox-to-volume integration.
n_emails, n_files = deliver_mail("today")
print(f"Delivered {n_emails} new emails with {n_files} PDF attachments to {INBOX}")
display(spark.sql("SELECT received_ts, split(mailbox, '@')[0] AS mailbox, from_address, subject, attachment_name "
                  "FROM emails WHERE to_date(received_ts) = current_date() ORDER BY received_ts"))

# COMMAND ----------

daily_run = run_job("daily job")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- What each run processed: the backfill (this notebook) vs the daily job (today's mail only)
# MAGIC SELECT processed_by AS run, doc_type, count(*) AS documents
# MAGIC FROM documents
# MAGIC GROUP BY ALL
# MAGIC ORDER BY run, documents DESC;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- What each run processed: the backfill (this notebook) vs the daily job (today's mail only)
# MAGIC SELECT processed_by AS run, doc_type, count(*) AS documents
# MAGIC FROM documents
# MAGIC GROUP BY ALL
# MAGIC ORDER BY run, documents DESC) SELECT `run`,SUM(`documents`) `column_6410014685`,`doc_type` FROM q GROUP BY `run`,`doc_type`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Exceptions the daily job found in today's mail, with the ai_decide recommendation and the ai_gen draft
# MAGIC SELECT x.exception_type, x.counterparty, x.reference, x.detail, d.route, round(d.hold_probability, 2) AS hold_probability, r.draft
# MAGIC FROM document_exceptions x
# MAGIC JOIN exception_decisions d USING (exception_id)
# MAGIC LEFT JOIN draft_replies r USING (exception_id)
# MAGIC WHERE x.processed_by = 'daily job'
# MAGIC ORDER BY d.business_impact DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 9.5 💬 Ask Genie — today
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("What did the daily job process today, and which new exceptions need attention?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 9.6 🛠 Try it yourself
# MAGIC Open the job run from **step 9.2** and compare it with the backfill run from **step 4.2** in the job's **Runs** tab — same tasks, a fraction of the rows. Then open the job → **Schedules & Triggers**: either **unpause** the 6:00 AM schedule, or — better for an inbox — replace it with a **file arrival trigger** on the `inbox` volume, so the job runs minutes after new attachments land. Add an **email notification** on failure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Executive summary
# MAGIC The whole play in one table — then one more `ai_gen` call turns the exceptions into a morning briefing (step 10.2).

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'Inbox' AS area, 'Emails triaged' AS metric, CAST(count(*) AS STRING) AS value FROM email_routing
# MAGIC UNION ALL SELECT 'Inbox', 'Emails that needed action', CAST(count_if(workflow <> 'no_action') AS STRING) FROM email_routing
# MAGIC UNION ALL SELECT 'Documents', 'Attachments parsed', CAST(count(*) AS STRING) FROM documents
# MAGIC UNION ALL SELECT 'Documents', 'Pages parsed', CAST(sum(page_count) AS STRING) FROM documents
# MAGIC UNION ALL SELECT 'Extraction', 'Customer order lines captured', CAST(count(*) AS STRING) FROM customer_order_lines
# MAGIC UNION ALL SELECT 'Extraction', 'Customer order value captured (USD)', format_number(sum(line_value), 0) FROM customer_order_lines
# MAGIC UNION ALL SELECT 'Extraction', 'Supplier invoice lines captured', CAST(count(*) AS STRING) FROM supplier_invoice_lines
# MAGIC UNION ALL SELECT 'Extraction', 'Inspection measurements captured', CAST(count(*) AS STRING) FROM inspection_results
# MAGIC UNION ALL SELECT 'Exceptions', 'Exceptions flagged', CAST(count(*) AS STRING) FROM document_exceptions
# MAGIC UNION ALL SELECT 'Exceptions', 'Dollar impact flagged (USD)', format_number(sum(amount_at_risk_usd), 0) FROM document_exceptions
# MAGIC UNION ALL SELECT 'Exceptions', 'Recommended holds (ai_decide)', CAST(count_if(d.hold_probability >= 0.5) AS STRING) FROM exception_decisions d JOIN document_exceptions USING (exception_id)
# MAGIC UNION ALL SELECT 'Exceptions', 'Emails drafted (ai_gen)', CAST(count(*) AS STRING) FROM draft_replies JOIN document_exceptions USING (exception_id)
# MAGIC UNION ALL SELECT 'Effort', 'Manual keying avoided (hours, at an illustrative 6 min / document)', format_number(count(*) * 6 / 60.0, 1) FROM documents;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- One more ai_gen call: turn the exception table into a briefing a plant or supply-chain lead can read in 30 seconds
# MAGIC SELECT ai_gen(concat(
# MAGIC   'You are an operations analyst at Ridgeline Precision Components, a manufacturer. Write a morning briefing for the operations lead: ',
# MAGIC   '4-6 bullets, max 150 words, most urgent first, with company names, document numbers and dollar amounts. Data (JSON): ',
# MAGIC   (SELECT to_json(transform(sort_array(collect_list(named_struct('k', -coalesce(amount_at_risk_usd, 0), 'v',
# MAGIC                     named_struct('type', exception_type, 'counterparty', counterparty, 'reference', reference, 'detail', detail, 'usd', amount_at_risk_usd)))),
# MAGIC                   x -> x.v))
# MAGIC    FROM (SELECT * FROM document_exceptions WHERE exception_type <> 'LOW_CONFIDENCE_EXTRACTION'
# MAGIC          ORDER BY amount_at_risk_usd DESC NULLS LAST LIMIT 15)))) AS morning_briefing;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Your turn — ask anything, then build more
# MAGIC **Ask your own question:** pick one in the **`question`** widget — or type anything in **`custom_question`** — and re-run **step 11.1**. Click **Continue this conversation in Genie** to drill down with follow-ups and charts.

# COMMAND ----------

question = dbutils.widgets.get("custom_question").strip() or dbutils.widgets.get("question")
ask(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11.2 🚀 Build more — ideas to take this further
# MAGIC | Idea | How | Effort |
# MAGIC |---|---|---|
# MAGIC | **Trigger on arrival** instead of 6:00 AM | Lakeflow Job **file arrival trigger** on the `inbox` volume — see step 9.6 | 5 min |
# MAGIC | **Scans and phone photos** of delivery notes or handwritten POs | Drop JPG / PNG / TIFF into the volume — the `parse_attachments` task handles them | 5 min |
# MAGIC | **Orders in the email body** (step 5.3 found some) | Add an **Information Extraction** task on `emails.email_text` for `customer_purchase_order` emails without an attachment | 15 min |
# MAGIC | **Build and tune schemas visually** | Agent Bricks **Information Extraction** / **Classification** — then *Import from Agent* into the job task | 30 min |
# MAGIC | **Supplier scorecard** | **AI/BI dashboard** on `document_exceptions` by supplier and month, with Genie built in | 30 min |
# MAGIC | **Ask the documents** — *"what's the max temperature of the SealTech O-ring?"* | `ai_prep_search` on `parsed_documents` + **Vector Search**, or a **Knowledge Assistant** (Agent Bricks) over the `inbox` volume | 1 hour |
# MAGIC | **Version the pipeline as code** | Export the job as **Declarative Automation Bundles** YAML (`ai_function_task` is supported) and deploy dev → prod | 1 hour |
# MAGIC | **Human-in-the-loop review app** | A **Databricks App** (with **Lakebase**) over `extraction_review_queue` and `exception_decisions` to approve, correct and send drafts | 1 day |
# MAGIC | **Write back to the ERP** | Post approved order lines / invoice holds to SAP, Oracle or NetSuite via API or partner connector as one more task | 1–2 days |
# MAGIC | **Share Genie** with the business | Open the Genie agent → **Share** → add your buyers, AP and quality groups | 2 min |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. What you just saw — and where to take it next
# MAGIC
# MAGIC * **Document intelligence as native job tasks.** `ai_classify` triaged the inbox, `ai_parse_document` read every PDF and `ai_extract` pulled line items, specs and measurements with confidence scores — each as an **AI Function task** that processes only new files and rows. `ai_decide` routed and prioritized every exception and `ai_gen` drafted the replies, as notebook tasks in the same job. No OCR templates, no ML infrastructure, all governed by Unity Catalog.
# MAGIC * **Design cheaply, run at scale.** Every task was previewed on a few rows with the same settings before the job ran.
# MAGIC * **AI extracts, SQL validates.** The exception rules are plain SQL against your ERP data — auditable, explainable and easy to change.
# MAGIC * **Backfill once, then pay only for new mail.** The job's first run was the 45-day backfill; the daily run processed only today's mail.
# MAGIC * **Genie turns the results into self-service.** Buyers, AP clerks, quality engineers and customer service ask in plain English; Genie writes the SQL against the same governed tables.
# MAGIC
# MAGIC **Production path**
# MAGIC 1. **Land real mail** — a Microsoft Graph / Gmail API job (or Power Automate / Logic Apps rule) writes attachments to the `inbox` volume and one row per email to `emails`.
# MAGIC 2. **Connect the ERP** — item, customer, supplier, PO and receipt tables via Lakeflow Connect or partner connectors; swap the views in *Bring your own data* (step 0.8).
# MAGIC 3. **Turn on the job** — unpause the schedule or add a file arrival trigger (step 9.6), add failure notifications, and manage it as a bundle.
# MAGIC 4. **Close the loop** — a review app for low-confidence values and drafts, write-back of approved orders and invoice holds to the ERP, and an AI/BI supplier scorecard.
# MAGIC
# MAGIC **Docs:** [AI Function tasks for jobs](https://docs.databricks.com/aws/en/jobs/tasks/ai-functions/) · [ai_parse_document](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document) · [ai_classify](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_classify) · [ai_extract](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_extract) · [ai_decide](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_decide) · [ai_gen](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_gen) · [Lakeflow Jobs](https://docs.databricks.com/aws/en/jobs/) · [Genie](https://docs.databricks.com/aws/en/genie/)
