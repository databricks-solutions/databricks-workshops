# Databricks notebook source

# MAGIC %md
# MAGIC # Patient Demand, Workforce & Claims Intelligence for Healthcare
# MAGIC **For providers and payers:** forecast patient demand, predict no-shows, close staffing gaps and predict claim denials — then let anyone ask Genie. Powered by Databricks AI Functions.
# MAGIC
# MAGIC **Industry:** Healthcare & Life Sciences — Providers (staffing & schedule optimization) and Payers (Medicaid / Medicare payment prediction)
# MAGIC
# MAGIC Every hospital runs the same weekly fire drill: *How many patients will we have by site and shift? Who won't show up? Where will we be short-staffed?*
# MAGIC This notebook answers all three **in SQL, in minutes, with no ML pipelines, feature stores, or model endpoints to manage** — using Databricks AI Functions — and then hands the answers to the business through **Genie**, so anyone can ask in plain English instead of writing SQL.
# MAGIC
# MAGIC | Step | Question | Databricks capability |
# MAGIC |---|---|---|
# MAGIC | 1 | What census / visit volume should we expect by **site × unit × shift** for the next N days? | `ai_forecast` |
# MAGIC | 2 | Which recent days were abnormal (surges, diversions, downtime)? | `ai_detect_anomalies` |
# MAGIC | 3 | Which upcoming appointments will **no-show**, and what does that do to real demand? | `ai_predict_class` |
# MAGIC | 4 | **Why** did our no-show rate go up this quarter? | `ai_top_drivers` |
# MAGIC | 5 | Where are the **staffing gaps** — and what do they cost? | all of the above, joined |
# MAGIC | 6 | *(Payers)* Which pending Medicaid / Medicare claims will be **denied**, why are denials rising, and what cash will go out the door? | `ai_predict_class`, `ai_top_drivers`, `ai_forecast` |
# MAGIC | 💬 | After every step: **ask Genie** the same question in plain English — no SQL | **Genie** agent on Unity Catalog |
# MAGIC | 🛠 | After every step: a **Try it yourself** exercise, and a *Build more* list at the end | |
# MAGIC
# MAGIC ---
# MAGIC ### How to run
# MAGIC 1. Keep the default **Serverless** compute. Steps that use `ai_forecast` / `ai_detect_anomalies` (which run on Databricks SQL) automatically send their SQL to a **Serverless or Pro SQL warehouse** in your workspace.
# MAGIC 2. A **workspace admin** enables **Predictive AI Functions** (Beta) and **AI Forecast V2** on the **Previews** page (profile menu → *Previews*). **If any AI Function cell fails, this is the first thing to check** — ask your admin to turn these previews on.
# MAGIC 3. Run the **Setup** steps (**S.1–S.3**) and set the widgets (`catalog`, `schema`, …) to a location where you can create tables. Then either **Run all** (about 7 minutes), or — better for learning — run section by section: each step ends with a **💬 Ask Genie** cell and a **🛠 Try it yourself** exercise.
# MAGIC 4. **Every section and cell is numbered** (S.1, 0.1, 1.1 …) — look for the number in each cell's title.
# MAGIC 5. **New to some of the code?** Ask **Genie Code** to explain any cell: click into it, open Genie Code and type `/explain` (or ask *"what does this cell do?"*).
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → enable **Predictive AI Functions** (Beta) and **AI Forecast V2**. Then re-run the failed cell.
# MAGIC
# MAGIC > **No data? No problem.** Section 0 (steps 0.1–0.7) generates realistic, fully synthetic data (no PHI). **Have your own data?** See *"Bring your own data"* (step 0.12) — swap in views over your tables and every downstream cell keeps working.
# MAGIC
# MAGIC *All names, sites, patients and claims in this notebook are fictitious.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## S. Setup — choose where to write the demo tables
# MAGIC Three short setup steps. The **widgets** (the notebook's parameters) drive everything below.
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | **S.1** | Creates the widgets |
# MAGIC | **S.2** | Turns the widget values into SQL session variables and creates / selects the schema |
# MAGIC | **S.3** | Loads the Python helper functions used later (Unity Catalog links, SQL warehouse runner, Genie) |
# MAGIC
# MAGIC | Widget | Meaning | Default |
# MAGIC |---|---|---|
# MAGIC | `catalog` / `schema` | Where tables are created in **Unity Catalog** (schema is created if missing) | your current catalog / `hls_ai_functions_demo` |
# MAGIC | `forecast_days` | Forecast horizon in days | `28` |
# MAGIC | `anomaly_threshold` | 0–1. Lower = more sensitive anomaly flagging | `0.95` |
# MAGIC | `site` | Filter for the display cells | `All` |
# MAGIC | `question` / `custom_question` | What to ask Genie in step 8.1 | starter questions |
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
    "Which units are understaffed in the next 7 days, and by how many staff?",
    "How many no-shows should we expect each day next week?",
    "Which pending Medicaid Managed Care claims are most likely to be denied, and why?",
    "What is the forecast patient volume for Harbor View Emergency Department next week by shift?",
    "Which site-shift-days were flagged as anomalies in the last 60 days?",
    "What is the total premium-pay exposure from understaffed shifts by site?",
]
dbutils.widgets.text("catalog", "", "1. Catalog (blank = current)")
dbutils.widgets.text("schema", "hls_ai_functions_demo", "2. Schema")
dbutils.widgets.dropdown("forecast_days", "28", ["14", "28", "56"], "3. Forecast horizon (days)")
dbutils.widgets.dropdown("anomaly_threshold", "0.95", ["0.90", "0.95", "0.99"], "4. Anomaly sensitivity (lower = more alerts)")
dbutils.widgets.dropdown("site", "All", ["All", "Harbor View Medical Center", "Cedar Ridge Community Hospital", "Pine Valley Ambulatory Clinic"], "5. Site filter")
dbutils.widgets.dropdown("question", GENIE_QUESTIONS[0], GENIE_QUESTIONS, "6. Ask Genie (any time)")
dbutils.widgets.text("custom_question", "", "7. ...or type your own Genie question")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Turn the widgets into session variables (with safe defaults if a widget is left blank).
# MAGIC -- Changed a widget? Re-run this cell (S.2) and the helpers (S.3), then the cells you want to refresh.
# MAGIC DECLARE OR REPLACE VARIABLE target_catalog    STRING DEFAULT coalesce(nullif(:catalog, ''), current_catalog());
# MAGIC DECLARE OR REPLACE VARIABLE target_schema     STRING DEFAULT coalesce(nullif(:schema, ''), 'hls_ai_functions_demo');
# MAGIC DECLARE OR REPLACE VARIABLE forecast_days     INT    DEFAULT CAST(coalesce(nullif(:forecast_days, ''), '28') AS INT);
# MAGIC DECLARE OR REPLACE VARIABLE anomaly_threshold DOUBLE DEFAULT CAST(coalesce(nullif(:anomaly_threshold, ''), '0.95') AS DOUBLE);
# MAGIC DECLARE OR REPLACE VARIABLE site_filter       STRING DEFAULT coalesce(nullif(:site, ''), 'All');
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS IDENTIFIER(target_catalog || '.' || target_schema)
# MAGIC COMMENT 'HLS AI Functions industry play: patient demand, no-show & staffing forecasting (synthetic data)';
# MAGIC
# MAGIC USE IDENTIFIER(target_catalog || '.' || target_schema);
# MAGIC
# MAGIC SELECT current_catalog() AS catalog, current_schema() AS schema, forecast_days, anomaly_threshold, site_filter;

# COMMAND ----------

# MAGIC %md
# MAGIC ### S.3 Load the helper functions
# MAGIC The next cell **doesn't create any tables or change any data** — it only defines Python helpers the rest of the notebook calls, so later cells stay short and readable:
# MAGIC
# MAGIC | Helper | Used in | What it does |
# MAGIC |---|---|---|
# MAGIC | `WorkspaceClient()`, `CATALOG`, `SCHEMA` | everywhere | Connects to this workspace with **your** identity and remembers where the demo tables live (from S.2) |
# MAGIC | `pick_warehouse()` / `run_on_sql_warehouse()` | steps 1.1, 2.1, 6.5 | Finds a **Serverless or Pro SQL warehouse** and sends `ai_forecast` / `ai_detect_anomalies` SQL to it *exactly as written* (these functions run on Databricks SQL), passing in your widget values |
# MAGIC | `uc_url()`, `button()`, `show_uc_cards()` | step 0.8 and later | Builds clickable **Unity Catalog** links and the table cards (schema · sample · lineage) |
# MAGIC | `ask_genie()`, `show_genie_answer()` | every 💬 Ask Genie step | Asks the **Genie agent** a question through the Genie API and renders the answer, the SQL Genie wrote and a *Continue in Genie* link |
# MAGIC
# MAGIC When it finishes it prints the catalog / schema and the SQL warehouse it will use. **Re-run this cell whenever you change the `catalog` or `schema` widget** (after S.2).
# MAGIC
# MAGIC > 💡 **Don't fully understand a cell?** Ask **Genie Code** to explain it: click into the cell, open Genie Code and type `/explain` — or just ask *"what does this cell do?"* in plain English.

# COMMAND ----------

# Helpers: Unity Catalog links, SQL warehouse runner (for ai_forecast / ai_detect_anomalies) and Genie
import json, time, uuid, html, re
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
CATALOG, SCHEMA = spark.sql("SELECT current_catalog(), current_schema()").first()
HOST = w.config.host.rstrip("/")
try:
    ORG = f"?o={w.get_workspace_id()}"
except Exception:
    ORG = ""

def pick_warehouse():
    whs = [x for x in w.warehouses.list() if x.enable_serverless_compute or "PRO" in str(x.warehouse_type)]
    if not whs:
        raise Exception("Create a Serverless or Pro SQL warehouse — ai_forecast, ai_detect_anomalies and Genie run on Databricks SQL.")
    whs.sort(key=lambda x: ("RUNNING" not in str(x.state), not x.enable_serverless_compute))
    return whs[0]
WAREHOUSE = pick_warehouse()

def run_on_sql_warehouse(statement):
    """Runs the SQL exactly as written on the SQL warehouse, with the notebook's widget values available."""
    fd = int(dbutils.widgets.get("forecast_days") or 28)
    at = float(dbutils.widgets.get("anomaly_threshold") or 0.95)
    wrapped = (f"BEGIN\n  DECLARE forecast_days INT DEFAULT {fd};\n  DECLARE anomaly_threshold DOUBLE DEFAULT {at};\n"
               f"{statement.strip()}\nEND")
    r = w.statement_execution.execute_statement(statement=wrapped, warehouse_id=WAREHOUSE.id,
                                                catalog=CATALOG, schema=SCHEMA, wait_timeout="50s")
    while r.status.state.value in ("PENDING", "RUNNING"):
        time.sleep(5)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise Exception(f"SQL warehouse '{WAREHOUSE.name}': {r.status.error.message if r.status.error else r.status.state}")
    print(f"✓ Ran on SQL warehouse '{WAREHOUSE.name}'")

def uc_url(*parts, tab=None):
    q = ORG + ((("&" if ORG else "?") + f"activeTab={tab}") if tab else "")
    return f"{HOST}/explore/data/" + "/".join(parts) + q

FONT = "font-family:DM Sans,Helvetica,Arial,sans-serif"
def button(label, url, color="#FF3621"):
    return (f'<a href="{url}" target="_blank" style="display:inline-block;margin:4px 8px 4px 0;padding:8px 14px;'
            f'background:{color};color:white;border-radius:6px;text-decoration:none;font-weight:600;{FONT};font-size:13px">'
            f'{html.escape(label)} ↗</a>')

TABLE_INFO = {
    "dim_care_unit": ("Source", "Care units and staffing ratios"),
    "census_history": ("Source", "2 years of census / visit volume by site × unit × shift"),
    "appointments": ("Source", "Outpatient appointments with no-show outcomes"),
    "staff_schedule": ("Source", "Published staff schedule for the next 4 weeks"),
    "claims": ("Source", "Payer claims — Medicaid, Medicare Advantage, D-SNP, Marketplace"),
    "claim_payments": ("Source", "Daily paid amount by line of business"),
    "census_forecast": ("ai_forecast", "Forecast volume with 80% prediction interval"),
    "census_anomalies": ("ai_detect_anomalies", "Expected value, bounds and anomaly score per day"),
    "noshow_backtest": ("ai_predict_class", "Backtest: predicted vs actual no-shows"),
    "noshow_predictions": ("ai_predict_class", "No-show probability for every upcoming appointment"),
    "staffing_gaps": ("Insight", "Required vs scheduled staff, status and cost per shift"),
    "claim_outcome_predictions": ("ai_predict_class", "Predicted Paid / Denied / Pended for pending claims"),
    "payment_forecast": ("ai_forecast", "Forecast paid amount by line of business"),
}
ROLE_COLOR = {"Source": "#5A6F77", "Insight": "#00A972"}

def show_uc_cards(roles=None, title=None):
    existing = {r.tableName for r in spark.sql(f"SHOW TABLES IN `{CATALOG}`.`{SCHEMA}`").collect()}
    cards = []
    for t, (role, desc) in TABLE_INFO.items():
        if t not in existing or (roles and role not in roles):
            continue
        n = spark.table(f"`{CATALOG}`.`{SCHEMA}`.`{t}`").count()
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

def _api(method, path, body=None):
    return w.api_client.do(method, path, body=body)

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

print(f"Writing to {CATALOG}.{SCHEMA} · SQL warehouse for ai_forecast / ai_detect_anomalies / Genie: '{WAREHOUSE.name}'")
displayHTML(button(f"Open {CATALOG}.{SCHEMA} in Unity Catalog", uc_url(CATALOG, SCHEMA), "#1B3139"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Generate synthetic data *(skip 0.1–0.6 if you bring your own)*
# MAGIC
# MAGIC A fictitious 3-site health system — **Harbor View Medical Center**, **Cedar Ridge Community Hospital**, and **Pine Valley Ambulatory Clinic** — with two years of history that behaves like the real thing:
# MAGIC
# MAGIC * **Weekly seasonality** (Monday ED rush, lighter weekend electives), **respiratory-season peaks** each winter, **holiday dips**, and **growth** at specific sites
# MAGIC * **Planted events** for the anomaly detector to find: a respiratory surge, an ICU mass-casualty night, and an ambulance diversion
# MAGIC * **Appointments** with realistic no-show behavior (lead time, prior no-shows, payer, distance, reminders, telehealth) — plus a **planted root cause** for `ai_top_drivers` to discover
# MAGIC * A **staffing schedule** built the way many hospitals still build it: from a static template that doesn't anticipate demand
# MAGIC * **Payer claims** for Medicaid Managed Care, Medicare Advantage, D-SNP and Marketplace members
# MAGIC
# MAGIC All randomness is hash-seeded, so results are reproducible, and dates are relative to today, so the data always looks current.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Care units and their staffing standards (patients per RN, or visits per clinician for the clinic)
# MAGIC CREATE OR REPLACE TABLE dim_care_unit AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('Harbor View Medical Center',     'Emergency Department',     'ED',        4.0, 3),
# MAGIC   ('Harbor View Medical Center',     'Medical-Surgical',         'Inpatient', 5.0, 3),
# MAGIC   ('Harbor View Medical Center',     'Intensive Care Unit',      'ICU',       2.0, 3),
# MAGIC   ('Cedar Ridge Community Hospital', 'Emergency Department',     'ED',        4.0, 2),
# MAGIC   ('Cedar Ridge Community Hospital', 'Medical-Surgical',         'Inpatient', 5.0, 2),
# MAGIC   ('Pine Valley Ambulatory Clinic',  'Primary & Specialty Care', 'Clinic',   16.0, 1)
# MAGIC AS t(site, unit, unit_type, patients_per_staff, min_staff);

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Outpatient appointments: 2 years of history (with outcomes) + the next 4 weeks (outcome unknown = NULL)
# MAGIC CREATE OR REPLACE TABLE appointments AS
# MAGIC WITH days AS (
# MAGIC   SELECT explode(sequence(date_sub(current_date(), 730), date_add(current_date(), 27))) AS appt_date
# MAGIC ),
# MAGIC slots AS (
# MAGIC   SELECT d.appt_date, s.shift,
# MAGIC          CAST(round(
# MAGIC            CASE WHEN s.shift = 'Day (07-15)' THEN
# MAGIC                   CASE dayofweek(d.appt_date) WHEN 1 THEN 14 WHEN 7 THEN 40 ELSE 100 END
# MAGIC                 ELSE
# MAGIC                   CASE dayofweek(d.appt_date) WHEN 1 THEN 0  WHEN 7 THEN 12 ELSE 36 END
# MAGIC            END * pow(1.05, datediff(d.appt_date, date_sub(current_date(), 730)) / 365.0)
# MAGIC          ) AS INT) AS n_slots
# MAGIC   FROM days d CROSS JOIN (SELECT explode(array('Day (07-15)', 'Evening (15-23)')) AS shift) s
# MAGIC ),
# MAGIC appts AS (
# MAGIC   SELECT appt_date, shift, slot
# MAGIC   FROM slots LATERAL VIEW explode(sequence(1, n_slots)) e AS slot
# MAGIC   WHERE n_slots > 0
# MAGIC ),
# MAGIC r AS (  -- deterministic pseudo-random draws per appointment
# MAGIC   SELECT *,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'payer'), 100000) / 100000.0 AS u_payer,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'age'),   100000) / 100000.0 AS u_age,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'vt'),    100000) / 100000.0 AS u_vt,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'lead'),  100000) / 100000.0 AS u_lead,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'prior'), 100000) / 100000.0 AS u_prior,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'dist'),  100000) / 100000.0 AS u_dist,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'rem'),   100000) / 100000.0 AS u_rem,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'tele'),  100000) / 100000.0 AS u_tele,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'hour'),  100000) / 100000.0 AS u_hour,
# MAGIC     pmod(xxhash64(appt_date, shift, slot, 'ns'),    100000) / 100000.0 AS u_ns
# MAGIC   FROM appts
# MAGIC ),
# MAGIC f AS (
# MAGIC   SELECT *,
# MAGIC     CASE WHEN u_payer < 0.30 THEN 'Medicaid' WHEN u_payer < 0.55 THEN 'Medicare'
# MAGIC          WHEN u_payer < 0.93 THEN 'Commercial' ELSE 'Self-Pay' END                          AS payer,
# MAGIC     CASE WHEN u_vt < 0.30 THEN 'Follow-up' WHEN u_vt < 0.48 THEN 'New Patient'
# MAGIC          WHEN u_vt < 0.65 THEN 'Annual Wellness' WHEN u_vt < 0.80 THEN 'Behavioral Health'
# MAGIC          WHEN u_vt < 0.92 THEN 'Chronic Care Mgmt' ELSE 'Procedure' END                      AS visit_type,
# MAGIC     CAST(floor(pow(u_lead, 2) * 60) AS INT)                                                   AS lead_time_days,
# MAGIC     CAST(floor(pow(u_prior, 3) * 6) AS INT)                                                   AS prior_no_shows_12m,
# MAGIC     round(1 + pow(u_dist, 1.5) * 39, 1)                                                       AS distance_miles,
# MAGIC     u_rem < 0.62                                                                              AS reminder_confirmed,
# MAGIC     CASE WHEN shift = 'Day (07-15)' THEN 7 + CAST(floor(u_hour * 8) AS INT)
# MAGIC          ELSE 15 + CAST(floor(u_hour * 4) AS INT) END                                         AS appt_hour
# MAGIC   FROM r
# MAGIC ),
# MAGIC g AS (
# MAGIC   SELECT *,
# MAGIC     CASE WHEN payer = 'Medicare' THEN 65 + CAST(floor(u_age * 25) AS INT)
# MAGIC          ELSE 18 + CAST(floor(u_age * 47) AS INT) END                                         AS patient_age,
# MAGIC     (visit_type IN ('Follow-up', 'Behavioral Health') AND u_tele < 0.30)                      AS is_telehealth,
# MAGIC     datediff(current_date(), appt_date)                                                       AS days_ago
# MAGIC   FROM f
# MAGIC ),
# MAGIC scored AS (
# MAGIC   SELECT *,
# MAGIC     -3.1 + 0.03 * lead_time_days + 0.5 * prior_no_shows_12m
# MAGIC          + CASE payer WHEN 'Medicaid' THEN 0.55 WHEN 'Self-Pay' THEN 0.35 ELSE 0 END
# MAGIC          - 1.0 * CAST(reminder_confirmed AS INT) + 0.025 * distance_miles
# MAGIC          - 0.6 * CAST(is_telehealth AS INT) + CASE WHEN patient_age < 30 THEN 0.35 ELSE 0 END
# MAGIC          + CASE visit_type WHEN 'Behavioral Health' THEN 0.3 WHEN 'Annual Wellness' THEN -0.25 ELSE 0 END
# MAGIC          + CASE WHEN appt_hour >= 17 THEN 0.2 ELSE 0 END
# MAGIC          + CASE WHEN dayofweek(appt_date) = 2 THEN 0.15 ELSE 0 END
# MAGIC          -- Planted root cause: ~90 days ago the region's Medicaid non-emergency medical transport (NEMT)
# MAGIC          -- vendor changed. Medicaid patients who live far away and need to come in person started missing visits.
# MAGIC          + CASE WHEN payer = 'Medicaid' AND distance_miles > 12 AND NOT is_telehealth AND days_ago <= 90
# MAGIC                 THEN 1.1 ELSE 0 END AS no_show_logit
# MAGIC   FROM g
# MAGIC )
# MAGIC SELECT
# MAGIC   format_string('APT%07d', row_number() OVER (ORDER BY appt_date, shift, slot)) AS appointment_id,
# MAGIC   'Pine Valley Ambulatory Clinic'  AS site,
# MAGIC   'Primary & Specialty Care'       AS unit,
# MAGIC   appt_date, shift, appt_hour,
# MAGIC   date_format(appt_date, 'EEE')    AS day_of_week,
# MAGIC   patient_age,
# MAGIC   CASE WHEN patient_age < 30 THEN '18-29' WHEN patient_age < 45 THEN '30-44'
# MAGIC        WHEN patient_age < 65 THEN '45-64' ELSE '65+' END AS age_band,
# MAGIC   payer, visit_type, lead_time_days, prior_no_shows_12m, distance_miles,
# MAGIC   reminder_confirmed, is_telehealth,
# MAGIC   CASE WHEN appt_date >= current_date() THEN NULL
# MAGIC        ELSE u_ns < 1 / (1 + exp(-no_show_logit)) END AS no_show
# MAGIC FROM scored;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Daily census / visit volume by site x unit x shift (2 years of history, through yesterday)
# MAGIC CREATE OR REPLACE TABLE census_history AS
# MAGIC WITH days AS (
# MAGIC   SELECT explode(sequence(date_sub(current_date(), 730), date_sub(current_date(), 1))) AS census_date
# MAGIC ),
# MAGIC base AS (
# MAGIC   SELECT * FROM VALUES
# MAGIC     ('Harbor View Medical Center',     'Emergency Department', 40.0, 0.06, 0.18),
# MAGIC     ('Harbor View Medical Center',     'Medical-Surgical',     30.0, 0.03, 0.10),
# MAGIC     ('Harbor View Medical Center',     'Intensive Care Unit',  14.0, 0.02, 0.12),
# MAGIC     ('Cedar Ridge Community Hospital', 'Emergency Department', 24.0, 0.02, 0.18),
# MAGIC     ('Cedar Ridge Community Hospital', 'Medical-Surgical',     20.0, 0.10, 0.10)
# MAGIC   AS t(site, unit, base_volume, annual_growth, resp_season_amp)
# MAGIC ),
# MAGIC shifts AS (
# MAGIC   SELECT * FROM VALUES ('Day (07-15)', 1.00, 1.00), ('Evening (15-23)', 1.15, 0.98), ('Night (23-07)', 0.55, 0.95)
# MAGIC   AS t(shift, ed_mult, inpatient_mult)
# MAGIC ),
# MAGIC grid AS (
# MAGIC   SELECT d.census_date, b.*, s.shift, u.unit_type,
# MAGIC          CASE WHEN u.unit_type = 'ED' THEN s.ed_mult ELSE s.inpatient_mult END AS shift_mult,
# MAGIC          datediff(current_date(), d.census_date) AS days_ago
# MAGIC   FROM days d CROSS JOIN base b CROSS JOIN shifts s
# MAGIC   JOIN dim_care_unit u ON u.site = b.site AND u.unit = b.unit
# MAGIC ),
# MAGIC expected AS (
# MAGIC   SELECT *,
# MAGIC     base_volume * shift_mult
# MAGIC       * pow(1 + annual_growth, (730 - days_ago) / 365.0)                                       -- growth
# MAGIC       * (1 + resp_season_amp * cos(2 * pi() * (dayofyear(census_date) - 20) / 365.25))         -- winter peak
# MAGIC       * CASE WHEN unit_type = 'ED' THEN
# MAGIC                CASE dayofweek(census_date) WHEN 2 THEN 1.12 WHEN 3 THEN 1.04 WHEN 7 THEN 0.93 WHEN 1 THEN 0.96 ELSE 1.0 END
# MAGIC              ELSE CASE dayofweek(census_date) WHEN 7 THEN 0.92 WHEN 1 THEN 0.90 ELSE 1.0 END END  -- weekly pattern
# MAGIC       * CASE WHEN (month(census_date) = 12 AND day(census_date) IN (24, 25))
# MAGIC                OR (month(census_date) = 1 AND day(census_date) = 1)
# MAGIC                OR (month(census_date) = 7 AND day(census_date) = 4)
# MAGIC                OR (month(census_date) = 11 AND dayofweek(census_date) = 5 AND day(census_date) BETWEEN 22 AND 28)
# MAGIC              THEN CASE WHEN unit_type = 'ED' THEN 0.90 ELSE 0.85 END ELSE 1.0 END                 -- US holidays
# MAGIC       AS mu
# MAGIC   FROM grid
# MAGIC ),
# MAGIC noisy AS (
# MAGIC   SELECT *,
# MAGIC     -- approx. N(0,1) from a sum of 4 uniforms
# MAGIC     ((pmod(xxhash64(census_date, site, unit, shift, 'n1'), 100000)
# MAGIC     + pmod(xxhash64(census_date, site, unit, shift, 'n2'), 100000)
# MAGIC     + pmod(xxhash64(census_date, site, unit, shift, 'n3'), 100000)
# MAGIC     + pmod(xxhash64(census_date, site, unit, shift, 'n4'), 100000)) / 100000.0 - 2.0) * sqrt(3.0) AS z
# MAGIC   FROM expected
# MAGIC )
# MAGIC SELECT census_date, site, unit, unit_type, shift,
# MAGIC   CAST(greatest(0, round(
# MAGIC     (mu + z * CASE WHEN unit_type = 'ED' THEN sqrt(mu) ELSE 0.06 * mu END)
# MAGIC     -- Planted events for ai_detect_anomalies
# MAGIC     * CASE WHEN site = 'Harbor View Medical Center' AND unit = 'Emergency Department' AND days_ago BETWEEN 17 AND 21 THEN 1.45  -- respiratory surge
# MAGIC            WHEN site = 'Harbor View Medical Center' AND unit = 'Intensive Care Unit' AND shift = 'Night (23-07)' AND days_ago = 30 THEN 1.70  -- mass-casualty night
# MAGIC            WHEN site = 'Cedar Ridge Community Hospital' AND unit = 'Emergency Department' AND shift = 'Evening (15-23)' AND days_ago = 3 THEN 0.40  -- ambulance diversion
# MAGIC            ELSE 1.0 END
# MAGIC   )) AS INT) AS patient_volume
# MAGIC FROM noisy
# MAGIC UNION ALL
# MAGIC -- Clinic visit volume = patients who actually arrived for appointments + a few walk-ins
# MAGIC SELECT appt_date AS census_date, site, unit, 'Clinic' AS unit_type, shift,
# MAGIC        CAST(count_if(NOT no_show) + 2 + pmod(xxhash64(appt_date, shift, 'walkin'), 4) AS INT) AS patient_volume
# MAGIC FROM appointments
# MAGIC WHERE appt_date < current_date()
# MAGIC GROUP BY ALL;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Published staff schedule for the next 4 weeks, built the "traditional" way:
# MAGIC -- a static same-weekday template from the last 8 weeks of volume (doesn't anticipate seasonality, growth or no-shows),
# MAGIC -- minus a handful of call-outs already logged.
# MAGIC CREATE OR REPLACE TABLE staff_schedule AS
# MAGIC WITH template AS (
# MAGIC   SELECT h.site, h.unit, h.shift, dayofweek(h.census_date) AS dow, avg(h.patient_volume) AS avg_volume
# MAGIC   FROM census_history h
# MAGIC   WHERE h.census_date BETWEEN date_sub(current_date(), 63) AND date_sub(current_date(), 8)
# MAGIC   GROUP BY ALL
# MAGIC ),
# MAGIC future AS (
# MAGIC   SELECT explode(sequence(current_date(), date_add(current_date(), 27))) AS shift_date
# MAGIC ),
# MAGIC booked AS (
# MAGIC   SELECT appt_date, shift, count(*) AS booked_appts FROM appointments WHERE appt_date >= current_date() GROUP BY ALL
# MAGIC )
# MAGIC SELECT f.shift_date, t.site, t.unit, t.shift,
# MAGIC   greatest(u.min_staff,
# MAGIC     CAST(CASE WHEN u.unit_type = 'Clinic'
# MAGIC               THEN ceil(coalesce(b.booked_appts, 0) / u.patients_per_staff)    -- clinics staff to the booked schedule
# MAGIC               ELSE ceil(t.avg_volume / u.patients_per_staff) END AS INT)
# MAGIC     - CASE WHEN pmod(xxhash64(f.shift_date, t.site, t.unit, t.shift, 'callout'), 100) < 8 THEN 1 ELSE 0 END
# MAGIC   ) AS scheduled_staff
# MAGIC FROM future f
# MAGIC JOIN template t ON t.dow = dayofweek(f.shift_date)
# MAGIC JOIN dim_care_unit u ON u.site = t.site AND u.unit = t.unit
# MAGIC LEFT JOIN booked b ON b.appt_date = f.shift_date AND b.shift = t.shift;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Payer claims (Medicaid Managed Care, Medicare Advantage, D-SNP, ACA Marketplace): 18 months.
# MAGIC -- Claims received in the last 14 days are still in adjudication (outcome = NULL).
# MAGIC CREATE OR REPLACE TABLE claims AS
# MAGIC WITH days AS (SELECT explode(sequence(date_sub(current_date(), 540), date_sub(current_date(), 1))) AS received_date),
# MAGIC c AS (
# MAGIC   SELECT received_date, n
# MAGIC   FROM days LATERAL VIEW explode(sequence(1, CASE WHEN dayofweek(received_date) IN (1, 7) THEN 40 ELSE 160 END)) e AS n
# MAGIC ),
# MAGIC r AS (
# MAGIC   SELECT *,
# MAGIC     pmod(xxhash64(received_date, n, 'lob'),  100000) / 100000.0 AS u_lob,
# MAGIC     pmod(xxhash64(received_date, n, 'st'),   100000) / 100000.0 AS u_st,
# MAGIC     pmod(xxhash64(received_date, n, 'cat'),  100000) / 100000.0 AS u_cat,
# MAGIC     pmod(xxhash64(received_date, n, 'amt'),  100000) / 100000.0 AS u_amt,
# MAGIC     pmod(xxhash64(received_date, n, 'pa'),   100000) / 100000.0 AS u_pa,
# MAGIC     pmod(xxhash64(received_date, n, 'ch'),   100000) / 100000.0 AS u_ch,
# MAGIC     pmod(xxhash64(received_date, n, 'sub'),  100000) / 100000.0 AS u_sub,
# MAGIC     pmod(xxhash64(received_date, n, 'net'),  100000) / 100000.0 AS u_net,
# MAGIC     pmod(xxhash64(received_date, n, 'code'), 100000) / 100000.0 AS u_code,
# MAGIC     pmod(xxhash64(received_date, n, 'elig'), 100000) / 100000.0 AS u_elig,
# MAGIC     pmod(xxhash64(received_date, n, 'out'),  100000) / 100000.0 AS u_out,
# MAGIC     pmod(xxhash64(received_date, n, 'pend'), 100000) / 100000.0 AS u_pend,
# MAGIC     pmod(xxhash64(received_date, n, 'lag'),  100000) / 100000.0 AS u_lag
# MAGIC   FROM c
# MAGIC ),
# MAGIC f AS (
# MAGIC   SELECT *,
# MAGIC     CASE WHEN u_lob < 0.45 THEN 'Medicaid Managed Care' WHEN u_lob < 0.80 THEN 'Medicare Advantage'
# MAGIC          WHEN u_lob < 0.88 THEN 'Dual Eligible SNP' ELSE 'ACA Marketplace' END                         AS line_of_business,
# MAGIC     element_at(array('TX', 'FL', 'OH', 'GA', 'AZ'), CAST(floor(u_st * 5) AS INT) + 1)                   AS member_state,
# MAGIC     CASE WHEN u_cat < 0.30 THEN 'Primary Care' WHEN u_cat < 0.50 THEN 'Specialist'
# MAGIC          WHEN u_cat < 0.62 THEN 'Behavioral Health' WHEN u_cat < 0.72 THEN 'Imaging'
# MAGIC          WHEN u_cat < 0.80 THEN 'Outpatient Surgery' WHEN u_cat < 0.86 THEN 'Inpatient'
# MAGIC          WHEN u_cat < 0.93 THEN 'Home Health' ELSE 'DME' END                                            AS service_category,
# MAGIC     CASE WHEN u_ch < 0.85 THEN 'EDI 837' WHEN u_ch < 0.95 THEN 'Provider Portal' ELSE 'Paper' END       AS submission_channel,
# MAGIC     CAST(floor(pow(u_sub, 2.2) * 130) AS INT)                                                            AS days_to_submit,
# MAGIC     CASE WHEN u_net < 0.88 THEN 'In-Network' ELSE 'Out-of-Network' END                                  AS provider_network,
# MAGIC     CASE WHEN u_code < 0.86 THEN 'Clean' WHEN u_code < 0.94 THEN 'Modifier Missing' ELSE 'Dx-Px Mismatch' END AS coding_check,
# MAGIC     CAST(1 + floor(pow(u_elig, 0.7) * 60) AS INT)                                                        AS member_months_enrolled,
# MAGIC     datediff(current_date(), received_date)                                                              AS days_ago
# MAGIC   FROM r
# MAGIC ),
# MAGIC g AS (
# MAGIC   SELECT *,
# MAGIC     round(CASE service_category
# MAGIC             WHEN 'Primary Care' THEN 120 WHEN 'Specialist' THEN 260 WHEN 'Behavioral Health' THEN 180
# MAGIC             WHEN 'Imaging' THEN 900 WHEN 'Outpatient Surgery' THEN 4200 WHEN 'Inpatient' THEN 18500
# MAGIC             WHEN 'Home Health' THEN 650 ELSE 1100 END * exp((u_amt - 0.5) * 1.2), 2)                      AS billed_amount,
# MAGIC     CASE WHEN service_category IN ('Inpatient', 'Outpatient Surgery', 'Imaging', 'Behavioral Health', 'DME', 'Home Health')
# MAGIC          THEN CASE WHEN u_pa < 0.85 THEN 'Approved' ELSE 'Missing' END
# MAGIC          ELSE 'Not Required' END                                                                         AS prior_auth_status
# MAGIC   FROM f
# MAGIC ),
# MAGIC s AS (
# MAGIC   SELECT *,
# MAGIC     -3.0 + CASE WHEN prior_auth_status = 'Missing' THEN 2.0 ELSE 0 END
# MAGIC          + CASE coding_check WHEN 'Dx-Px Mismatch' THEN 1.6 WHEN 'Modifier Missing' THEN 0.8 ELSE 0 END
# MAGIC          + CASE WHEN days_to_submit > 95 THEN 2.5 ELSE 0 END                         -- timely-filing limit
# MAGIC          + CASE WHEN provider_network = 'Out-of-Network' THEN 0.8 ELSE 0 END
# MAGIC          + CASE WHEN submission_channel = 'Paper' THEN 0.4 ELSE 0 END
# MAGIC          + CASE WHEN member_months_enrolled <= 2 THEN 0.6 ELSE 0 END                 -- eligibility churn
# MAGIC          -- Planted root cause: a new Medicaid MCO prior-auth / medical-necessity policy for
# MAGIC          -- Behavioral Health went live ~90 days ago in TX and FL.
# MAGIC          + CASE WHEN line_of_business = 'Medicaid Managed Care' AND service_category = 'Behavioral Health'
# MAGIC                      AND member_state IN ('TX', 'FL') AND days_ago <= 105 THEN 1.8 ELSE 0 END AS denial_logit
# MAGIC   FROM g
# MAGIC )
# MAGIC SELECT
# MAGIC   format_string('CLM%08d', row_number() OVER (ORDER BY received_date, n)) AS claim_id,
# MAGIC   received_date, line_of_business, member_state, service_category, submission_channel,
# MAGIC   days_to_submit, provider_network, coding_check, prior_auth_status, member_months_enrolled, billed_amount,
# MAGIC   CASE WHEN days_ago <= 14 THEN NULL
# MAGIC        WHEN u_out < 1 / (1 + exp(-denial_logit)) THEN 'Denied'
# MAGIC        WHEN u_pend < 0.05 + CASE WHEN billed_amount > 15000 THEN 0.12 ELSE 0 END THEN 'Pended'
# MAGIC        ELSE 'Paid' END AS claim_outcome,
# MAGIC   CAST(15 + floor(u_lag * 21) AS INT) AS days_to_pay
# MAGIC FROM s;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Paid-amount ledger (payer cash out) derived from adjudicated claims
# MAGIC CREATE OR REPLACE TABLE claim_payments AS
# MAGIC SELECT date_add(received_date, days_to_pay) AS paid_date, line_of_business,
# MAGIC        round(sum(billed_amount * CASE line_of_business WHEN 'Medicaid Managed Care' THEN 0.42
# MAGIC                                                        WHEN 'Medicare Advantage' THEN 0.58
# MAGIC                                                        WHEN 'Dual Eligible SNP' THEN 0.55 ELSE 0.66 END), 2) AS paid_amount,
# MAGIC        count(*) AS claims_paid
# MAGIC FROM claims
# MAGIC WHERE claim_outcome = 'Paid' AND date_add(received_date, days_to_pay) < current_date()
# MAGIC GROUP BY ALL;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'dim_care_unit' AS table_name, count(*) AS row_count FROM dim_care_unit UNION ALL
# MAGIC SELECT 'census_history', count(*) FROM census_history UNION ALL
# MAGIC SELECT 'appointments', count(*) FROM appointments UNION ALL
# MAGIC SELECT 'staff_schedule', count(*) FROM staff_schedule UNION ALL
# MAGIC SELECT 'claims', count(*) FROM claims UNION ALL
# MAGIC SELECT 'claim_payments', count(*) FROM claim_payments;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.8 Your source data in Unity Catalog
# MAGIC Every table is governed by **Unity Catalog** — permissions, lineage and audit come for free. Click a card to see the schema, sample data or lineage.

# COMMAND ----------

# MAGIC %sql WITH q AS (SELECT 'dim_care_unit' AS table_name, count(*) AS row_count FROM dim_care_unit UNION ALL
# MAGIC SELECT 'census_history', count(*) FROM census_history UNION ALL
# MAGIC SELECT 'appointments', count(*) FROM appointments UNION ALL
# MAGIC SELECT 'staff_schedule', count(*) FROM staff_schedule UNION ALL
# MAGIC SELECT 'claims', count(*) FROM claims UNION ALL
# MAGIC SELECT 'claim_payments', count(*) FROM claim_payments) SELECT `table_name`,SUM(`row_count`) `column_9101723582` FROM q GROUP BY `table_name`

# COMMAND ----------

show_uc_cards(roles={"Source"})

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.9 Meet your Genie agent
# MAGIC Steps 1–6 each end with **💬 Ask Genie**: the same insight, asked in plain English instead of SQL. That's how charge nurses, staffing coordinators and claims managers will use these results — no SQL, no waiting on the data team, and every answer respects **Unity Catalog** permissions.
# MAGIC
# MAGIC Step 0.9 creates (or reuses) the **Patient Demand, Workforce & Claims Genie** agent over this schema. Tables that later steps will build start as empty placeholders and fill in as you go.

# COMMAND ----------

# Create (or reuse) the Genie agent now, so you can ask it questions after every step
TABLE_COMMENTS = {
    "census_history": "Historical daily patient volume per site, unit, unit_type and shift.",
    "appointments": "Outpatient appointments; no_show is NULL for future appointments.",
    "staff_schedule": "Published staff schedule (scheduled_staff) per site, unit, shift and day.",
    "claims": "Payer claims; claim_outcome is NULL while the claim is still in adjudication.",
    "dim_care_unit": "Care units with staffing standard patients_per_staff and min_staff.",
    "census_forecast": "ai_forecast output: forecast patient volume per series_id ('site | unit | shift') and day, with 80% prediction interval (patient_volume_lower/upper).",
    "census_anomalies": "ai_detect_anomalies output: per series_id ('site | unit | shift') and day: expected volume, bounds, anomaly_score (0-1) and is_anomaly.",
    "noshow_predictions": "ai_predict_class output: one row per upcoming appointment with no_show_probability (0-1).",
    "staffing_gaps": "Staffing gap per site, unit, shift and day: required_staff vs scheduled_staff, status (UNDERSTAFFED / SURGE RISK / OVERSTAFFED / OK) and est_cost_usd.",
    "claim_outcome_predictions": "ai_predict_class output: claim_outcome_prediction (Paid / Denied / Pended) and claim_outcome_confidence for claims in adjudication.",
    "payment_forecast": "ai_forecast output: forecast daily paid_amount by line_of_business with prediction interval.",
}
COLUMN_COMMENTS = {
    ("staffing_gaps", "staff_gap"): "scheduled_staff minus required_staff; negative = short-staffed",
    ("staffing_gaps", "required_staff"): "Staff needed for expected_patients at the unit's staffing ratio",
    ("staffing_gaps", "est_cost_usd"): "Illustrative premium-pay (understaffed) or idle (overstaffed) cost in USD",
    ("noshow_predictions", "no_show_probability"): "Predicted probability (0-1) that the patient will not show up",
    ("census_anomalies", "anomaly_score"): "0-1; higher = more unusual versus expected volume",
}
GENIE_TABLES = sorted(TABLE_COMMENTS)
fq = lambda t: f"{CATALOG}.{SCHEMA}.{t}"
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
instructions = f"""This agent answers patient-demand, no-show, staffing and payer-claims questions for a health system.
- Staffing: use {fq('staffing_gaps')}. status = 'UNDERSTAFFED' means scheduled_staff < required_staff; staff_gap < 0 is the shortfall; est_cost_usd is premium-pay exposure. Shifts are 'Day (07-15)', 'Evening (15-23)', 'Night (23-07)'.
- History: use {fq('census_history')} (patient_volume per site, unit, shift and census_date).
- Forecasts: use {fq('census_forecast')}; series_id is 'site | unit | shift' (split on ' | ').
- Anomalies: use {fq('census_anomalies')} WHERE is_anomaly.
- No-shows: expected no-shows = SUM(no_show_probability) from {fq('noshow_predictions')}.
- Claims: predicted outcomes in {fq('claim_outcome_predictions')}; history in {fq('claims')}.
- Always state the date range and filters used, and round to 1 decimal place."""
serialized = {
    "version": 2,
    "config": {"sample_questions": sorted([{"id": hexid(), "question": [x]} for x in GENIE_QUESTIONS], key=lambda x: x["id"])},
    "data_sources": {"tables": [{"identifier": fq(t)} for t in GENIE_TABLES]},
    "instructions": {
        "text_instructions": [{"id": hexid(), "content": [instructions]}],
        "example_question_sqls": sorted([
            {"id": hexid(), "question": ["Which units are understaffed in the next 7 days?"],
             "sql": [f"SELECT shift_date, site, unit, shift, required_staff, scheduled_staff, staff_gap, est_cost_usd\\nFROM {fq('staffing_gaps')}\\n"
                     f"WHERE status = 'UNDERSTAFFED' AND shift_date < date_add(current_date(), 7)\\nORDER BY shift_date, staff_gap"]},
            {"id": hexid(), "question": ["Expected no-shows by day"],
             "sql": [f"SELECT appt_date, round(sum(no_show_probability), 1) AS expected_no_shows, count(*) AS booked_appointments\\n"
                     f"FROM {fq('noshow_predictions')}\\nGROUP BY appt_date ORDER BY appt_date"]},
        ], key=lambda x: x["id"]),
    },
}
GENIE_TITLE = f"Patient Demand, Workforce & Claims Genie ({SCHEMA})"
existing = next((s for s in _api("GET", "/api/2.0/genie/spaces").get("spaces", []) if s.get("title") == GENIE_TITLE), None)
if existing:
    GENIE_SPACE_ID = existing["space_id"]
    print(f"Reusing Genie agent '{GENIE_TITLE}'")
else:
    GENIE_SPACE_ID = _api("POST", "/api/2.0/genie/spaces", body={
        "title": GENIE_TITLE,
        "description": "Ask about patient volume, forecasts, anomalies, staffing gaps, predicted no-shows and claim denials — built on Databricks AI Functions outputs.",
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
# MAGIC #### 0.10 💬 Ask Genie — explore the source data
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Which site had the highest average daily patient volume over the last 30 days, by unit?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0.11 🛠 Try it yourself
# MAGIC Open the **`appointments`** card from **step 0.8** in Unity Catalog. Which columns do you think predict a no-show? You'll see in step 3.1 whether `ai_predict_class` agrees.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.12 Bring your own data
# MAGIC Every downstream cell reads only these tables. To run on real data, skip the generator cells (0.1–0.6) and point the names at your own tables with views. Minimum columns:
# MAGIC
# MAGIC | Table | Required columns | Typical source |
# MAGIC |---|---|---|
# MAGIC | `dim_care_unit` | `site, unit, unit_type, patients_per_staff, min_staff` | Nursing staffing grids / policy |
# MAGIC | `census_history` | `census_date, site, unit, unit_type, shift, patient_volume` | ADT feed (Epic Clarity / Cerner), ED tracking board |
# MAGIC | `appointments` | `appointment_id, site, unit, appt_date, shift` + any features + `no_show` (NULL for future) | Scheduling system (Epic Cadence, athenaOne, NextGen) |
# MAGIC | `staff_schedule` | `shift_date, site, unit, shift, scheduled_staff` | Workforce mgmt (API Healthcare, Kronos/UKG, Epic Staffing) |
# MAGIC | `claims` | `claim_id, received_date, line_of_business` + features + `claim_outcome` (NULL = pending) | Claims adjudication (Facets, QNXT, HealthRules) |
# MAGIC
# MAGIC ```sql
# MAGIC -- Example: point the notebook at your ADT data
# MAGIC CREATE OR REPLACE VIEW census_history AS
# MAGIC SELECT CAST(census_ts AS DATE) AS census_date, facility_name AS site, dept_name AS unit,
# MAGIC        dept_type AS unit_type, shift_name AS shift, count(DISTINCT encounter_id) AS patient_volume
# MAGIC FROM my_catalog.clinical.adt_census_snapshots
# MAGIC GROUP BY ALL;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0.13 Explore the history
# MAGIC Weekly patterns, winter peaks and site growth are all visible here. *Tip: click **+ → Visualization** on any result to chart it (e.g. line chart: `census_date` × `patient_volume`, grouped by `shift`).*

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT census_date, site, unit, shift, patient_volume
# MAGIC FROM census_history
# MAGIC WHERE (site_filter = 'All' OR site = site_filter)
# MAGIC   AND census_date >= date_sub(current_date(), 180)
# MAGIC ORDER BY site, unit, shift, census_date;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Forecast census & visit volume by site × unit × shift — `ai_forecast`
# MAGIC
# MAGIC One SQL call fits a forecast for **every series** (16 site/unit/shift combinations here — it works the same for thousands). Version 2 uses a **time-series foundation model** with built-in **US holiday effects** and **non-negative** forecasts, and returns prediction intervals you can staff against.
# MAGIC
# MAGIC *If AI Forecast V2 isn't enabled in this workspace yet, the cell automatically falls back to version 1 so the rest of the notebook still runs — check the `model_version` column.*
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → enable **Predictive AI Functions** (Beta) and **AI Forecast V2**. Then re-run the failed cell.
# MAGIC
# MAGIC > **Going further:** V2 also accepts **covariates** (`covariate_col`) — e.g. scheduled elective surgeries or booked appointments for future dates, weather, or local flu-positivity rates.

# COMMAND ----------

# MAGIC %sql WITH q AS (SELECT census_date, site, unit, shift, patient_volume
# MAGIC FROM census_history
# MAGIC WHERE (site_filter = 'All' OR site = site_filter)
# MAGIC   AND census_date >= date_sub(current_date(), 180)
# MAGIC ORDER BY site, unit, shift, census_date) SELECT `census_date`,SUM(`patient_volume`) `column_5994352485`,`site` FROM q GROUP BY `census_date`,`site`

# COMMAND ----------

# ai_forecast runs on Databricks SQL — this cell sends the SQL below, exactly as written, to your SQL warehouse.
run_on_sql_warehouse(r"""
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    -- Fallback: ai_forecast version 1 (Public Preview)
    CREATE OR REPLACE TABLE census_forecast AS
    SELECT f.*, 'ai_forecast v1 (fallback)' AS model_version
    FROM ai_forecast(
      observed                  => TABLE(SELECT census_date, concat_ws(' | ', site, unit, shift) AS series_id, patient_volume FROM census_history),
      horizon                   => date_add(current_date(), forecast_days),
      time_col                  => 'census_date',
      value_col                 => 'patient_volume',
      group_col                 => 'series_id',
      prediction_interval_width => 0.80
    ) f;
  END;

  CREATE OR REPLACE TABLE census_forecast AS
  SELECT f.*, 'ai_forecast v2' AS model_version
  FROM ai_forecast(
    observed                  => TABLE(SELECT census_date, concat_ws(' | ', site, unit, shift) AS series_id, patient_volume FROM census_history),
    horizon                   => date_add(current_date(), forecast_days),
    time_col                  => 'census_date',
    value_col                 => 'patient_volume',
    group_col                 => 'series_id',
    prediction_interval_width => 0.80,
    holiday_region            => 'US',
    positive_only             => true,
    version                   => '2'
  ) f;
END;
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Next 2 weeks by site / unit / shift, with an 80% prediction interval
# MAGIC SELECT f.census_date,
# MAGIC        split(f.series_id, ' \\| ')[0] AS site,
# MAGIC        split(f.series_id, ' \\| ')[1] AS unit,
# MAGIC        split(f.series_id, ' \\| ')[2] AS shift,
# MAGIC        round(f.patient_volume_forecast, 1) AS forecast,
# MAGIC        round(greatest(f.patient_volume_lower, 0), 1) AS low_80,
# MAGIC        round(f.patient_volume_upper, 1)    AS high_80,
# MAGIC        f.model_version
# MAGIC FROM census_forecast f
# MAGIC WHERE (site_filter = 'All' OR split(f.series_id, ' \\| ')[0] = site_filter)
# MAGIC   AND f.census_date < date_add(current_date(), 14)
# MAGIC ORDER BY site, unit, shift, census_date;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- History + forecast in one series — chart this as a line: x = census_date, y = volume, color = kind
# MAGIC SELECT census_date, 'actual' AS kind, patient_volume AS volume
# MAGIC FROM census_history
# MAGIC WHERE site = 'Harbor View Medical Center' AND unit = 'Emergency Department' AND shift = 'Evening (15-23)'
# MAGIC   AND census_date >= date_sub(current_date(), 120)
# MAGIC UNION ALL
# MAGIC SELECT census_date, 'forecast', round(patient_volume_forecast, 1)
# MAGIC FROM census_forecast WHERE series_id = 'Harbor View Medical Center | Emergency Department | Evening (15-23)'
# MAGIC UNION ALL
# MAGIC SELECT census_date, 'forecast high (80%)', round(patient_volume_upper, 1)
# MAGIC FROM census_forecast WHERE series_id = 'Harbor View Medical Center | Emergency Department | Evening (15-23)'
# MAGIC ORDER BY census_date, kind;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Next 2 weeks by site / unit / shift, with an 80% prediction interval
# MAGIC SELECT f.census_date,
# MAGIC        split(f.series_id, ' \\| ')[0] AS site,
# MAGIC        split(f.series_id, ' \\| ')[1] AS unit,
# MAGIC        split(f.series_id, ' \\| ')[2] AS shift,
# MAGIC        round(f.patient_volume_forecast, 1) AS forecast,
# MAGIC        round(greatest(f.patient_volume_lower, 0), 1) AS low_80,
# MAGIC        round(f.patient_volume_upper, 1)    AS high_80,
# MAGIC        f.model_version
# MAGIC FROM census_forecast f
# MAGIC WHERE (site_filter = 'All' OR split(f.series_id, ' \\| ')[0] = site_filter)
# MAGIC   AND f.census_date < date_add(current_date(), 14)
# MAGIC ORDER BY site, unit, shift, census_date) SELECT `census_date`,SUM(`forecast`) `column_3593072318`,`site` FROM q GROUP BY `census_date`,`site`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 💬 Ask Genie — the forecast
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

# MAGIC %sql WITH q AS (-- History + forecast in one series — chart this as a line: x = census_date, y = volume, color = kind
# MAGIC SELECT census_date, 'actual' AS kind, patient_volume AS volume
# MAGIC FROM census_history
# MAGIC WHERE site = 'Harbor View Medical Center' AND unit = 'Emergency Department' AND shift = 'Evening (15-23)'
# MAGIC   AND census_date >= date_sub(current_date(), 120)
# MAGIC UNION ALL
# MAGIC SELECT census_date, 'forecast', round(patient_volume_forecast, 1)
# MAGIC FROM census_forecast WHERE series_id = 'Harbor View Medical Center | Emergency Department | Evening (15-23)'
# MAGIC UNION ALL
# MAGIC SELECT census_date, 'forecast high (80%)', round(patient_volume_upper, 1)
# MAGIC FROM census_forecast WHERE series_id = 'Harbor View Medical Center | Emergency Department | Evening (15-23)'
# MAGIC ORDER BY census_date, kind) SELECT `census_date`,SUM(`volume`) `column_5888013016`,`kind` FROM q GROUP BY `census_date`,`kind`

# COMMAND ----------

ask("What is the forecast patient volume for Harbor View Emergency Department next week by shift?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.5 🛠 Try it yourself
# MAGIC Switch the **`forecast_days`** widget to `56`, re-run **S.2 and S.3** so the new value takes effect, then re-run **steps 1.1–1.3**. Then open `census_forecast` in Unity Catalog (cards in step 0.8) → **Lineage** to see it was built from `census_history`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Flag abnormal days — `ai_detect_anomalies`
# MAGIC
# MAGIC Before trusting history to staff the future, find the days that weren't normal. `ai_detect_anomalies` scores each point against what the preceding history predicted — per series, with **US holiday awareness** — so a quiet Christmas isn't a false alarm but a mid-week surge is.
# MAGIC
# MAGIC Use this as a **daily operations alert** (surges, ambulance diversions, EHR downtime, data-feed breaks) and to decide which days to exclude from staffing templates. Adjust the `anomaly_threshold` widget to tune sensitivity.
# MAGIC
# MAGIC *Fallback: if the Predictive AI Functions preview isn't enabled, a simple rolling-baseline method runs instead (`method` column).*
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → enable **Predictive AI Functions** (Beta) and **AI Forecast V2**. Then re-run the failed cell.

# COMMAND ----------

# ai_detect_anomalies runs on Databricks SQL — this cell sends the SQL below, exactly as written, to your SQL warehouse.
run_on_sql_warehouse(r"""
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    -- Fallback: rolling 28-day same-series baseline (mean ± 3 sd) — illustrative only
    CREATE OR REPLACE TABLE census_anomalies AS
    WITH w AS (
      SELECT census_date, concat_ws(' | ', site, unit, shift) AS series_id, CAST(patient_volume AS DOUBLE) AS patient_volume,
             avg(patient_volume)    OVER (PARTITION BY site, unit, shift ORDER BY census_date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS mu,
             stddev(patient_volume) OVER (PARTITION BY site, unit, shift ORDER BY census_date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS sd
      FROM census_history
    )
    SELECT census_date, series_id, patient_volume,
           mu AS expected, mu - 3 * sd AS lower_bound, mu + 3 * sd AS upper_bound,
           least(1.0, abs(patient_volume - mu) / nullif(4 * sd, 0)) AS anomaly_score,
           abs(patient_volume - mu) > 3 * sd AS is_anomaly,
           'rolling baseline (fallback)' AS method
    FROM w
    WHERE census_date >= date_sub(current_date(), 120);
  END;

  CREATE OR REPLACE TABLE census_anomalies AS
  SELECT a.*, 'ai_detect_anomalies' AS method
  FROM ai_detect_anomalies(
    TABLE(SELECT census_date, concat_ws(' | ', site, unit, shift) AS series_id, patient_volume FROM census_history),
    time_col          => 'census_date',
    value_col         => 'patient_volume',
    group_col         => 'series_id',
    anomaly_threshold => anomaly_threshold,
    holiday_region    => 'US',
    only_last_n       => 120
  ) a;
END;
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Anomalies in the last 120 days — the planted surge, mass-casualty night and ambulance diversion should surface
# MAGIC SELECT census_date,
# MAGIC        split(series_id, ' \\| ')[0] AS site, split(series_id, ' \\| ')[1] AS unit, split(series_id, ' \\| ')[2] AS shift,
# MAGIC        patient_volume AS actual, round(expected, 1) AS expected,
# MAGIC        round(lower_bound, 1) AS lower_bound, round(upper_bound, 1) AS upper_bound,
# MAGIC        round(anomaly_score, 3) AS anomaly_score,
# MAGIC        CASE WHEN patient_volume > upper_bound THEN 'SURGE' ELSE 'DROP' END AS direction,
# MAGIC        method
# MAGIC FROM census_anomalies
# MAGIC WHERE is_anomaly AND (site_filter = 'All' OR split(series_id, ' \\| ')[0] = site_filter)
# MAGIC ORDER BY census_date DESC, anomaly_score DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 💬 Ask Genie — anomalies
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Anomalies in the last 120 days — the planted surge, mass-casualty night and ambulance diversion should surface
# MAGIC SELECT census_date,
# MAGIC        split(series_id, ' \\| ')[0] AS site, split(series_id, ' \\| ')[1] AS unit, split(series_id, ' \\| ')[2] AS shift,
# MAGIC        patient_volume AS actual, round(expected, 1) AS expected,
# MAGIC        round(lower_bound, 1) AS lower_bound, round(upper_bound, 1) AS upper_bound,
# MAGIC        round(anomaly_score, 3) AS anomaly_score,
# MAGIC        CASE WHEN patient_volume > upper_bound THEN 'SURGE' ELSE 'DROP' END AS direction,
# MAGIC        method
# MAGIC FROM census_anomalies
# MAGIC WHERE is_anomaly AND (site_filter = 'All' OR split(series_id, ' \\| ')[0] = site_filter)
# MAGIC ORDER BY census_date DESC, anomaly_score DESC) SELECT `census_date`,MAX(`anomaly_score`) `column_4661261676`,`direction` FROM q GROUP BY `census_date`,`direction`

# COMMAND ----------

ask("Which site-shift-days were flagged as anomalies in the last 60 days, and were they surges or drops?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4 🛠 Try it yourself
# MAGIC Set **`anomaly_threshold`** to `0.90`, re-run **S.2 and S.3**, then re-run **steps 2.1–2.2**. How many more alerts do you get? Pick the sensitivity your operations team would act on.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Predict no-shows — `ai_predict_class`
# MAGIC
# MAGIC `ai_predict_class` **trains a classifier on the rows where the label is known and scores the rows where it's NULL — in one SQL statement.** No feature engineering pipeline, no model registry, no serving endpoint.
# MAGIC
# MAGIC **3.1–3.2 Prove it first (backtest):** hide the outcomes of the last 4 weeks of appointments, let the function predict them from the prior 12 months, and compare against what actually happened.
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → enable **Predictive AI Functions** (Beta) and **AI Forecast V2**. Then re-run the failed cell.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE noshow_backtest AS
# MAGIC SELECT *
# MAGIC FROM ai_predict_class(
# MAGIC   input => TABLE(
# MAGIC     SELECT appointment_id, appt_date, shift, no_show AS actual_no_show,
# MAGIC            CASE WHEN appt_date >= date_sub(current_date(), 28) THEN NULL ELSE no_show END AS no_show,
# MAGIC            payer, visit_type, age_band, patient_age, lead_time_days, prior_no_shows_12m,
# MAGIC            distance_miles, reminder_confirmed, is_telehealth, appt_hour, day_of_week
# MAGIC     FROM appointments
# MAGIC     WHERE appt_date >= date_sub(current_date(), 393) AND appt_date < current_date()
# MAGIC   ),
# MAGIC   target_col   => 'no_show',
# MAGIC   feature_cols => ARRAY('payer', 'visit_type', 'age_band', 'patient_age', 'lead_time_days', 'prior_no_shows_12m',
# MAGIC                         'distance_miles', 'reminder_confirmed', 'is_telehealth', 'appt_hour', 'day_of_week'),
# MAGIC   seed         => 42
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC -- How good is it? Two operational questions:
# MAGIC --   1) If outreach calls the 20% riskiest patients, what share of actual no-shows do they reach?
# MAGIC --   2) Does the predicted number of no-shows per shift match reality? (that's what staffing needs)
# MAGIC WITH p AS (
# MAGIC   SELECT *, CASE WHEN no_show_prediction THEN no_show_confidence ELSE 1 - no_show_confidence END AS p_no_show
# MAGIC   FROM noshow_backtest
# MAGIC ),
# MAGIC ranked AS (SELECT *, percent_rank() OVER (ORDER BY p_no_show DESC) AS risk_pct FROM p),
# MAGIC per_shift AS (
# MAGIC   SELECT appt_date, shift, sum(p_no_show) AS predicted, count_if(actual_no_show) AS actual
# MAGIC   FROM p GROUP BY ALL
# MAGIC )
# MAGIC SELECT
# MAGIC   (SELECT count(*) FROM p)                                                            AS appointments_scored,
# MAGIC   (SELECT round(avg(CAST(actual_no_show AS INT)) * 100, 1) FROM p)                    AS actual_no_show_rate_pct,
# MAGIC   (SELECT round(avg(CAST(no_show_prediction = actual_no_show AS INT)) * 100, 1) FROM p) AS accuracy_pct,
# MAGIC   (SELECT round(count_if(risk_pct <= 0.20 AND actual_no_show) * 100.0 / count_if(actual_no_show), 1) FROM ranked)
# MAGIC                                                                                       AS pct_of_no_shows_in_top_20pct_risk,
# MAGIC   (SELECT round(sum(predicted), 0) FROM per_shift)                                    AS predicted_no_shows,
# MAGIC   (SELECT sum(actual) FROM per_shift)                                                 AS actual_no_shows,
# MAGIC   (SELECT round(avg(abs(predicted - actual)), 1) FROM per_shift)                      AS avg_abs_error_per_shift;

# COMMAND ----------

# MAGIC %md
# MAGIC **3.3–3.4 Score the next 4 weeks.** Train on the last 12 months of outcomes and predict every upcoming appointment. The output feeds two workflows:
# MAGIC * **Patient outreach:** call / text / offer transport or telehealth to the highest-risk patients
# MAGIC * **Capacity:** expected arrivals = booked − predicted no-shows → smarter overbooking and clinician staffing

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE noshow_predictions AS
# MAGIC SELECT *, CASE WHEN no_show_prediction THEN no_show_confidence ELSE 1 - no_show_confidence END AS no_show_probability
# MAGIC FROM ai_predict_class(
# MAGIC   input => TABLE(
# MAGIC     SELECT appointment_id, site, unit, appt_date, shift, no_show,
# MAGIC            payer, visit_type, age_band, patient_age, lead_time_days, prior_no_shows_12m,
# MAGIC            distance_miles, reminder_confirmed, is_telehealth, appt_hour, day_of_week
# MAGIC     FROM appointments
# MAGIC     WHERE appt_date >= date_sub(current_date(), 365)
# MAGIC   ),
# MAGIC   target_col   => 'no_show',
# MAGIC   feature_cols => ARRAY('payer', 'visit_type', 'age_band', 'patient_age', 'lead_time_days', 'prior_no_shows_12m',
# MAGIC                         'distance_miles', 'reminder_confirmed', 'is_telehealth', 'appt_hour', 'day_of_week'),
# MAGIC   seed         => 42
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Outreach worklist: highest-risk appointments in the next 7 days
# MAGIC SELECT appointment_id, appt_date, shift, appt_hour, visit_type, payer, age_band,
# MAGIC        lead_time_days, prior_no_shows_12m, distance_miles, reminder_confirmed, is_telehealth,
# MAGIC        round(no_show_probability * 100, 1) AS no_show_risk_pct,
# MAGIC        CASE WHEN payer = 'Medicaid' AND distance_miles > 12 AND NOT is_telehealth THEN 'Offer transport (NEMT) or convert to telehealth'
# MAGIC             WHEN NOT reminder_confirmed THEN 'Live reminder call'
# MAGIC             WHEN prior_no_shows_12m >= 2 THEN 'Care navigator outreach'
# MAGIC             ELSE 'SMS reminder' END AS suggested_action
# MAGIC FROM noshow_predictions
# MAGIC WHERE appt_date < date_add(current_date(), 7)
# MAGIC ORDER BY no_show_probability DESC
# MAGIC LIMIT 50;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.5 💬 Ask Genie — no-shows
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Outreach worklist: highest-risk appointments in the next 7 days
# MAGIC SELECT appointment_id, appt_date, shift, appt_hour, visit_type, payer, age_band,
# MAGIC        lead_time_days, prior_no_shows_12m, distance_miles, reminder_confirmed, is_telehealth,
# MAGIC        round(no_show_probability * 100, 1) AS no_show_risk_pct,
# MAGIC        CASE WHEN payer = 'Medicaid' AND distance_miles > 12 AND NOT is_telehealth THEN 'Offer transport (NEMT) or convert to telehealth'
# MAGIC             WHEN NOT reminder_confirmed THEN 'Live reminder call'
# MAGIC             WHEN prior_no_shows_12m >= 2 THEN 'Care navigator outreach'
# MAGIC             ELSE 'SMS reminder' END AS suggested_action
# MAGIC FROM noshow_predictions
# MAGIC WHERE appt_date < date_add(current_date(), 7)
# MAGIC ORDER BY no_show_probability DESC
# MAGIC LIMIT 50) SELECT `suggested_action`,COUNT(`appointment_id`) `column_5319713834` FROM q GROUP BY `suggested_action`

# COMMAND ----------

ask("How many no-shows should we expect each day next week?", before=('Step 3 (no-show prediction)', 15))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.6 🛠 Try it yourself
# MAGIC In **step 3.1** (backtest), remove `'reminder_confirmed'` from `feature_cols`, then re-run **steps 3.1 and 3.2**. How much does the top-20% capture rate drop? That's the value of reminder confirmations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Explain *why* no-shows went up — `ai_top_drivers`
# MAGIC
# MAGIC The no-show rate jumped this quarter. Instead of slicing pivot tables for a week, `ai_top_drivers` compares a **test** period (last 90 days) to a **control** period (the 90 days before) and ranks the segments — and **pairs** of segments — that explain the change.
# MAGIC
# MAGIC *Expected finding in the synthetic data:* the increase is concentrated in **Medicaid** patients who live **far from the clinic** and come **in person** — consistent with the transport-vendor change planted in the generator. Note it also surfaces the **pairs** (e.g. `payer=Medicaid` + `visit_mode=in person`) — the interaction a pivot table rarely finds. In real life, this is the insight that sends you to the NEMT contract, not to the scheduling team.
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → enable **Predictive AI Functions** (Beta) and **AI Forecast V2**. Then re-run the failed cell.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT round(avg(CASE WHEN appt_date >= date_sub(current_date(), 90) THEN CAST(no_show AS INT) END) * 100, 1)  AS no_show_rate_last_90d_pct,
# MAGIC        round(avg(CASE WHEN appt_date <  date_sub(current_date(), 90) THEN CAST(no_show AS INT) END) * 100, 1)  AS no_show_rate_prior_90d_pct
# MAGIC FROM appointments
# MAGIC WHERE appt_date >= date_sub(current_date(), 180) AND appt_date < current_date();

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT contributor,
# MAGIC        metric_control AS no_shows_prior_90d,
# MAGIC        metric_test    AS no_shows_last_90d,
# MAGIC        change         AS additional_no_shows,
# MAGIC        round(relative_change * 100, 1) AS pct_change,
# MAGIC        round(support * 100, 1)         AS pct_of_appointments
# MAGIC FROM ai_top_drivers(
# MAGIC   input => TABLE(
# MAGIC     SELECT CAST(no_show AS INT) AS no_shows,
# MAGIC            appt_date >= date_sub(current_date(), 90) AS is_test,
# MAGIC            payer, visit_type, age_band, shift,
# MAGIC            CASE WHEN distance_miles > 12 THEN 'over 12 mi' ELSE '12 mi or less' END AS distance_band,
# MAGIC            CASE WHEN is_telehealth THEN 'telehealth' ELSE 'in person' END            AS visit_mode,
# MAGIC            CASE WHEN reminder_confirmed THEN 'confirmed' ELSE 'not confirmed' END    AS reminder_status
# MAGIC     FROM appointments
# MAGIC     WHERE appt_date >= date_sub(current_date(), 180) AND appt_date < current_date()
# MAGIC   ),
# MAGIC   metric      => 'no_shows',
# MAGIC   is_test     => 'is_test',
# MAGIC   dimensions  => ARRAY('payer', 'visit_type', 'age_band', 'shift', 'distance_band', 'visit_mode', 'reminder_status'),
# MAGIC   aggregation => 'sum'
# MAGIC )
# MAGIC ORDER BY abs(change) DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3 🛠 Try it yourself
# MAGIC In **step 4.2**, check the `dimensions` array. If `'age_band'` isn't already in it, add it (it's already in the input) and re-run — does a new pair of segments appear? If it's already there, try removing it (or `'visit_mode'`) and re-run to see which pairs disappear. `ai_top_drivers` checks every pair for you.

# COMMAND ----------

# MAGIC %sql WITH q AS (SELECT contributor,
# MAGIC        metric_control AS no_shows_prior_90d,
# MAGIC        metric_test    AS no_shows_last_90d,
# MAGIC        change         AS additional_no_shows,
# MAGIC        round(relative_change * 100, 1) AS pct_change,
# MAGIC        round(support * 100, 1)         AS pct_of_appointments
# MAGIC FROM ai_top_drivers(
# MAGIC   input => TABLE(
# MAGIC     SELECT CAST(no_show AS INT) AS no_shows,
# MAGIC            appt_date >= date_sub(current_date(), 90) AS is_test,
# MAGIC            payer, visit_type, age_band, shift,
# MAGIC            CASE WHEN distance_miles > 12 THEN 'over 12 mi' ELSE '12 mi or less' END AS distance_band,
# MAGIC            CASE WHEN is_telehealth THEN 'telehealth' ELSE 'in person' END            AS visit_mode,
# MAGIC            CASE WHEN reminder_confirmed THEN 'confirmed' ELSE 'not confirmed' END    AS reminder_status
# MAGIC     FROM appointments
# MAGIC     WHERE appt_date >= date_sub(current_date(), 180) AND appt_date < current_date()
# MAGIC   ),
# MAGIC   metric      => 'no_shows',
# MAGIC   is_test     => 'is_test',
# MAGIC   dimensions  => ARRAY('payer', 'visit_type', 'age_band', 'shift', 'distance_band', 'visit_mode', 'reminder_status'),
# MAGIC   aggregation => 'sum'
# MAGIC )
# MAGIC ORDER BY abs(change) DESC
# MAGIC LIMIT 15) SELECT `contributor`,SUM(`additional_no_shows`) `column_3978332207` FROM q GROUP BY `contributor`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Flag staffing gaps — putting it all together
# MAGIC
# MAGIC For each site × unit × shift × day in the horizon:
# MAGIC * **Unscheduled care** (ED, inpatient, ICU): required staff = `ai_forecast` volume ÷ staffing ratio; **surge risk** = you'd be 2+ staff short if demand hits the upper 80% prediction bound
# MAGIC * **Scheduled care** (clinic): required clinicians = (booked − `ai_predict_class` expected no-shows) ÷ visits per clinician
# MAGIC * Compare with the **published schedule** and attach an **illustrative cost** — agency / incentive pay to fill a gap vs. idle cost of an overstaffed shift
# MAGIC
# MAGIC > Cost assumptions (edit to match your organization): **$1,450** per gap shift filled with agency / premium pay, **$520** per overstaffed shift.
# MAGIC
# MAGIC > 💡 **Step 5.1 is the longest query in the notebook.** Ask **Genie Code** to explain it: click into the cell, open Genie Code and type `/explain` — or just ask *"what does this cell do?"* in plain English.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE staffing_gaps AS
# MAGIC WITH demand AS (
# MAGIC   -- Unscheduled care: from ai_forecast
# MAGIC   SELECT f.census_date AS shift_date,
# MAGIC          split(f.series_id, ' \\| ')[0] AS site, split(f.series_id, ' \\| ')[1] AS unit, split(f.series_id, ' \\| ')[2] AS shift,
# MAGIC          f.patient_volume_forecast AS expected_patients, f.patient_volume_upper AS high_patients,
# MAGIC          'ai_forecast' AS demand_source
# MAGIC   FROM census_forecast f
# MAGIC   WHERE split(f.series_id, ' \\| ')[1] <> 'Primary & Specialty Care'
# MAGIC   UNION ALL
# MAGIC   -- Scheduled care: booked appointments minus predicted no-shows (ai_predict_class)
# MAGIC   SELECT appt_date, site, unit, shift,
# MAGIC          count(*) - sum(no_show_probability)          AS expected_patients,
# MAGIC          count(*) - sum(no_show_probability) * 0.7    AS high_patients,
# MAGIC          'booked - ai_predict_class no-shows'         AS demand_source
# MAGIC   FROM noshow_predictions
# MAGIC   GROUP BY ALL
# MAGIC ),
# MAGIC j AS (
# MAGIC   SELECT d.*, s.scheduled_staff, u.patients_per_staff, u.min_staff, u.unit_type,
# MAGIC          greatest(u.min_staff, CAST(ceil(d.expected_patients / u.patients_per_staff) AS INT)) AS required_staff,
# MAGIC          greatest(u.min_staff, CAST(ceil(d.high_patients     / u.patients_per_staff) AS INT)) AS surge_staff
# MAGIC   FROM demand d
# MAGIC   JOIN staff_schedule s ON s.shift_date = d.shift_date AND s.site = d.site AND s.unit = d.unit AND s.shift = d.shift
# MAGIC   JOIN dim_care_unit u  ON u.site = d.site AND u.unit = d.unit
# MAGIC )
# MAGIC SELECT *,
# MAGIC   scheduled_staff - required_staff AS staff_gap,
# MAGIC   CASE WHEN scheduled_staff < required_staff     THEN 'UNDERSTAFFED'
# MAGIC        WHEN scheduled_staff <= surge_staff - 2   THEN 'SURGE RISK'   -- 2+ short if demand hits the upper bound
# MAGIC        WHEN scheduled_staff > required_staff + 1 THEN 'OVERSTAFFED'
# MAGIC        ELSE 'OK' END AS status,
# MAGIC   CASE WHEN scheduled_staff < required_staff     THEN (required_staff - scheduled_staff) * 1450
# MAGIC        WHEN scheduled_staff > required_staff + 1 THEN (scheduled_staff - required_staff - 1) * 520
# MAGIC        ELSE 0 END AS est_cost_usd
# MAGIC FROM j;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Staffing summary for the horizon, by site and unit
# MAGIC SELECT site, unit,
# MAGIC        count_if(status = 'UNDERSTAFFED') AS understaffed_shifts,
# MAGIC        count_if(status = 'SURGE RISK')   AS surge_risk_shifts,
# MAGIC        count_if(status = 'OVERSTAFFED')  AS overstaffed_shifts,
# MAGIC        count_if(status = 'OK')           AS ok_shifts,
# MAGIC        sum(CASE WHEN status = 'UNDERSTAFFED' THEN required_staff - scheduled_staff ELSE 0 END) AS staff_shifts_to_fill,
# MAGIC        format_number(sum(CASE WHEN status = 'UNDERSTAFFED' THEN est_cost_usd ELSE 0 END), 0)   AS premium_pay_exposure_usd,
# MAGIC        format_number(sum(CASE WHEN status = 'OVERSTAFFED'  THEN est_cost_usd ELSE 0 END), 0)   AS overstaffing_cost_usd
# MAGIC FROM staffing_gaps
# MAGIC WHERE (site_filter = 'All' OR site = site_filter)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY site, unit;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Scheduled care: capacity freed up by predicted no-shows → overbook or open same-day / walk-in slots
# MAGIC SELECT site, unit, shift,
# MAGIC        count(*) AS clinic_sessions,
# MAGIC        sum(scheduled_staff * patients_per_staff) AS visit_capacity,
# MAGIC        round(sum(expected_patients), 0) AS expected_arrivals,
# MAGIC        round(sum(scheduled_staff * patients_per_staff - expected_patients), 0) AS open_slots_to_backfill
# MAGIC FROM staffing_gaps
# MAGIC WHERE unit_type = 'Clinic' AND (site_filter = 'All' OR site = site_filter)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY shift;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Staffing summary for the horizon, by site and unit
# MAGIC SELECT site, unit,
# MAGIC        count_if(status = 'UNDERSTAFFED') AS understaffed_shifts,
# MAGIC        count_if(status = 'SURGE RISK')   AS surge_risk_shifts,
# MAGIC        count_if(status = 'OVERSTAFFED')  AS overstaffed_shifts,
# MAGIC        count_if(status = 'OK')           AS ok_shifts,
# MAGIC        sum(CASE WHEN status = 'UNDERSTAFFED' THEN required_staff - scheduled_staff ELSE 0 END) AS staff_shifts_to_fill,
# MAGIC        format_number(sum(CASE WHEN status = 'UNDERSTAFFED' THEN est_cost_usd ELSE 0 END), 0)   AS premium_pay_exposure_usd,
# MAGIC        format_number(sum(CASE WHEN status = 'OVERSTAFFED'  THEN est_cost_usd ELSE 0 END), 0)   AS overstaffing_cost_usd
# MAGIC FROM staffing_gaps
# MAGIC WHERE (site_filter = 'All' OR site = site_filter)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY site, unit) SELECT `site`,SUM(`understaffed_shifts`) `column_4098426692`,SUM(`surge_risk_shifts`) `column_1493150308`,SUM(`overstaffed_shifts`) `column_9547458698`,SUM(`ok_shifts`) `column_2675505399` FROM q GROUP BY `site`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Action list for the staffing office — next 7 days, worst gaps first
# MAGIC SELECT shift_date, date_format(shift_date, 'EEE') AS dow, site, unit, shift,
# MAGIC        round(expected_patients, 1) AS expected_patients, required_staff, surge_staff, scheduled_staff,
# MAGIC        staff_gap, status, demand_source,
# MAGIC        CASE status WHEN 'UNDERSTAFFED' THEN 'Post open shift / float pool / incentive pay'
# MAGIC                    WHEN 'SURGE RISK'   THEN 'Place an on-call RN'
# MAGIC                    WHEN 'OVERSTAFFED'  THEN 'Flex staff to a short unit or offer PTO'
# MAGIC                    ELSE '' END AS recommended_action
# MAGIC FROM staffing_gaps
# MAGIC WHERE status <> 'OK' AND shift_date < date_add(current_date(), 7)
# MAGIC   AND (site_filter = 'All' OR site = site_filter)
# MAGIC ORDER BY shift_date, CASE status WHEN 'UNDERSTAFFED' THEN 1 WHEN 'SURGE RISK' THEN 2 ELSE 3 END, staff_gap;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Scheduled care: capacity freed up by predicted no-shows → overbook or open same-day / walk-in slots
# MAGIC SELECT site, unit, shift,
# MAGIC        count(*) AS clinic_sessions,
# MAGIC        sum(scheduled_staff * patients_per_staff) AS visit_capacity,
# MAGIC        round(sum(expected_patients), 0) AS expected_arrivals,
# MAGIC        round(sum(scheduled_staff * patients_per_staff - expected_patients), 0) AS open_slots_to_backfill
# MAGIC FROM staffing_gaps
# MAGIC WHERE unit_type = 'Clinic' AND (site_filter = 'All' OR site = site_filter)
# MAGIC GROUP BY ALL
# MAGIC ORDER BY shift) SELECT `shift`,SUM(`expected_arrivals`) `column_8390594030`,SUM(`open_slots_to_backfill`) `column_1345580213` FROM q GROUP BY `shift`

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.5 💬 Ask Genie — staffing gaps
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Action list for the staffing office — next 7 days, worst gaps first
# MAGIC SELECT shift_date, date_format(shift_date, 'EEE') AS dow, site, unit, shift,
# MAGIC        round(expected_patients, 1) AS expected_patients, required_staff, surge_staff, scheduled_staff,
# MAGIC        staff_gap, status, demand_source,
# MAGIC        CASE status WHEN 'UNDERSTAFFED' THEN 'Post open shift / float pool / incentive pay'
# MAGIC                    WHEN 'SURGE RISK'   THEN 'Place an on-call RN'
# MAGIC                    WHEN 'OVERSTAFFED'  THEN 'Flex staff to a short unit or offer PTO'
# MAGIC                    ELSE '' END AS recommended_action
# MAGIC FROM staffing_gaps
# MAGIC WHERE status <> 'OK' AND shift_date < date_add(current_date(), 7)
# MAGIC   AND (site_filter = 'All' OR site = site_filter)
# MAGIC ORDER BY shift_date, CASE status WHEN 'UNDERSTAFFED' THEN 1 WHEN 'SURGE RISK' THEN 2 ELSE 3 END, staff_gap) SELECT `shift_date`,COUNT(`site`) `column_1237837984`,`status` FROM q GROUP BY `shift_date`,`status`

# COMMAND ----------

ask("Which units are understaffed in the next 7 days, and by how many staff?", before=('Step 5 (staffing gaps)', 34))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.6 🛠 Try it yourself
# MAGIC Replace the illustrative **$1,450** (premium pay) and **$520** (idle shift) in **step 5.1** (`staffing_gaps`) with your own numbers and re-run **steps 5.1–5.5** — the summary, charts and Genie answers update.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. For Payers — Medicaid & Medicare payment prediction
# MAGIC
# MAGIC The same three functions answer the payer-side questions for a health plan's **Medicaid Managed Care, Medicare Advantage, D-SNP and Marketplace** lines:
# MAGIC
# MAGIC | Question | AI Function |
# MAGIC |---|---|
# MAGIC | Which **pending claims** will be paid, denied or pended? (multi-class) | `ai_predict_class` (steps 6.1–6.3) |
# MAGIC | **Why** is our denial rate rising? | `ai_top_drivers` (step 6.4) |
# MAGIC | How much **cash** will go out by line of business over the next N days? | `ai_forecast` (steps 6.5–6.6) |
# MAGIC
# MAGIC > ⚠️ **If an AI Function cell fails** (e.g. *function not found* / *not enabled*), a **workspace admin** needs to turn on the previews: profile menu → **Previews** → enable **Predictive AI Functions** (Beta) and **AI Forecast V2**. Then re-run the failed cell.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 6.1 Predict the outcome of claims still in adjudication (received in the last 14 days)
# MAGIC CREATE OR REPLACE TABLE claim_outcome_predictions AS
# MAGIC SELECT *
# MAGIC FROM ai_predict_class(
# MAGIC   input => TABLE(
# MAGIC     SELECT claim_id, received_date, claim_outcome, line_of_business, member_state, service_category,
# MAGIC            submission_channel, days_to_submit, provider_network, coding_check, prior_auth_status,
# MAGIC            member_months_enrolled, billed_amount
# MAGIC     FROM claims
# MAGIC     WHERE received_date >= date_sub(current_date(), 270)
# MAGIC   ),
# MAGIC   target_col   => 'claim_outcome',
# MAGIC   feature_cols => ARRAY('line_of_business', 'member_state', 'service_category', 'submission_channel', 'days_to_submit',
# MAGIC                         'provider_network', 'coding_check', 'prior_auth_status', 'member_months_enrolled', 'billed_amount'),
# MAGIC   seed         => 42
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Expected outcome mix and dollars at risk for pending claims
# MAGIC SELECT line_of_business, claim_outcome_prediction AS predicted_outcome,
# MAGIC        count(*) AS claims, format_number(sum(billed_amount), 0) AS billed_usd,
# MAGIC        round(avg(claim_outcome_confidence), 3) AS avg_confidence
# MAGIC FROM claim_outcome_predictions
# MAGIC GROUP BY ALL
# MAGIC ORDER BY line_of_business, predicted_outcome;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Pre-adjudication worklist: likely denials — fix before they're denied (missing auth, coding, timely filing)
# MAGIC SELECT claim_id, received_date, line_of_business, member_state, service_category, prior_auth_status, coding_check,
# MAGIC        days_to_submit, provider_network, format_number(billed_amount, 2) AS billed_usd,
# MAGIC        round(claim_outcome_confidence * 100, 1) AS denial_confidence_pct
# MAGIC FROM claim_outcome_predictions
# MAGIC WHERE claim_outcome_prediction = 'Denied'
# MAGIC ORDER BY claim_outcome_confidence DESC, billed_amount DESC
# MAGIC LIMIT 50;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Expected outcome mix and dollars at risk for pending claims
# MAGIC SELECT line_of_business, claim_outcome_prediction AS predicted_outcome,
# MAGIC        count(*) AS claims, format_number(sum(billed_amount), 0) AS billed_usd,
# MAGIC        round(avg(claim_outcome_confidence), 3) AS avg_confidence
# MAGIC FROM claim_outcome_predictions
# MAGIC GROUP BY ALL
# MAGIC ORDER BY line_of_business, predicted_outcome) SELECT `line_of_business`,SUM(`claims`) `column_1340665078`,`predicted_outcome` FROM q GROUP BY `line_of_business`,`predicted_outcome`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 6.4 Why are denials rising? Expected finding: Behavioral Health in Medicaid Managed Care — even with prior auth approved
# MAGIC --     (a medical-necessity policy change), not a coding or submission problem.
# MAGIC --  Last 90 adjudicated days vs the prior 90 days (excludes the 14 days still in adjudication)
# MAGIC SELECT contributor,
# MAGIC        metric_control AS denials_prior_90d, metric_test AS denials_last_90d,
# MAGIC        change AS additional_denials, round(relative_change * 100, 1) AS pct_change, round(support * 100, 1) AS pct_of_claims
# MAGIC FROM ai_top_drivers(
# MAGIC   input => TABLE(
# MAGIC     SELECT CASE WHEN claim_outcome = 'Denied' THEN 1 ELSE 0 END AS denials,
# MAGIC            received_date >= date_sub(current_date(), 104) AS is_test,  -- last 90 adjudicated days
# MAGIC            line_of_business, member_state, service_category, submission_channel, provider_network,
# MAGIC            coding_check, prior_auth_status
# MAGIC     FROM claims
# MAGIC     WHERE received_date BETWEEN date_sub(current_date(), 194) AND date_sub(current_date(), 15)  -- 90 vs 90 days, adjudicated only
# MAGIC   ),
# MAGIC   metric      => 'denials',
# MAGIC   is_test     => 'is_test',
# MAGIC   dimensions  => ARRAY('line_of_business', 'member_state', 'service_category', 'submission_channel',
# MAGIC                        'provider_network', 'coding_check', 'prior_auth_status'),
# MAGIC   aggregation => 'sum'
# MAGIC )
# MAGIC ORDER BY abs(change) DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# MAGIC %sql WITH q AS (-- Pre-adjudication worklist: likely denials — fix before they're denied (missing auth, coding, timely filing)
# MAGIC SELECT claim_id, received_date, line_of_business, member_state, service_category, prior_auth_status, coding_check,
# MAGIC        days_to_submit, provider_network, format_number(billed_amount, 2) AS billed_usd,
# MAGIC        round(claim_outcome_confidence * 100, 1) AS denial_confidence_pct
# MAGIC FROM claim_outcome_predictions
# MAGIC WHERE claim_outcome_prediction = 'Denied'
# MAGIC ORDER BY claim_outcome_confidence DESC, billed_amount DESC
# MAGIC LIMIT 50) SELECT `service_category`,COUNT(`claim_id`) `column_2424008717`,`line_of_business` FROM q GROUP BY `service_category`,`line_of_business`

# COMMAND ----------

# ai_forecast runs on Databricks SQL — this cell sends the SQL below, exactly as written, to your SQL warehouse.
run_on_sql_warehouse(r"""
-- 6.5 Forecast daily paid amount by line of business (cash-flow / reserving input)
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    CREATE OR REPLACE TABLE payment_forecast AS
    SELECT f.*, 'ai_forecast v1 (fallback)' AS model_version
    FROM ai_forecast(
      observed  => TABLE(SELECT paid_date, line_of_business, paid_amount FROM claim_payments
                         WHERE paid_date >= date_sub(current_date(), 365)),
      horizon   => date_add(current_date(), forecast_days),
      time_col  => 'paid_date',
      value_col => 'paid_amount',
      group_col => 'line_of_business'
    ) f;
  END;

  CREATE OR REPLACE TABLE payment_forecast AS
  SELECT f.*, 'ai_forecast v2' AS model_version
  FROM ai_forecast(
    observed       => TABLE(SELECT paid_date, line_of_business, paid_amount FROM claim_payments
                            WHERE paid_date >= date_sub(current_date(), 365)),
    horizon        => date_add(current_date(), forecast_days),
    time_col       => 'paid_date',
    value_col      => 'paid_amount',
    group_col      => 'line_of_business',
    holiday_region => 'US',
    positive_only  => true,
    version        => '2'
  ) f;
END;
""")

# COMMAND ----------

# MAGIC %sql WITH q AS (-- 6b. Why are denials rising? Expected finding: Behavioral Health in Medicaid Managed Care — even with prior auth approved
# MAGIC --     (a medical-necessity policy change), not a coding or submission problem.
# MAGIC --  Last 90 adjudicated days vs the prior 90 days (excludes the 14 days still in adjudication)
# MAGIC SELECT contributor,
# MAGIC        metric_control AS denials_prior_90d, metric_test AS denials_last_90d,
# MAGIC        change AS additional_denials, round(relative_change * 100, 1) AS pct_change, round(support * 100, 1) AS pct_of_claims
# MAGIC FROM ai_top_drivers(
# MAGIC   input => TABLE(
# MAGIC     SELECT CASE WHEN claim_outcome = 'Denied' THEN 1 ELSE 0 END AS denials,
# MAGIC            received_date >= date_sub(current_date(), 104) AS is_test,  -- last 90 adjudicated days
# MAGIC            line_of_business, member_state, service_category, submission_channel, provider_network,
# MAGIC            coding_check, prior_auth_status
# MAGIC     FROM claims
# MAGIC     WHERE received_date BETWEEN date_sub(current_date(), 194) AND date_sub(current_date(), 15)  -- 90 vs 90 days, adjudicated only
# MAGIC   ),
# MAGIC   metric      => 'denials',
# MAGIC   is_test     => 'is_test',
# MAGIC   dimensions  => ARRAY('line_of_business', 'member_state', 'service_category', 'submission_channel',
# MAGIC                        'provider_network', 'coding_check', 'prior_auth_status'),
# MAGIC   aggregation => 'sum'
# MAGIC )
# MAGIC ORDER BY abs(change) DESC
# MAGIC LIMIT 15) SELECT `contributor`,SUM(`additional_denials`) `column_7607198207` FROM q GROUP BY `contributor`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT line_of_business,
# MAGIC        format_number(sum(paid_amount_forecast), 0) AS forecast_paid_usd,
# MAGIC        format_number(sum(greatest(paid_amount_lower, 0)), 0) AS low_usd,
# MAGIC        format_number(sum(paid_amount_upper), 0)    AS high_usd,
# MAGIC        min(paid_date) AS from_date, max(paid_date) AS to_date,
# MAGIC        any_value(model_version) AS model_version
# MAGIC FROM payment_forecast
# MAGIC GROUP BY ALL
# MAGIC ORDER BY sum(paid_amount_forecast) DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.7 💬 Ask Genie — claim denials
# MAGIC Same insight, no SQL. Genie writes and runs the query against the governed tables (about 15 seconds).

# COMMAND ----------

ask("Which pending Medicaid Managed Care claims are most likely to be denied, and why?", before=('Step 6 (denial prediction and drivers)', 35))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.8 🛠 Try it yourself
# MAGIC Pick one of Genie's **Try next** questions from step 6.7 and ask it in the Genie agent (Continue this conversation). Notice it remembers the context.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Executive summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'Provider' AS audience, 'Understaffed shifts in horizon' AS metric,
# MAGIC        CAST(count_if(status = 'UNDERSTAFFED') AS STRING) AS value FROM staffing_gaps
# MAGIC UNION ALL SELECT 'Provider', 'Premium-pay exposure (USD)',
# MAGIC        format_number(sum(CASE WHEN status = 'UNDERSTAFFED' THEN est_cost_usd ELSE 0 END), 0) FROM staffing_gaps
# MAGIC UNION ALL SELECT 'Provider', 'Overstaffing cost (USD)',
# MAGIC        format_number(sum(CASE WHEN status = 'OVERSTAFFED' THEN est_cost_usd ELSE 0 END), 0) FROM staffing_gaps
# MAGIC UNION ALL SELECT 'Provider', 'Anomalous site-shift-days (last 120d)',
# MAGIC        CAST(count_if(is_anomaly) AS STRING) FROM census_anomalies
# MAGIC UNION ALL SELECT 'Provider', 'Expected no-shows, next 4 weeks',
# MAGIC        format_number(sum(no_show_probability), 0) FROM noshow_predictions
# MAGIC UNION ALL SELECT 'Provider', 'High-risk appointments (>40% risk) to target',
# MAGIC        CAST(count_if(no_show_probability > 0.40) AS STRING) FROM noshow_predictions
# MAGIC UNION ALL SELECT 'Payer', 'Pending claims predicted to deny',
# MAGIC        CAST(count_if(claim_outcome_prediction = 'Denied') AS STRING) FROM claim_outcome_predictions
# MAGIC UNION ALL SELECT 'Payer', 'Billed USD on predicted denials',
# MAGIC        format_number(sum(CASE WHEN claim_outcome_prediction = 'Denied' THEN billed_amount ELSE 0 END), 0) FROM claim_outcome_predictions
# MAGIC UNION ALL SELECT 'Payer', 'Forecast paid amount, horizon (USD)',
# MAGIC        format_number(sum(paid_amount_forecast), 0) FROM payment_forecast;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Your turn — ask anything, then build more
# MAGIC **Ask your own question:** pick one in the **`question`** widget — or type anything in **`custom_question`** — and re-run **step 8.1**. Click **Continue this conversation in Genie** to drill down with follow-ups and charts.

# COMMAND ----------

question = dbutils.widgets.get("custom_question").strip() or dbutils.widgets.get("question")
ask(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.2 🚀 Build more — ideas to take this further
# MAGIC | Idea | How | Effort |
# MAGIC |---|---|---|
# MAGIC | **Add a covariate** to the census forecast (scheduled electives, flu-positivity rate, weather) | `covariate_col =>` in `ai_forecast` (v2) — see step 1.1 | 15 min |
# MAGIC | **Predict length of stay** or **days-to-pay** as a number | `ai_predict_value` — same pattern as `ai_predict_class` | 15 min |
# MAGIC | **Explain a forecast miss** — why was last week busier than forecast? | `ai_top_drivers` with `is_test` = last week | 15 min |
# MAGIC | **Summarize the staffing action list** into a shift-huddle note | `ai_query` / `ai_gen` over `staffing_gaps` | 15 min |
# MAGIC | **Daily refresh** before the 6 a.m. huddle | Schedule this notebook as a **Lakeflow Job** (Schedule button, top right) | 5 min |
# MAGIC | **Huddle dashboard** | **AI/BI dashboard** on `staffing_gaps` and `noshow_predictions`, with Genie built in | 30 min |
# MAGIC | **Share Genie** with the business | Open the Genie agent → **Share** → add your charge-nurse / revenue-cycle groups | 2 min |
# MAGIC | **Your real data** | Point the views in *Bring your own data* (step 0.12) at your ADT, scheduling and claims tables | 1 day |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. What you just saw — and where to take it next
# MAGIC
# MAGIC * **Seven AI-powered analyses, ~630 lines of SQL, zero ML infrastructure.** Every model trains and scores inside the query, governed by Unity Catalog.
# MAGIC * **Genie turns those results into self-service.** Charge nurses, staffing coordinators and revenue-cycle analysts ask questions in plain English; Genie writes the SQL against the same governed tables.
# MAGIC * **Same pattern, provider and payer.** Census & staffing for health systems; denials & cash flow for Medicaid / Medicare plans.
# MAGIC
# MAGIC **Production path**
# MAGIC 1. **Schedule it** — run this notebook as a Lakeflow Job each morning before the staffing huddle, so Genie always answers on fresh forecasts.
# MAGIC 2. **Land real data** — ADT, scheduling and workforce feeds via Lakeflow Connect / partner connectors into Unity Catalog; swap the views in *Bring your own data* (step 0.12).
# MAGIC 3. **Share the Genie agent** with the business, and add an AI/BI dashboard on `staffing_gaps` and `noshow_predictions` for the daily huddle.
# MAGIC 4. **Graduate where needed** — persisted models, custom features or real-time scoring with Databricks ML (feature store, MLflow, Model Serving).
# MAGIC
# MAGIC **Docs:** [ai_forecast](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_forecast) · [ai_detect_anomalies](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_detect_anomalies) · [ai_predict_class](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_predict_class) · [ai_top_drivers](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_top_drivers) · [AI Functions overview](https://docs.databricks.com/aws/en/large-language-models/ai-functions) · [Genie](https://docs.databricks.com/aws/en/genie/)
# MAGIC
# MAGIC <sub>Predictive AI Functions (`ai_predict_class`, `ai_detect_anomalies`, `ai_top_drivers`) are in Beta. Review your organization's compliance requirements (e.g. HIPAA, compliance security profile) before using them with PHI. All data in this notebook is synthetic.</sub>
