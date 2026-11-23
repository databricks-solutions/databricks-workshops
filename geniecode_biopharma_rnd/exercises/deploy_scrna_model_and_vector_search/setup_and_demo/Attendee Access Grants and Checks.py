# Databricks notebook source
# DBTITLE 1,Overview and run order
# MAGIC %md
# MAGIC # Attendee access: grants and checks
# MAGIC
# MAGIC Sets up and verifies what attendees need for the hands-on (`../[Attendees] PROMPT_handson.md`).
# MAGIC Attendees **read** shared objects in `main.owm` and **write** only to their own `dbacademy.<me>` schema
# MAGIC (created by the lab, owned by them) and their home folder.
# MAGIC
# MAGIC Run in order:
# MAGIC 1. **Grant read access, revoke shared writes** - SELECT / EXECUTE grants on shared objects; revokes CREATE TABLE on `main.owm` and WRITE VOLUME / APPLY TAG on `teddy_files`.
# MAGIC 2. **Check: attendee grants** - read grants present, shared writes absent, endpoint CAN_QUERY and VS CAN_USE.
# MAGIC 3. **Check: attendee prompt** - the Shared prompt uses `dbacademy.<me>` write paths, its answer-key images resolve in `setup_and_demo/imgs/`, and it matches the master copy.
# MAGIC
# MAGIC Step 1 only needs re-running if Step 2 fails. Steps 2 and 3 should each end with a green "all checks pass" line before the session.
# MAGIC
# MAGIC > **Note:** everything under `/Shared` is editable by all users (CAN_MANAGE inherited from `/Shared`, which can't be lowered). Re-run Step 3 before and during the session; if it fails, restore the prompt from the master copy in the instructor's home folder.

# COMMAND ----------

# DBTITLE 1,1. Grant read access, revoke shared writes
# MAGIC %sql
# MAGIC -- Step 1 of 3. Attendees READ from main.owm and WRITE only to their own dbacademy.<me> schema (see ../[Attendees] PROMPT_handson.md).
# MAGIC -- H2 needs SELECT on the VS index backing table AND on the VS index itself (to query it)
# MAGIC GRANT SELECT ON TABLE main.owm.teddy_ref_cells TO `account users`;
# MAGIC GRANT SELECT ON TABLE main.owm.teddy_ref_index TO `account users`;
# MAGIC -- Optional, read-only: lets attendees see versions / load the shared model (not needed to query the endpoint).
# MAGIC -- Registered models use the FUNCTION keyword in SQL. Do NOT grant on teddy_model_demo (instructor only).
# MAGIC GRANT EXECUTE ON FUNCTION main.owm.teddy_model_shared TO `account users`;
# MAGIC -- READ VOLUME on main.owm.teddy_files stays (shared files are read in H1).
# MAGIC
# MAGIC -- Remove shared write access: attendees no longer write to main.owm or teddy_files.
# MAGIC -- Existing tables attendees created stay (they own them); this only stops new writes.
# MAGIC REVOKE CREATE TABLE ON SCHEMA main.owm FROM `account users`;
# MAGIC REVOKE WRITE VOLUME ON VOLUME main.owm.teddy_files FROM `account users`;
# MAGIC REVOKE APPLY TAG ON VOLUME main.owm.teddy_files FROM `account users`;

# COMMAND ----------

# DBTITLE 1,2. Check: attendee grants (reads present, shared writes absent)
# Step 2 of 3. Verify the grants the hands-on exercise needs (reads present) and that shared writes are gone.
# REQUIRED: PASS = grant present. MUST_BE_ABSENT: PASS = grant absent on the shared main.owm objects.
# Attendees write to their own dbacademy.<me> schema, which they own (no grants needed there).

from pyspark.sql import functions as F

REQUIRED = [
    # (description, SQL to check, expected ActionType, object level)
    ("USE CATALOG  on main",
     "SHOW GRANTS `account users` ON CATALOG main",
     "USE CATALOG"),
    ("USE SCHEMA   on main.owm",
     "SHOW GRANTS `account users` ON SCHEMA main.owm",
     "USE SCHEMA"),
    ("USE CATALOG  on dbacademy (own schemas)",
     "SHOW GRANTS `account users` ON CATALOG dbacademy",
     "USE CATALOG"),
    ("SELECT       on main.owm.teddy_ref_cells",
     "SHOW GRANTS `account users` ON TABLE main.owm.teddy_ref_cells",
     "SELECT"),
    ("SELECT       on main.owm.teddy_ref_index (VS index)",
     "SHOW GRANTS `account users` ON TABLE main.owm.teddy_ref_index",
     "SELECT"),
    ("READ VOLUME  on main.owm.teddy_files",
     "SHOW GRANTS `account users` ON VOLUME main.owm.teddy_files",
     "READ VOLUME"),
    ("EXECUTE      on main.owm.teddy_model_shared (model)",
     "SHOW GRANTS `account users` ON FUNCTION main.owm.teddy_model_shared",
     "EXECUTE"),
]

MUST_BE_ABSENT = [
    ("no CREATE TABLE on main.owm",
     "SHOW GRANTS `account users` ON SCHEMA main.owm",
     "CREATE TABLE"),
    ("no WRITE VOLUME on main.owm.teddy_files",
     "SHOW GRANTS `account users` ON VOLUME main.owm.teddy_files",
     "WRITE VOLUME"),
    ("no APPLY TAG on main.owm.teddy_files",
     "SHOW GRANTS `account users` ON VOLUME main.owm.teddy_files",
     "APPLY TAG"),
]

# Workspace-level permissions (serving endpoint, VS endpoint)
from databricks.sdk import WorkspaceClient
import requests

w = WorkspaceClient()
results = []

# ── Unity Catalog grants ────────────────────────────────────────────
for desc, sql, expected_action in REQUIRED:
    df = spark.sql(sql)
    found = df.filter(F.col("ActionType") == expected_action).count() > 0
    results.append((desc, "UC grant", "PASS ✅" if found else "FAIL ❌"))

for desc, sql, action in MUST_BE_ABSENT:
    df = spark.sql(sql)
    found = df.filter(F.col("ActionType") == action).count() > 0
    results.append((desc, "UC no-write", "FAIL ❌" if found else "PASS ✅"))

# ── Serving endpoint: teddy_ep_shared  (CAN_QUERY for users) ───────
try:
    ep = w.serving_endpoints.get("teddy_ep_shared")
    ep_perms = w.api_client.do(
        "GET", f"/api/2.0/permissions/serving-endpoints/{ep.id}"
    )
    acl = ep_perms.get("access_control_list", [])
    has_query = any(
        entry.get("group_name") == "users"
        and any(
            p.get("permission_level") in ("CAN_QUERY", "CAN_MANAGE")
            for p in entry.get("all_permissions", [])
        )
        for entry in acl
    )
    results.append(("CAN_QUERY    on teddy_ep_shared", "Endpoint", "PASS ✅" if has_query else "FAIL ❌"))
except Exception as e:
    results.append(("CAN_QUERY    on teddy_ep_shared", "Endpoint", f"ERROR: {e}"))

# ── VS endpoint: teddy_vs_workshop  (CAN_USE for users) ────────────
try:
    vs_perms = w.api_client.do(
        "GET", f"/api/2.0/permissions/vector-search-endpoints/{w.vector_search_endpoints.get_endpoint('teddy_vs_workshop').id}"
    )
    acl = vs_perms.get("access_control_list", [])
    has_use = any(
        entry.get("group_name") == "users"
        and any(
            p.get("permission_level") in ("CAN_USE", "CAN_MANAGE")
            for p in entry.get("all_permissions", [])
        )
        for entry in acl
    )
    results.append(("CAN_USE      on teddy_vs_workshop", "VS endpoint", "PASS ✅" if has_use else "FAIL ❌"))
except Exception as e:
    results.append(("CAN_USE      on teddy_vs_workshop", "VS endpoint", f"ERROR: {e}"))

# ── Summary ─────────────────────────────────────────────────────────
df_results = spark.createDataFrame(results, ["Permission", "Scope", "Status"])
display(df_results)

fails = [r for r in results if "FAIL" in r[2]]
if fails:
    print(f"\n⚠️  {len(fails)} check(s) failed (missing read grant or leftover shared write) — run the cell above first.")
else:
    print(f"\n✅ All {len(results)} checks pass: reads in place, no shared writes. Attendees are ready.")

# COMMAND ----------

# DBTITLE 1,3. Check: attendee prompt uses dbacademy write paths
# Step 3 of 3. Check that the attendee prompt matches these grants: reads from main.owm, writes only to dbacademy.<me>,
# notebooks in the home folder. Also checks the Shared copy is identical to the master copy.
import hashlib, os, re

SHARED_PROMPT = "/Workspace/Shared/exercises/deploy_scrna_model_and_vector_search/[Attendees] PROMPT_handson.md"
# Master copy: this repository's Git folder in your home (Open in Databricks clones it there). If it isn't
# there, the comparison below is skipped.
_ME = spark.sql("SELECT current_user()").first()[0]
MASTER_PROMPT = (f"/Workspace/Users/{_ME}/databricks-solutions-databricks-workshops/geniecode_biopharma_rnd/exercises/"
                 "deploy_scrna_model_and_vector_search/[Attendees] PROMPT_handson.md")

MUST_CONTAIN = [
    "schema `dbacademy.<me>`",
    "`dbacademy.<me>.teddy_tutorial_cells`",
    "`dbacademy.<me>.teddy_handson`",
    "never write to `main.owm` or the shared",
    "`/Users/<my full user name>/teddy_handson/`",
    "Spark's current user and the SDK's",
]
MUST_NOT_CONTAIN = [          # old shared-write wording
    "teddy_tutorial_cells_<me>",
    "shared/<me>/",
    "`runs/` folder next to this prompt",
    "Everyone shares one schema and volume",
]

prompt_results = []
if not os.path.exists(SHARED_PROMPT):
    prompt_results.append(("Shared prompt file exists", "FAIL ❌ (not found: " + SHARED_PROMPT + ")"))
else:
    text = open(SHARED_PROMPT).read()
    for s in MUST_CONTAIN:
        prompt_results.append((f"has: {s}", "PASS ✅" if s in text else "FAIL ❌"))
    for s in MUST_NOT_CONTAIN:
        prompt_results.append((f"no old text: {s}", "FAIL ❌" if s in text else "PASS ✅"))
    # Answer-key images: each ![](...) must resolve relative to the prompt and live in setup_and_demo/imgs/ (not runs/).
    prompt_dir = os.path.dirname(SHARED_PROMPT)
    img_paths = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text)
    prompt_results.append(("has answer-key images", f"PASS ✅ ({len(img_paths)})" if img_paths else "FAIL ❌ (none found)"))
    for p in img_paths:
        exists = os.path.exists(os.path.normpath(os.path.join(prompt_dir, p)))
        in_imgs = p.startswith("./setup_and_demo/imgs/")
        prompt_results.append((f"image ok: {p}",
                               "PASS ✅" if exists and in_imgs
                               else "FAIL ❌ (" + ("missing" if not exists else "not in setup_and_demo/imgs/") + ")"))
    if os.path.exists(MASTER_PROMPT):
        same = hashlib.md5(text.encode()).hexdigest() == hashlib.md5(open(MASTER_PROMPT).read().encode()).hexdigest()
        prompt_results.append(("Shared copy identical to master", "PASS ✅" if same else "FAIL ❌ (out of sync)"))
    else:
        prompt_results.append(("Shared copy identical to master", "SKIP (master not readable)"))

display(spark.createDataFrame(prompt_results, ["Prompt check", "Status"]))
fails = [r for r in prompt_results if "FAIL" in r[1]]
print(f"\n⚠️  {len(fails)} prompt check(s) failed." if fails
      else f"\n✅ All {len(prompt_results)} prompt checks pass: the attendee prompt uses dbacademy.<me> writes.")