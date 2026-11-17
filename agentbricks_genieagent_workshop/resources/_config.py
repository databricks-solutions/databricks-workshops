# Databricks notebook source
# MAGIC %md
# MAGIC # Shared configuration
# MAGIC Every notebook runs this with `%run ./resources/_config`. It works out *your* names from your login. You don't need to edit anything here.
# MAGIC
# MAGIC In your own workspace (for example a free trial) everything goes into one schema, `workspace.agentbricks_genieagent_workshop_lab`. In the presenters' workshop workspace each person gets their own schema in `kasterwell_labs`.

# COMMAND ----------

import hashlib
import re

ME = spark.sql("SELECT current_user()").first()[0]
_local = re.sub(r"[^a-z0-9]+", "-", ME.split("@")[0].lower()).strip("-")
_hash = hashlib.sha1(ME.encode()).hexdigest()[:4]

try:  # presenters only: a fixed schema instead of your own (for the presenters' shared backup agent)
    _KIT_SCHEMA = dbutils.widgets.get("kit_schema")
except Exception:
    _KIT_SCHEMA = ""

_catalogs = {r[0] for r in spark.sql("SHOW CATALOGS").collect()}
WORKSHOP_S = "kasterwell_labs.agent_bricks_workshop"     # shared backup copy behind the workshop's backup agent
if "kasterwell_labs" in _catalogs:          # presenters' workshop workspace
    CATALOG, SCHEMA = "kasterwell_labs", _KIT_SCHEMA or f"{_local.replace('-', '_')}_{_hash}"
else:                                       # your own workspace
    CATALOG, SCHEMA = "workspace", "agentbricks_genieagent_workshop_lab"   # named after the workshop (databricks-workshops naming rule)
S = f"{CATALOG}.{SCHEMA}"

_slug = _local if _local[:1].isalpha() else f"u{_local}"   # memory/session store names must start with a letter
if _KIT_SCHEMA:                             # shared copies: agent_bricks_workshop -> app agent-bricks-workshop-kit
    _base = _KIT_SCHEMA.removeprefix("agent_bricks_").replace("_", "-")
    AGENT_NAME, _STORE = f"{_base}-kit", f"agent-bricks-{_base}"
    TRACE_EXPERIMENT = f"/Workspace/Kasterwell-Workshop/agents/{_STORE}"
else:
    AGENT_NAME = f"{_slug[:11].strip('-')}-{_hash}"   # deployed app = agent-bricks-<AGENT_NAME> (Apps names max 30 chars)
    _STORE = AGENT_NAME
    TRACE_EXPERIMENT = f"/Users/{ME}/agentbricks_traces/{AGENT_NAME}"
APP_NAME = f"agent-bricks-{AGENT_NAME}"
MEMORY_STORE = f"{_STORE}-memory"
SESSION_STORE = f"{_STORE}-sessions"

KIT_DIR = "/Workspace" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 1)[0]
RESOURCES = f"{KIT_DIR}/resources"   # helpers, data and screenshots; the three notebooks sit one level up
import sys
if RESOURCES not in sys.path:
    sys.path.insert(0, RESOURCES)  # so the notebooks can `import genie_config`
GENIE_TABLES = ["dim_clinic", "tenant_health_weekly", "claims_monthly", "appointment_access", "revenue_monthly"]
PERSONAS = ["billing_analyst", "care_coordinator_west", "clinic_admin_vellamy", "compliance_officer"]


def set_persona(persona: str):
    """Switch which persona you act as. Row filters on every clinic table follow it immediately."""
    assert persona in PERSONAS, f"choose one of {PERSONAS}"
    spark.sql(f"INSERT INTO {S}.persona_assignments VALUES (sha2(current_user(), 256), '{persona}', current_timestamp())")
    if CATALOG == "kasterwell_labs" and S != WORKSHOP_S:  # presenters: keep the shared backup agent on the same persona
        try:
            spark.sql(f"INSERT INTO {WORKSHOP_S}.persona_assignments VALUES (sha2(current_user(), 256), '{persona}', current_timestamp())")
        except Exception:
            pass
    return spark.sql(f"SELECT persona, display_name FROM {S}.my_persona").first()


def setting(key: str, value: str | None = None):
    """Read (or save) a value shared between notebooks, such as your Genie Agent's id."""
    if value is not None:
        spark.sql(f"DELETE FROM {S}.kit_settings WHERE key = '{key}'")
        spark.sql(f"INSERT INTO {S}.kit_settings VALUES ('{key}', '{value}')")
    row = spark.sql(f"SELECT value FROM {S}.kit_settings WHERE key = '{key}'").first()
    return row[0] if row else None


print(f"You are {ME}\n  schema:      {S}\n  agent name:  {AGENT_NAME}  (app {APP_NAME})")
