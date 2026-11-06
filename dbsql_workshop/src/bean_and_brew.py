"""Helpers for the Bean & Brew workshop notebook: connect to the bundle's deployment and create the workspace assets.

The bundle stores its settings as properties on the bronze schema it creates (see resources/dbsql_workshop.schemas.yml),
so the notebook needs no widgets or lookups: it finds the current user's schemas and reads the settings from there.
"""
import base64
import html
import json
import os
import re
import urllib.parse

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
PROPERTY = "dbsql_workshop."
SETTINGS = ("silver_schema", "gold_schema", "warehouse_id", "llm_endpoint", "alert_email", "asset_prefix", "assets_folder")


def connect(spark):
    """Find the current user's Bean & Brew schemas, USE their catalog, declare the schema session variables and
    return the bundle's settings."""
    from databricks.sdk import WorkspaceClient

    # Same naming as the bundle: dbsql_workshop_<user name before the @, non-alphanumerics as _>_bronze
    prefix = spark.sql(r"SELECT 'dbsql_workshop_' || regexp_replace(lower(split_part(current_user(), '@', 1)), '[^\\p{L}\\p{N}]', '_') || '_'").first()[0]
    bronze = prefix + "bronze"
    catalogs = [r[0] for r in spark.sql("SELECT catalog_name FROM system.information_schema.schemata WHERE schema_name = :s",
                                        args={"s": bronze}).collect()]
    if not catalogs:
        raise RuntimeError(f"No schema named {bronze} was found. Deploy the dbsql_workshop bundle first.")
    if len(catalogs) > 1:
        raise RuntimeError(f"{bronze} exists in {len(catalogs)} catalogs ({', '.join(catalogs)}). Destroy the extra deployment.")
    catalog = catalogs[0]

    props = WorkspaceClient().schemas.get(f"{catalog}.{bronze}").properties or {}
    missing = [k for k in SETTINGS if not props.get(PROPERTY + k)]
    if missing:
        raise RuntimeError(f"{catalog}.{bronze} is missing the bundle settings {missing}. Redeploy the dbsql_workshop bundle.")
    cfg = {"catalog": catalog, "bronze_schema": bronze, **{k: props[PROPERTY + k] for k in SETTINGS}}
    cfg["asset_prefix"] += " "  # stored without its trailing space

    for k in ("catalog", "bronze_schema", "silver_schema", "gold_schema"):
        if not re.fullmatch(r"[A-Za-z0-9_]+", cfg[k]):
            raise ValueError(f"{k} must contain only letters, digits and underscores: {cfg[k]!r}")
    spark.sql(f"USE CATALOG `{catalog}`")
    for k in ("bronze_schema", "silver_schema", "gold_schema"):
        spark.sql(f"DECLARE OR REPLACE VARIABLE {k} STRING DEFAULT '{cfg[k]}'")
    print(f"Catalog {catalog}: {cfg['bronze_schema']}, {cfg['silver_schema']}, {cfg['gold_schema']}\n"
          f"SQL warehouse {cfg['warehouse_id']}, assets folder {cfg['assets_folder']}")
    return cfg


def fill(text, cfg):
    """Replace the __CATALOG__/__BRONZE__/__SILVER__/__GOLD__/__LLM_ENDPOINT__ placeholders."""
    for placeholder, key in (("__CATALOG__", "catalog"), ("__BRONZE__", "bronze_schema"), ("__SILVER__", "silver_schema"),
                             ("__GOLD__", "gold_schema"), ("__LLM_ENDPOINT__", "llm_endpoint")):
        if placeholder in text:
            text = text.replace(placeholder, cfg[key])
    return text


class WorkshopAssets:
    """Creates the workshop's workspace assets inside the bundle's own folder, so `databricks bundle destroy` deletes them.

    Only objects created here (their IDs are kept in asset_ids.json in that folder) are ever updated. Nothing is looked
    up or adopted by name, so an existing object with the same name is never touched.
    """

    def __init__(self, cfg):
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        self.cfg, self.api, self.host = cfg, w.api_client.do, w.config.host.rstrip("/")
        with open(os.path.join(ASSETS, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.prefix, self.warehouse_id = cfg["asset_prefix"], cfg["warehouse_id"]
        self.dir = cfg["assets_folder"].removeprefix("/Workspace")
        self.ws_dir = "/Workspace" + self.dir
        self.state_path = f"{self.dir}/asset_ids.json"
        self.api("POST", "/api/2.0/workspace/mkdirs", body={"path": self.dir})
        try:
            self.state = json.loads(base64.b64decode(
                self.api("GET", "/api/2.0/workspace/export", query={"path": self.state_path, "format": "AUTO"})["content"]))
        except Exception:
            self.state = {}
        print(f"Assets folder: {self.url('/browse/folders/' + str(self.object_id(self.dir)))}  ({self.ws_dir})")

    # -- helpers --
    def url(self, path):
        return self.host + path

    def get(self, path):
        try:
            return self.api("GET", path)
        except Exception:
            return None

    def status(self, path):
        return self.get("/api/2.0/workspace/get-status?" + urllib.parse.urlencode({"path": path}))

    def object_id(self, path):
        s = self.status(path)
        return s.get("object_id") if s else None

    @staticmethod
    def active(obj):
        return bool(obj) and obj.get("lifecycle_state", "ACTIVE") == "ACTIVE"

    def template(self, rel):
        with open(os.path.join(ASSETS, rel)) as f:
            return fill(f.read(), self.cfg)

    def save(self):  # written after every asset, so a failed run can be re-run without creating duplicates
        content = base64.b64encode(json.dumps(self.state, indent=2).encode()).decode()
        self.api("POST", "/api/2.0/workspace/import", body={"path": self.state_path, "format": "AUTO", "overwrite": True, "content": content})

    def upsert(self, key, get_path, create, update):
        """Update the object this notebook created for `key`, or create it if there is none (or it was deleted)."""
        oid = self.state.get(key)
        if oid and self.active(self.get(get_path.format(oid))):
            update(oid)
        else:
            oid = create()
        self.state[key] = oid
        self.save()
        return oid

    # -- assets --
    def saved_queries(self):
        """The SQL editor script and saved queries 01-09."""
        def query(key, name, text, description=""):
            body = {"display_name": self.prefix + name, "query_text": text, "warehouse_id": self.warehouse_id, "description": description}
            return self.upsert(
                key, "/api/2.0/sql/queries/{}",
                create=lambda: self.api("POST", "/api/2.0/sql/queries", body={"query": {**body, "parent_path": self.ws_dir}})["id"],
                update=lambda qid: self.api("PATCH", f"/api/2.0/sql/queries/{qid}",
                                            body={"update_mask": "display_name,query_text,warehouse_id,description", "query": body}))

        def statements(parts):
            return ";\n\n".join(self.template(f"queries/{p}.sql").strip() for p in parts)

        s = self.manifest["script"]
        query(s["key"], s["name"], statements(s["parts"]) + ";", s["description"])
        for q in self.manifest["saved_queries"]:
            query(q["key"], q["name"], statements(q["parts"]))
        print(f"SQL editor script: {self.url('/sql/editor/' + self.state[s['key']])}\n"
              f"{len(self.manifest['saved_queries'])} saved queries (ai_query endpoint {self.cfg['llm_endpoint']})")

    def flow(self):
        """The visual data prep flow, pointed at the bundle's catalog and schemas."""
        flow = self.manifest["flow"]
        path = f"{self.dir}/{self.prefix}{flow['name']}"
        exists = self.status(path) is not None
        if exists and self.state.get("flow") != path:
            raise RuntimeError(f"{path} already exists and was not created by this notebook, so it is left untouched.")
        content = base64.b64encode(self.template(flow["file"]).encode()).decode()
        self.api("POST", "/api/2.0/workspace/import", body={"path": path, "format": "AUTO", "overwrite": exists, "content": content})
        self.state["flow"] = path
        self.save()
        print(f"Visual data prep flow: {self.url('/editor/designer-files/' + str(self.object_id(path)))}")

    def dashboard(self):
        """The AI/BI dashboard: created or updated, published with embedded credentials, plus a paused daily refresh."""
        d = self.manifest["dashboard"]
        body = {"display_name": self.prefix + d["name"], "warehouse_id": self.warehouse_id, "serialized_dashboard": self.template(d["file"])}
        did = self.upsert(
            d["key"], "/api/2.0/lakeview/dashboards/{}",
            create=lambda: self.api("POST", "/api/2.0/lakeview/dashboards", body={**body, "parent_path": self.dir})["dashboard_id"],
            update=lambda i: self.api("PATCH", f"/api/2.0/lakeview/dashboards/{i}", body=body))
        self.api("POST", f"/api/2.0/lakeview/dashboards/{did}/published", body={"embed_credentials": True, "warehouse_id": self.warehouse_id})
        if not self.api("GET", f"/api/2.0/lakeview/dashboards/{did}/schedules").get("schedules"):
            self.api("POST", f"/api/2.0/lakeview/dashboards/{did}/schedules", body={
                "display_name": "Daily refresh at 01:00", "pause_status": "PAUSED",
                "cron_schedule": {"quartz_cron_expression": "0 0 1 * * ?", "timezone_id": "America/New_York"}})
        print(f"Dashboard (published): {self.url(f'/dashboardsv3/{did}/published')}")

    def genie(self):
        """The Genie agent: instructions, examples, snippets, joins, sample questions and benchmarks in one serialized space."""
        g = self.manifest["genie"]
        body = {"title": self.prefix + g["title"], "description": g["description"], "warehouse_id": self.warehouse_id,
                "serialized_space": self.template(g["file"])}
        gid = self.upsert(
            g["key"], "/api/2.0/genie/spaces/{}",
            create=lambda: self.api("POST", "/api/2.0/genie/spaces", body={**body, "parent_path": self.ws_dir})["space_id"],
            update=lambda i: self.api("PATCH", f"/api/2.0/genie/spaces/{i}", body=body))
        print(f"Genie agent: {self.url(f'/genie/rooms/{gid}')}")

    def alerts(self):
        """The three SQL alerts: weekly, Monday 8 AM New York, created paused."""
        for a in json.loads(self.template(self.manifest["alerts"]["file"])):
            key = a.pop("key")  # alert_inventory, alert_traffic, alert_labor
            a["display_name"] = self.prefix + a["display_name"]
            a["warehouse_id"] = self.warehouse_id
            a["evaluation"]["notification"]["subscriptions"] = [{"user_email": self.cfg["alert_email"]}]
            a["schedule"]["pause_status"] = "PAUSED"
            aid = self.upsert(
                key, "/api/2.0/alerts/{}",
                create=lambda: self.api("POST", "/api/2.0/alerts", body={**a, "parent_path": self.ws_dir})["id"],
                update=lambda i: self.api("PATCH", f"/api/2.0/alerts/{i}",
                                          query={"update_mask": "display_name,query_text,warehouse_id,evaluation,schedule"}, body=a))
            print(f"Alert {a['display_name']}: {self.url(f'/sql/alerts-v2/{aid}')}")

    def index(self):
        """HTML table with live links to every asset."""
        def link(path, text):
            return f'<a href="{path}" target="_blank">{html.escape(text)}</a>' if path else html.escape(text)

        c, st, m, p = self.cfg, self.state, self.manifest, self.prefix
        rows = [(link(f"/browse/folders/{self.object_id(self.dir)}", self.dir.rsplit('/', 1)[-1] + " (bundle folder)"), "Folder", "Every asset below that this notebook created")]
        rows.append((link(f"/sql/warehouses/{c['warehouse_id']}", "SQL warehouse"), "Warehouse", "Serverless 2X-Small, stops itself when idle"))
        rows += [(link(f"/explore/data/{c['catalog']}/{c[k]}", f"{c['catalog']}.{c[k]}"), "Schema", d) for k, d in
                 [("bronze_schema", "Raw source data, defects included"), ("silver_schema", "Cleaned dimensions and facts with keys and tags"),
                  ("gold_schema", "Business tables and the 3 metric views")]]
        s = m["script"]
        if s["key"] in st:
            rows.append((link(f"/sql/editor/{st[s['key']]}", p + s["name"]), "Query", "SQL editor demo script"))
            rows += [(link(f"/sql/editor/{st[q['key']]}", p + q["name"]), "Query", "Saved query, one feature each") for q in m["saved_queries"] if q["key"] in st]
        if "flow" in st:
            rows.append((link(f"/editor/designer-files/{self.object_id(st['flow'])}", p + m["flow"]["name"].removesuffix(".designer.ipynb")),
                         "Visual data prep", "No-code promo list for marketing"))
        if m["dashboard"]["key"] in st:
            rows.append((link(f"/dashboardsv3/{st[m['dashboard']['key']]}/published", p + m["dashboard"]["name"]), "AI/BI dashboard", "Published 4-page operations dashboard"))
        if m["genie"]["key"] in st:
            rows.append((link(f"/genie/rooms/{st[m['genie']['key']]}", p + m["genie"]["title"]), "Genie agent", "Natural-language Q&A on the gold layer"))
        rows += [(link(f"/sql/alerts-v2/{v}", k.removeprefix("alert_").title() + " alert"), "Alert", "Weekly, Monday 8 AM New York (paused)")
                 for k, v in st.items() if k.startswith("alert_")]
        body = "".join(f'<tr style="background:{"#FBF6EF" if i % 2 else "#FFFFFF"};"><td style="padding:5px 8px;">{a}</td>'
                       f'<td style="padding:5px 8px;">{t}</td><td style="padding:5px 8px;">{d}</td></tr>' for i, (a, t, d) in enumerate(rows))
        return (f'<div style="font-family:sans-serif;color:#3E2723;"><div style="font-weight:700;font-size:16px;margin-bottom:6px;">&#128193; Asset index</div>'
                f'<table style="border-collapse:collapse;font-size:13px;width:100%;"><tr style="background:#5D4037;color:#FFF8F0;">'
                f'<th style="text-align:left;padding:5px 8px;">Asset</th><th style="text-align:left;padding:5px 8px;">Type</th>'
                f'<th style="text-align:left;padding:5px 8px;">Purpose</th></tr>{body}</table></div>')
