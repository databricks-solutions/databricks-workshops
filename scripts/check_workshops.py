"""Check every workshop in this repository against the rules in CONTRIBUTING.md.

Run from the repository root:  python scripts/check_workshops.py   (needs PyYAML)
Exits 1 if any rule is broken. Warnings point at things a reviewer should look at but don't fail the check.

What it can't check from the files alone (deploying, running and destroying in a test workspace) is listed in
CONTRIBUTING.md under "Test before you open a pull request".
"""
import glob
import os
import re
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOT_WORKSHOPS = {".github", "scripts"}
MAX_FILE_MB = 5  # Open in Databricks clones the whole repository with a 3-minute limit
DISCLAIMER = "Databricks support does not cover this content"
STARTER_LOOKUP = {"warehouse": "Serverless Starter Warehouse"}
# Pay-per-token models a workshop may call. Choose the cheapest model that does the job and isn't legacy or
# deprecated (see the AI models maintenance policy in the Databricks docs), and add it here in the same pull request.
ALLOWED_MODELS = {
    "databricks-gpt-oss-20b",  # 1.0 DBU per 1M input tokens, 4.3 per 1M output (checked 2026-11)
}
SMALLEST_WAREHOUSE = {"enable_serverless_compute": True, "cluster_size": "2X-Small", "max_num_clusters": 1,
                      "auto_stop_mins": 1}
ALWAYS_ON = ("model_serving_endpoints", "vector_search_endpoints", "apps", "database_instances", "postgres_projects")

errors, warnings = [], []


def error(path, msg):
    errors.append(f"{path}: {msg}")


def warn(path, msg):
    warnings.append(f"{path}: {msg}")


def tracked_files():
    """Committed and new files that git doesn't ignore, so this also works before the first commit."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return sorted({f for f in out.split("\0") if f and os.path.isfile(os.path.join(ROOT, f))})


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def bundle_config(bundle_dir):
    """databricks.yml merged with its includes: resources, targets and variables."""
    cfg = load_yaml(os.path.join(bundle_dir, "databricks.yml"))
    resources = {}
    for doc in [cfg] + [load_yaml(p) for pat in cfg.get("include") or []
                        for p in sorted(glob.glob(os.path.join(bundle_dir, pat), recursive=True))]:
        for kind, items in (doc.get("resources") or {}).items():
            resources.setdefault(kind, {}).update(items or {})
        for name, target in (doc.get("targets") or {}).items():
            if doc is not cfg:
                cfg.setdefault("targets", {}).setdefault(name, {}).update(target or {})
    return cfg, resources


def check_bundle(workshop, files):
    rel = f"{workshop}/databricks.yml"
    cfg, resources = bundle_config(os.path.join(ROOT, workshop))
    name = (cfg.get("bundle") or {}).get("name")

    # Names: everything a workshop creates is named after it, so two workshops never collide
    if name != workshop:
        error(rel, f"bundle.name must equal the folder name `{workshop}`, not `{name}`")
    presets = cfg.get("presets") or {}
    if presets.get("name_prefix") != f"[{workshop}] ":
        error(rel, f'presets.name_prefix must be "[{workshop}] " (schemas then become {workshop}_<name>)')
    if presets.get("trigger_pause_status") != "PAUSED":
        error(rel, "presets.trigger_pause_status must be PAUSED, so nothing runs on a schedule after deploy")
    if (presets.get("tags") or {}).get("workshop") != workshop:
        error(rel, f"presets.tags must include `workshop: {workshop}`, so its usage can be measured")

    # Deployable with one click and no input
    if "python" in cfg or "python" in (cfg.get("experimental") or {}):
        error(rel, "no Python bundle configuration: workspace deploys (Open in Databricks) can't run it")
    targets = cfg.get("targets") or {}
    for tname, target in targets.items():
        if (target or {}).get("mode"):
            error(rel, f"target `{tname}` sets mode: {target['mode']}; use no mode (the presets give fixed names)")
    defaults = [t for t, v in targets.items() if (v or {}).get("default")]
    if len(defaults) != 1:
        error(rel, f"exactly one target needs `default: true` (found {defaults or 'none'})")
    elif ((targets[defaults[0]] or {}).get("workspace") or {}).get("host"):
        error(rel, f"the default target `{defaults[0]}` must not set workspace.host: it deploys wherever it's opened")
    for vname, var in (cfg.get("variables") or {}).items():
        var = var if isinstance(var, dict) else {"default": var}
        if "lookup" in var:
            if var["lookup"] != STARTER_LOOKUP:
                error(rel, f"variable `{vname}`: the only allowed lookup is `warehouse: Serverless Starter Warehouse`")
        elif var.get("default") in (None, ""):
            error(rel, f"variable `{vname}` needs a default: Open in Databricks can't prompt for it")
    if resources.get("schemas"):
        catalog = (cfg.get("variables") or {}).get("catalog") or {}
        if (catalog.get("default") if isinstance(catalog, dict) else catalog) != "workspace":
            error(rel, "a `catalog` variable defaulting to `workspace` (the Free Edition default) is required")

    # A full destroy path: the bundle creates only what destroy can remove, and never a catalog
    if resources.get("catalogs"):
        error(rel, "workshops never create catalogs; deploy schemas into the `catalog` variable")
    for key, schema in (resources.get("schemas") or {}).items():
        if (schema or {}).get("catalog_name") != "${var.catalog}":
            error(rel, f"schema `{key}` must use catalog_name: ${{var.catalog}}")

    # Minimal DBUs: serverless, smallest sizes, nothing always on
    for key, wh in (resources.get("sql_warehouses") or {}).items():
        warn(rel, f"sql_warehouse `{key}`: prefer the Serverless Starter Warehouse lookup (Free Edition allows one warehouse)")
        for field, value in SMALLEST_WAREHOUSE.items():
            if (wh or {}).get(field) != value:
                error(rel, f"sql_warehouse `{key}`: {field} must be {value}")
    if resources.get("clusters"):
        error(rel, "no classic clusters: use a SQL warehouse or serverless compute")
    for key, job in (resources.get("jobs") or {}).items():
        job = job or {}
        if "job_clusters" in job:
            error(rel, f"job `{key}`: no job clusters; use serverless or a SQL warehouse")
        for task in job.get("tasks") or []:
            if "new_cluster" in task or "existing_cluster_id" in task or "job_cluster_key" in task:
                error(rel, f"job `{key}` task `{task.get('task_key')}`: no classic compute; use serverless or a SQL warehouse")
        uses_serverless = any(not (t.get("notebook_task") or t.get("sql_task") or {}).get("warehouse_id")
                              for t in job.get("tasks") or [])
        if uses_serverless and job.get("performance_target") not in (None, "STANDARD"):
            warn(rel, f"job `{key}`: performance_target {job['performance_target']} costs more than STANDARD")
    for key, pipe in (resources.get("pipelines") or {}).items():
        if not (pipe or {}).get("serverless"):
            error(rel, f"pipeline `{key}` must set serverless: true")
    for key, alert in (resources.get("alerts") or {}).items():
        if ((alert or {}).get("schedule") or {}).get("pause_status") != "PAUSED":
            error(rel, f"alert `{key}`: set schedule.pause_status: PAUSED")
    for kind in ALWAYS_ON:
        for key in resources.get(kind) or {}:
            warn(rel, f"{kind} `{key}` runs (and bills) until destroyed: say so in the README and keep it minimal")

    # Code: write only inside the workshop's own schemas, call only allowed models
    others = [w for w in WORKSHOPS if w != workshop]
    create = re.compile(r"\b(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|MATERIALIZED\s+VIEW|FUNCTION|VOLUME)"
                        r"(?:\s+IF\s+NOT\s+EXISTS)?|INSERT\s+(?:INTO|OVERWRITE)(?:\s+TABLE)?)\s+(`?\w+`?(?:\.`?\w+`?)+)", re.I)
    for path in files:
        if not path.endswith((".sql", ".py", ".yml", ".yaml")):
            continue
        text = open(os.path.join(ROOT, path), errors="replace").read()
        if re.search(r"\bCREATE\s+CATALOG\b", text, re.I):
            error(path, "never CREATE CATALOG")
        if re.search(r"\bCREATE\s+SCHEMA\b", text, re.I):
            error(path, "don't CREATE SCHEMA in code: declare schemas as bundle resources so destroy drops them")
        for target in create.findall(text):
            parts = target.replace("`", "").split(".")
            if len(parts) >= 2 and not parts[-2].startswith(f"{workshop}_"):
                error(path, f"writes to `{target}`, outside this workshop's schemas ({workshop}_*)")
        for model in re.findall(r"ai_query\(\s*'([^']+)'", text):
            if model not in ALLOWED_MODELS:
                error(path, f"model `{model}` isn't in ALLOWED_MODELS in scripts/check_workshops.py")
        for other in others:
            if re.search(rf"\b{re.escape(other)}_\w+", text) or f"[{other}]" in text:
                error(path, f"refers to another workshop's objects (`{other}`)")
        if re.search(r"\bWorkspaceClient\(|api_client\.do\(|/api/2\.\d/", text) and path.endswith(".py"):
            warn(path, "creates workspace objects through the API? Destroy won't remove them unless they're in the "
                       "bundle's folder; list any others under Tear down in the README")


def check_workshop(workshop):
    files = [f for f in TRACKED if f.startswith(workshop + "/")]
    readme = f"{workshop}/README.md"
    if readme not in files:
        error(workshop, "needs a README.md")
    else:
        text = open(os.path.join(ROOT, readme)).read()
        if DISCLAIMER not in text:
            error(readme, f'needs the support disclaimer ("{DISCLAIMER} ...") under "How to get help"')
        if f"{workshop}/databricks.yml" in files and not re.search(r"^## Tear down", text, re.M | re.I):
            error(readme, 'needs a "## Tear down" section: what Destroy removes and what to delete by hand')
    for path in files:
        if path.endswith(".ipynb"):
            error(path, "save notebooks in source format (.py or .sql with a `Databricks notebook source` header)")
        if "/presenter/" in f"/{path}":
            error(path, "presenter material must stay out of the repository (gitignore presenter/)")
        if os.path.getsize(os.path.join(ROOT, path)) > MAX_FILE_MB * 1024 * 1024:
            error(path, f"larger than {MAX_FILE_MB} MB: generate data in SQL instead of committing it")
    if f"{workshop}/databricks.yml" in files:
        check_bundle(workshop, files)
    nested = [f for f in files if f.endswith("/databricks.yml") and f != f"{workshop}/databricks.yml"]
    for path in nested:
        error(path, "one bundle per workshop, at the workshop folder's root")


TRACKED = tracked_files()
WORKSHOPS = sorted({f.split("/")[0] for f in TRACKED if "/" in f and not f.startswith(".")} - NOT_WORKSHOPS)
for w in WORKSHOPS:
    check_workshop(w)
for path in TRACKED:
    if "/" not in path and os.path.getsize(os.path.join(ROOT, path)) > MAX_FILE_MB * 1024 * 1024:
        error(path, f"larger than {MAX_FILE_MB} MB")

for w in warnings:
    print(f"warning: {w}")
for e in errors:
    print(f"error: {e}")
print(f"Checked {len(WORKSHOPS)} workshops ({', '.join(WORKSHOPS)}): {len(errors)} errors, {len(warnings)} warnings")
sys.exit(1 if errors else 0)
