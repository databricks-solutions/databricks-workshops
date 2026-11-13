# Instructions for coding agents

This repository holds Databricks workshops. Each top-level folder is one workshop; most deploy with a Declarative
Automation Bundle through an Open in Databricks button, into strangers' workspaces, and must tear down with Destroy.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you change anything. The rules that matter most:

- Work only inside the workshop folder you were asked to change. Never edit another workshop's files.
- Change root files (`README.md`, `CONTRIBUTING.md`, `USER_GUIDE.md`, this file, `LICENSE`, `NOTICE`, `SECURITY.md`,
  `scripts/`, `.github/`) only when asked. A rule change updates `CONTRIBUTING.md`, this file,
  `scripts/check_workshops.py` and `.github/pull_request_template.md` together.
- Names: `bundle.name` equals the folder name; no `mode:` on any target; `presets.name_prefix: "[<folder>] "` and
  `presets.tags.workshop: <folder>`; code writes only inside `<folder>_*` schemas.
- One click, no input: every variable has a default; the only lookup is `warehouse: Serverless Starter Warehouse`;
  the default target has no `host`; no Python bundle configuration (`python:`).
- Destroy removes everything: schemas, dashboards, Genie agents, alerts, jobs and pipelines are bundle resources.
  Schemas go in `${var.catalog}`, a variable that defaults to `workspace`. No `CREATE CATALOG`, no `CREATE SCHEMA` in
  code, no workspace objects created through the API.
- Minimal DBUs: the Starter warehouse or serverless compute only, schedules paused, models from `ALLOWED_MODELS`.
- Notebooks in source format (`.py` / `.sql` with the `Databricks notebook source` header), never `.ipynb`.
- Each workshop's README has a `## Tear down` section (if it deploys anything) and the support disclaimer under
  `## How to get help`; the root README lists every workshop in its Workshops table.
- Never commit a `presenter/` folder or any file over 5 MB.

Before you commit, run `python scripts/check_workshops.py` (or, without PyYAML installed,
`uv run --no-project --with pyyaml python scripts/check_workshops.py`) and fix every error. If you change a bundle and
have a Databricks profile, also run `databricks bundle validate` in its folder.
