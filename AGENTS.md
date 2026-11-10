# Instructions for coding agents

This repository holds Databricks workshops. Each top-level folder is one workshop; most deploy with a Declarative
Automation Bundle through an Open in Databricks button, into strangers' workspaces, and must tear down with Destroy.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you change anything. The rules that matter most:

- Work only inside the workshop folder you were asked to change. Never edit another workshop's files.
- Names: no `mode:` on any target; `presets.name_prefix: "[<folder>] "`, and code writes only inside `<folder>_*`
  schemas.
- One click, no input: every variable has a default; the only lookup is `warehouse: Serverless Starter Warehouse`;
  the default target has no `host`; no Python bundle configuration (`python:`).
- Destroy removes everything: schemas, dashboards, Genie agents, alerts and jobs are bundle resources. No
  `CREATE CATALOG`, no `CREATE SCHEMA` in code, no workspace objects created through the API.
- Minimal DBUs: the Starter warehouse or serverless compute only, schedules paused, models from `ALLOWED_MODELS`.
- Notebooks in source format (`.py` / `.sql` with the `Databricks notebook source` header), never `.ipynb`.
- Never commit a `presenter/` folder or any file over 5 MB.

Before you commit, run `python scripts/check_workshops.py` and fix every error. If you change a bundle and have a
Databricks profile, also run `databricks bundle validate` in its folder.
