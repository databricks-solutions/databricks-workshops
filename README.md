# Databricks Workshops

This repository contains a collection of Databricks workshops designed for direct implementation. Each workshop provides hands-on, step-by-step guidance for specific data engineering, analytics, or machine learning tasks within Databricks. The content is modular and ready to use, helping teams accelerate solution development and learning.


## Workshops

| Workshop | What it is | How to run it |
|---|---|---|
| [`dbsql_workshop`](dbsql_workshop/) | Bean & Brew Coffee Co.: Databricks SQL on a fictional coffee chain, from raw data to a dashboard, a Genie agent and alerts | **Open in Databricks** button in its README |
| [`v2v`](v2v/) | Vibing to Value: the guide for the hands-on build day in the Databricks Seattle office on 2026-11-17 | Open the notebook `Vibing_to_Value_Seattle.py` |


## Open in Databricks

Workshops with an **Open in Databricks** button at the top of their README set themselves up in your own workspace:

1. You sign in to your Databricks workspace.
2. This whole repository is cloned into your workspace as a Git folder, at `/Users/<you>/databricks-solutions-databricks-workshops`.
3. The workshop's bundle opens in the bundle editor and deploys its default target automatically. You aren't asked for any input.

Each workshop's README lists what its deploy creates. Every workshop runs in a Databricks Free Edition or trial workspace as is; for an existing company workspace, [USER_GUIDE.md](USER_GUIDE.md) lists the workspace settings and permissions you need. Opening a button again is safe: the existing Git folder is reused.

If the automatic deploy fails, the repository is still cloned. Fix the cause (usually a missing catalog or permission; the workshop's README explains its settings), then open the workshop's `databricks.yml`, click the **Deployments** icon and deploy again from there.

To remove a workshop, open its `databricks.yml`, click the **Deployments** icon, then **Destroy**. This removes everything the workshop deployed. The Git folder stays; delete it by hand when you no longer need it.


## Contributing

New workshops follow the rules in [CONTRIBUTING.md](CONTRIBUTING.md), so that any number of them can run in the same workspace and each one tears down cleanly. A check runs on every pull request.


## How to get help

Databricks support doesn't cover this content. For questions or bugs, please open a GitHub issue and the team will help on a best effort basis.


## Security

To report a security vulnerability, follow [SECURITY.md](SECURITY.md). Please don't open a public issue for it.


## License

&copy; 2026 Databricks, Inc. All rights reserved. The source in this repository is provided subject to the
[Databricks License](LICENSE) (also at https://databricks.com/db-license-source). All included or referenced third
party libraries are subject to the licenses set forth below.

| library | description | license | source |
|---------|-------------|---------|--------|
| MLflow  | Open-source platform for managing the complete machine learning lifecycle, including experiment tracking, model management, and deployment | Apache License 2.0 | https://github.com/mlflow/mlflow/blob/master/LICENSE.txt |
| PyYAML  | YAML parser for Python, used by `scripts/check_workshops.py` and the pull request check | MIT License | https://github.com/yaml/pyyaml/blob/main/LICENSE |
