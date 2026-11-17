# Build Governed AI Agents with Agent Bricks and Genie Agents

The hands-on kit for the Databricks 1:M workshop (L100–L200). In about 45 minutes you build:

1. **A Genie Agent** over a fictional health company's data, made accurate with a business definition, example SQL
   and a benchmark.
2. **An agent with the Agent Bricks CLI** (Beta) that uses your Genie Agent as a tool, remembers each user, and
   deploys as a Databricks App.

This workshop has no bundle and no Open in Databricks button: the notebooks set it up as you go, and
[Tear down](#tear-down) lists what to delete afterwards.

## What it needs

More than Databricks Free Edition (see also [USER_GUIDE.md](../USER_GUIDE.md)):

- **A Databricks free trial created with your work email.** Notebook 02 installs the Agent Bricks CLI from PyPI and
  the agent calls a Claude model; workspaces created with a personal email (Gmail, Outlook and so on) can't reach
  PyPI or frontier models. Untested on Free Edition.
- **Serverless compute**, the **web terminal**, **Databricks Apps** and **Genie** (all on by default in a trial).
- The **Managed MCP Servers** preview (**Settings → Previews**), so the agent can call Unity Catalog functions as
  tools.
- `CREATE SCHEMA` on the `workspace` catalog (you own it in a trial).

## What it creates

| What | Name | Created by |
|---|---|---|
| Schema with 9 tables, a persona table, a row filter and 3 functions | `workspace.agentbricks_genieagent_workshop_lab` | `00_setup` |
| Genie Agent | *Kasterwell Clinic Ops* (you name it) | You, in `01_genie_agent` (or its *Skip the clicks* cell) |
| Databricks App and its project copy | `agent-bricks-<your-agent-name>`, in `/Users/<you>/agentbricks_deployments/` | `agentbricks deploy` in `02_agent_bricks_cli` |
| Memory store and session store | `<your-agent-name>-memory`, `<your-agent-name>-sessions` | `agentbricks deploy` |
| MLflow experiment (traces) | `/Users/<you>/agentbricks_traces/<your-agent-name>` | `agentbricks deploy` |

`_config` works out `<your-agent-name>` from your login and prints it at the top of every notebook.

**What bills while idle:** the Databricks App runs until you delete it. Everything else bills only when used: the
notebooks and the terminal run on serverless compute, Genie on your SQL warehouse, and the agent calls a
pay-per-token model.

## Run the workshop

1. Bring this repository into your workspace as a **Git folder** (no GitHub login needed):
   1. In the left sidebar, click **Workspace**, then open your **Home** folder.
   2. Click **Create → Git folder**.
   3. Paste `https://github.com/databricks-solutions/databricks-workshops`.
   4. Optional, this workshop only: open **Advanced**, turn on **Sparse checkout mode**, and enter
      `agentbricks_genieagent_workshop`.
   5. Click **Create Git folder**.
2. Open the `agentbricks_genieagent_workshop` folder and run its three notebooks, in order:

   | # | Notebook | Time | How to run it |
   |---|---|---|---|
   | 1 | `00_setup` | ~3 min | Connect to **Serverless**, click **Run all** |
   | 2 | `01_genie_agent` | ~17 min | One cell at a time, as each step says |
   | 3 | `02_agent_bricks_cli` | ~27 min | One cell at a time; uses the web terminal |

   You don't open anything in `resources/`: the notebooks use it.

## Tear down

There's no bundle, so delete these by hand:

1. Run the **Clean up** cell at the end of `02_agent_bricks_cli`. It prints three commands for the web terminal that
   delete your app (and its project copy), its memory store and its session store.
2. In **Genie Agents**, delete the agent you built (and the one *Skip the clicks* made, if you ran it).
3. Drop the schema: set `DROP_MY_SCHEMA = True` in the **Clean up** cell and run it again, or run
   `DROP SCHEMA workspace.agentbricks_genieagent_workshop_lab CASCADE`.
4. Optional: delete the MLflow experiment the **Clean up** cell names, and the Git folder.

## Files

```
agentbricks_genieagent_workshop/
  00_setup               run 1st: loads the data, creates the personas and the three tools your agent uses
  01_genie_agent         run 2nd: build, teach and test a Genie Agent; call it from code
  02_agent_bricks_cli    run 3rd: build, deploy, chat with and change an agent; clean up
  resources/             used by the notebooks; you don't open these
    _config              works out your names (schema, agent, app) and switches personas
    genie_config.py      the finished Genie Agent's configuration (Skip the clicks in 01)
    lab2_customize.py    helpers that notebook 02 tells you when to run
    lab2_catchup.sh
    data/                the workshop data (parquet, 5 MB in total) and table descriptions
    images/              screenshots used in the notebooks
```

All company, clinic, payer and patient names are fictional; the data is synthetic.

## How to get help

Databricks support does not cover this content. For questions or bugs, open a GitHub issue. The team will help on a best effort basis.
