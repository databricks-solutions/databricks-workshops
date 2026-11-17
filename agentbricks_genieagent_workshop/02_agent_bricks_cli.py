# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Build and deploy an agent with the Agent Bricks CLI (about 27 minutes)
# MAGIC
# MAGIC The **Agent Bricks CLI** (Beta) is a command-line tool that takes you from an empty folder to a deployed agent in a handful of commands. You'll give the agent your **Genie Agent from notebook 01** as a tool, plus three Unity Catalog functions and a memory, and ship it as a **Databricks App** with its own web address.
# MAGIC
# MAGIC | Step | Command | What you get |
# MAGIC |---|---|---|
# MAGIC | A | `agentbricks init` | A working agent project (LangGraph by default) with a chat UI |
# MAGIC | B | `agentbricks tools list` / `tools add` | Governed tools: your Genie Agent and three Unity Catalog functions |
# MAGIC | C | `memory bind` / `sessions bind` / `tracing bind` | Memory per user, saved conversations, and MLflow tracing |
# MAGIC | D | `lab2_customize.py` + `cat agent.toml` | The agent's instructions, and the one file that declares the whole agent |
# MAGIC | E | `agentbricks deploy` | A Databricks App with its own identity and web address |
# MAGIC | F | Chat with it | Answers that combine numbers, documents and your permissions |
# MAGIC | G | Look at its memory | What it remembers about you, and how to make it forget |
# MAGIC | H | Change it and redeploy | The build, test, change loop in about 30 seconds |
# MAGIC
# MAGIC **How to work through this notebook:** you'll type commands in the **web terminal**, a Linux command line in your browser that's already signed in as you. This notebook prints your exact commands; you copy them into the terminal. Keep this notebook in one browser tab and the terminal in another.
# MAGIC
# MAGIC > ⚠️ **Run the cells one at a time, from top to bottom, when a step tells you to. Don't use Run all.** One cell switches which persona you act as, and it only makes sense at that step.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before you start
# MAGIC - You've finished **00_setup** and **01_genie_agent**. Step 6 of notebook 01 saved your Genie Agent's id, and this notebook reads it.
# MAGIC - This notebook is connected to **Serverless** (top right).
# MAGIC
# MAGIC Run the next two cells. The first loads your names, as in the earlier notebooks. The second finds the Genie Agent you built in notebook 01.
# MAGIC
# MAGIC **Expected output:** `Your Genie Agent:` followed by a long id such as `01f1d350f1e61412a2c3a08cacace631`.

# COMMAND ----------

# MAGIC %run ./resources/_config

# COMMAND ----------

GENIE_SPACE_ID = setting("genie_space_id")
assert GENIE_SPACE_ID, "No Genie Agent id saved yet: finish Step 6 of notebook 01 (or run its 'Skip the clicks' cell)."
print(f"Your Genie Agent: {GENIE_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC | If something goes wrong | Fix |
# MAGIC |---|---|
# MAGIC | `No Genie Agent id saved yet` | Go back to notebook 01 and run Step 6, or its **Skip the clicks** cell, then run this cell again |
# MAGIC | `Table or view not found ... kit_settings` | Notebook 00 didn't finish. Run **00_setup** again |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Open the web terminal
# MAGIC
# MAGIC ### 1a. Open the terminal panel
# MAGIC Click the **terminal icon** (a small `>_` box) near the bottom of the narrow bar on the right edge of this notebook. A **Terminal** panel opens below the notebook. It takes about 10 seconds to start.
# MAGIC
# MAGIC ![The terminal icon in the right sidebar](./resources/images/02_open_terminal.png)
# MAGIC
# MAGIC ### 1b. Move it to its own browser tab
# MAGIC The panel is small. Click the **Start a terminal session in a new tab** icon (a box with an arrow) at the top right of the panel. The terminal opens full size in a new browser tab. You can close the panel in this notebook with the **X**.
# MAGIC
# MAGIC ![Open the terminal in a new tab](./resources/images/02_terminal_new_tab.png)
# MAGIC
# MAGIC When the terminal is ready you'll see *Welcome to the Databricks Web Terminal!* and a prompt ending in `$`. That's where you paste commands.
# MAGIC
# MAGIC **Tips for the terminal**
# MAGIC - **Paste:** `Cmd+V` on a Mac. On Windows, use `Ctrl+Shift+V` or right-click and choose **Paste**.
# MAGIC - **Run:** after pasting, press **Enter**. If you paste several lines, each line runs in turn.
# MAGIC - **Wait for the prompt:** a command is finished when you see the `$` prompt again. Don't paste the next block before then.
# MAGIC - The first line about **tmux** is normal: it keeps your session alive if you close the tab.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Print your commands
# MAGIC Run the cell below. It prints **your** commands for steps A to E, already filled in with your names, schema and Genie Agent id. You'll copy one block at a time into the terminal in Step 3.

# COMMAND ----------

steps = {
    "A · Install the CLI and create your project (about 30 seconds)": f"""cd ~
pip install -q "databricks-agentbricks==0.9.0"
agentbricks init {AGENT_NAME}
cd {AGENT_NAME}""",
    "B · See what tools you can add, then add yours": f"""agentbricks tools list
agentbricks tools add genie-agent {GENIE_SPACE_ID} --name clinic_ops
agentbricks tools add uc-function {S}.clinic_snapshot
agentbricks tools add uc-function {S}.my_access
agentbricks tools add uc-function {S}.search_docs""",
    "C · Memory, sessions and tracing": f"""agentbricks memory bind {MEMORY_STORE}
agentbricks sessions bind {SESSION_STORE}
agentbricks tracing bind --experiment-name {TRACE_EXPERIMENT}""",
    "D · Set its instructions, then look at what you built": f"""python {RESOURCES}/lab2_customize.py
cat agent.toml""",
    "E · Deploy (3 to 5 minutes the first time)": f"""databricks apps get {APP_NAME} >/dev/null 2>&1 || databricks api delete /api/2.0/agents/runtime-stores/{APP_NAME} >/dev/null 2>&1
agentbricks deploy {AGENT_NAME}""",
}
for title, cmd in steps.items():
    print(f"# ── {title}\n{cmd}\n")
print(f"# ── Catch-up (only if you fall behind): steps A to E in one command\n"
      f"bash {RESOURCES}/lab2_catchup.sh {AGENT_NAME} {GENIE_SPACE_ID} {S}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Build your agent in the terminal
# MAGIC For each block below: select the lines under its `# ──` heading in the output above (not the heading itself), copy them, paste them into the terminal, and press **Enter**. Then compare what you see with the screenshot.
# MAGIC
# MAGIC ### A. Install the CLI and create your project
# MAGIC `pip install` downloads the CLI (about 20 seconds, with no output). `agentbricks init` then creates a folder with a complete agent project: the agent code, a chat app and a settings file called `agent.toml`.
# MAGIC
# MAGIC **Expected output:** a green box that starts with **✔ Scaffolded 'agent-langgraph'**, and the prompt changes to `~/<your-agent-name>$` because the last line moved you into the project folder.
# MAGIC
# MAGIC ![init creates the project](./resources/images/02_init.png)
# MAGIC
# MAGIC > You can ignore the *Next steps* in that box (`.env`, `agentbricks dev`). They're for running the agent on a laptop. The store names it lists end in a random suffix; Step C replaces them with your own names.
# MAGIC
# MAGIC ### B. Add tools
# MAGIC A **tool** is something the agent can call to get information. The first command, `tools list`, shows every kind of tool you can add:
# MAGIC 1. **Built-in recipes:** a sandbox for running code, Unity Catalog functions, Genie One, and **Genie Agents** like the one you built.
# MAGIC 2. **MCP services** managed by Databricks, such as web search, Slack, GitHub and Google Drive. You won't add these today.
# MAGIC
# MAGIC ![tools list shows the kinds of tools you can add](./resources/images/02_tools_list.png)
# MAGIC
# MAGIC The next four commands add your tools:
# MAGIC
# MAGIC | Tool | What the agent uses it for |
# MAGIC |---|---|
# MAGIC | `clinic_ops` | Your Genie Agent from notebook 01: any question about the numbers |
# MAGIC | `clinic_snapshot` | A one-row summary of a clinic |
# MAGIC | `my_access` | Which persona you are and how many clinics you can see |
# MAGIC | `search_docs` | Kasterwell's policies, playbooks and product docs |
# MAGIC
# MAGIC **Expected output:** four green boxes: **✔ Added clinic_ops**, **✔ Added clinic_snapshot**, **✔ Added my_access** and **✔ Added search_docs**.
# MAGIC
# MAGIC ![Four tools added](./resources/images/02_tools_add.png)
# MAGIC
# MAGIC ### C. Turn on memory, sessions and tracing
# MAGIC - **Memory** stores what the agent learns about each person (your role, your preferences) across conversations.
# MAGIC - **Sessions** store each conversation, so you can come back to it later.
# MAGIC - **Tracing** records every step the agent takes in an MLflow experiment, so you can see exactly what it did.
# MAGIC
# MAGIC **Expected output:** three green boxes: **✔ Bound memory store**, **✔ Bound session store** and **✔ Tracing on**. The stores themselves are created when you deploy.
# MAGIC
# MAGIC ![Memory, sessions and tracing bound](./resources/images/02_bind.png)
# MAGIC
# MAGIC ### D. Give it instructions and look at what you built
# MAGIC `lab2_customize.py` writes the agent's **instructions**: what it's for, which tool to use for what, and rules such as *never save patient information*. `cat agent.toml` then prints the settings file.
# MAGIC
# MAGIC **Expected output:** *✅ Instructions set.*, then the file. Look for:
# MAGIC - `[memory_store]`, `[session_store]` and `[tracing]` with your names from Step C.
# MAGIC - Four `[[tools]]` sections, each with **`auth = "user"`**. That means every tool runs **as the person chatting**, so the agent can never see more data than they can. The persona row filters from notebook 01 apply automatically.
# MAGIC
# MAGIC ![agent.toml: four tools, each with auth = "user"](./resources/images/02_agent_toml.png)
# MAGIC
# MAGIC ### E. Deploy
# MAGIC The first line is a safety check: if no app with your name exists yet, it removes any storage left behind by an earlier app with the same name. It prints nothing. Then `agentbricks deploy` packages the project as a Databricks App, creates the memory and session stores, and starts the app. The first deploy takes **3 to 5 minutes**; read Step 4 while you wait. Lines scroll by as files upload; that's normal.
# MAGIC
# MAGIC **Expected output:** a green box that starts with **✔ Deployed agent 'agent-bricks-<your-agent-name>'**, and *Deployed in ...* at the bottom.
# MAGIC 1. **URL** is your app's web address. You'll open it in Step 5.
# MAGIC 2. *Couldn't check the agent answers ... uses a PAT* is a harmless warning: the terminal can't run the CLI's automatic check. Your app is fine.
# MAGIC 3. The **Traces** link starts with `http://127.0.0.1` and won't open from your browser. Use the **Traces** item in the app instead (Step 5).
# MAGIC
# MAGIC ![Deployed: your app URL](./resources/images/02_deployed.png)

# COMMAND ----------

# MAGIC %md
# MAGIC | If something goes wrong in Step 3 | Fix |
# MAGIC |---|---|
# MAGIC | `agentbricks: command not found` | Run `pip install -q "databricks-agentbricks==0.9.0"` again, or open a new terminal tab |
# MAGIC | `pip` can't reach PyPI | Your workspace has no internet access. Use a trial workspace you signed up for with your work email (see the pre-work email) |
# MAGIC | `Run this from your project folder` | Type `cd ~/<your-agent-name>` (run `ls ~` to see the folder name), then repeat the block |
# MAGIC | `tools add` says the tool already exists | You ran the block twice. That's fine; carry on |
# MAGIC | Deploy says *"no files found"* | You created the project inside a git-ignored folder. Type `cd ~` and start again from block A |
# MAGIC | Deploy says *"Runtime Store '...' does not belong to this app identity"* | An earlier app with the same name left its storage behind. Copy block E again, including its first line, and run both lines. If it still fails, ask a TA |
# MAGIC | Tool error *"managed MCP servers are not enabled"* | Workspace settings → **Previews** → turn on **Managed MCP Servers**, then deploy again |
# MAGIC | You fell behind | Paste the **Catch-up** command printed at the end of Step 2. It runs A to E for you |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4. While it deploys: what did you just build?
# MAGIC
# MAGIC **`agent.toml` is the whole agent.** It declares the tools, the memory store, the session store and the trace experiment. When you run `deploy`, the CLI:
# MAGIC - packages the project as a **Databricks App** named `agent-bricks-<your-agent-name>`,
# MAGIC - creates the memory and session stores if they don't exist yet,
# MAGIC - gives the app its own **service principal** (an identity for software) with exactly the access it needs,
# MAGIC - and calls models through Databricks model serving (Claude Sonnet by default), so there are no API keys anywhere.
# MAGIC
# MAGIC 🔐 **`auth = "user"`.** Each tool runs **on behalf of the person chatting**. A West care coordinator and a billing analyst can use the same agent and each sees only their own data.
# MAGIC
# MAGIC 🧠 **Sessions vs. memory.** A *session* is one conversation's history. *Memory* is what the agent keeps about **you** across all your conversations, stored separately for each person.
# MAGIC
# MAGIC 📈 **Tracing.** Every request, tool call and answer is recorded in an MLflow experiment. Reading traces and measuring quality is the subject of the *End-to-End AI Agents on Databricks* workshop.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5. Open your agent
# MAGIC Run the cell below and click the link, or copy the **URL** from the deploy output. The app opens in a new tab. The first time, it can show a spinner for up to a minute while the app starts, and it may ask you to allow access: click **Allow**.

# COMMAND ----------

host = spark.conf.get("spark.databricks.workspaceUrl")
displayHTML(f'<a href="https://{host}/apps/{APP_NAME}" target="_blank" style="font-size:16px">Open your app: {APP_NAME} →</a>'
            '<p>On the app page, click the app URL at the top to open the chat.</p>')

# COMMAND ----------

# MAGIC %md
# MAGIC Your agent opens in the **Agent Bricks Workbench**, a test chat that comes with every project:
# MAGIC 1. **Capabilities:** a green dot for **Sessions**, **Tools** (*4 connected*), **Memory** and **Traces** means they're all working. Click any of them for details.
# MAGIC 2. **New session** starts a fresh conversation. Your earlier sessions are listed underneath.
# MAGIC 3. **The message box.** Type here and press **Enter**. Leave the mode on **Streaming** and the model on the default.
# MAGIC 4. **Events** shows the raw requests behind the page. You can ignore it.
# MAGIC
# MAGIC ![The Agent Bricks Workbench](./resources/images/02_workbench.png)
# MAGIC
# MAGIC > ⏳ **Wait for each answer to finish before you type the next message.** The box stays greyed out while the agent works, and anything you type then is lost. Each answer shows the tool calls the agent made (*Tool request*, *Tool result*) before its reply.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6. Chat with your agent (F)
# MAGIC
# MAGIC ### F1. Tell it about yourself
# MAGIC Paste this message:
# MAGIC
# MAGIC > *I'm a care coordinator for the West region. I care most about wait times and no-shows, and I like short bullet answers. Please remember that.*
# MAGIC
# MAGIC **Expected:** a few **remember** tool results (*Remembered in /role.md*, */focus.md*, */preferences.md*; the names can vary), then *Saved. Here's what I'll remember:* with your role, focus and format.
# MAGIC
# MAGIC ![The agent saves three memories](./resources/images/02_remember.png)
# MAGIC
# MAGIC ### F2. Ask a question that needs two tools
# MAGIC > *Are evening no-shows a problem for our behavioral health clinics, and what does our protocol say?*
# MAGIC
# MAGIC Watch the tool calls: **clinic_ops** (your Genie Agent) gets the numbers and **search_docs** finds the protocol. **Expected:** evening no-shows are about **twice** the rate of other times (about 14% vs 7% over the last three months), with what the protocol documents (`OPS-...`) say to do, in short bullets.

# COMMAND ----------

# MAGIC %md
# MAGIC ### F3. Ask about a clinic you can't see
# MAGIC First switch your persona to the **West care coordinator**, so the agent only sees West clinics, the same as in Step 5 of notebook 01. Run the cell below.
# MAGIC
# MAGIC **Expected output:** `Row(persona='care_coordinator_west', display_name='Care coordinator (West)')`

# COMMAND ----------

set_persona("care_coordinator_west")

# COMMAND ----------

# MAGIC %md
# MAGIC Back in the app, ask:
# MAGIC > *How is Vellamy Family Health doing?*
# MAGIC
# MAGIC Vellamy is in the Central region. **Expected:** the agent calls **my_access**, then says Vellamy is **outside your access** (you can see 52 of the 120 clinics). It doesn't pretend the clinic doesn't exist, and it doesn't leak its numbers.
# MAGIC
# MAGIC ![Vellamy is outside your access](./resources/images/02_access.png)
# MAGIC
# MAGIC ### F4. Check that it remembers you
# MAGIC 1. Click **New session** (top of the Sessions list). The chat clears, so this conversation knows nothing yet.
# MAGIC 2. Ask: *What should I focus on this week?*
# MAGIC
# MAGIC **Expected:** without being told again, it answers as a **West care coordinator**, focuses on **no-shows and wait times**, and uses **short bullets**: for example the West clinics with the highest no-show rates. That's memory working across conversations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7. Look at its memory, and make it forget (G)
# MAGIC Memory is data, stored per person, that you can look at and delete.
# MAGIC
# MAGIC ### G1. Ask what it remembers
# MAGIC In the same session, ask:
# MAGIC > *What do you remember about me?*
# MAGIC
# MAGIC **Expected:** a **list_memories** or **recall** tool call, then your role, focus and preferences.
# MAGIC
# MAGIC > ℹ️ **The Memory panel shows "No memories ... yet". That's a known issue in this CLI preview:** the agent stores memories under an internal id, and the panel looks them up under your email. Your memories are saved; ask the agent, as above, to see them.
# MAGIC >
# MAGIC > ![The Memory panel is empty for now](./resources/images/02_memory_empty.png)
# MAGIC
# MAGIC ### G2. Make it forget one thing
# MAGIC > *Please forget that I care about wait times and no-shows.*
# MAGIC
# MAGIC **Expected:** a **forget** tool call (*Forgot 1 fact in ...*), then confirmation that the rest (your role and format) is still saved.
# MAGIC
# MAGIC ![The agent forgets one fact](./resources/images/02_forget.png)
# MAGIC
# MAGIC ### G3. Check
# MAGIC Click **New session** and ask *What should I focus on this week?* again. It still knows you're a West care coordinator who likes short answers, but its answer is no longer built around wait times and no-shows (it may lead with upcoming renewals instead). That's the "right to be forgotten", done by the person themselves.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8. Change it and redeploy (H)
# MAGIC Agents are software: you change them, redeploy and check. Run the cell below, then paste both lines into the terminal. They add one rule to the instructions and ship the agent again (**about 30 seconds**, because the stores already exist).
# MAGIC
# MAGIC **Expected output in the terminal:** *✅ Instructions set.*, then the same **✔ Deployed agent** box as in Step E.

# COMMAND ----------

print(f"""# ── H · Change and redeploy
python {RESOURCES}/lab2_customize.py --add-rule "End every answer with one suggested next step for the user."
agentbricks deploy {AGENT_NAME}""")

# COMMAND ----------

# MAGIC %md
# MAGIC Back in the app, **refresh the page**, click **New session** and ask any question, for example *Which West clinics had the highest no-show rate in the last 3 months?*
# MAGIC
# MAGIC **Expected:** the answer now ends with **Suggested next step:**.
# MAGIC
# MAGIC ![Every answer now ends with a suggested next step](./resources/images/02_rule.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9. Switch back to the default persona
# MAGIC You're done. Run the cell below to go back to the **billing analyst** persona, which sees all 120 clinics, so the other notebooks behave as expected.

# COMMAND ----------

set_persona("billing_analyst")

# COMMAND ----------

# MAGIC %md
# MAGIC ## If something goes wrong in the app
# MAGIC | Symptom | Fix |
# MAGIC |---|---|
# MAGIC | The app shows a spinner or *starting* | The first start takes a minute or two. Wait, then refresh |
# MAGIC | A red dot in **Capabilities** | Click it to see the error. If it's Tools, check `cat agent.toml` lists four tools, then deploy again |
# MAGIC | Your message disappeared | You typed while the agent was still answering. Wait until it finishes, then send it again |
# MAGIC | The agent says it can't see any clinic | Run the `set_persona(...)` cell in F3 again. To see everything, run the cell in Step 9 |
# MAGIC | F4 doesn't remember you | Make sure F1 showed **remember** tool results. If not, send the F1 message again, then start a new session |
# MAGIC | The answer doesn't end with a next step after Step 8 | Check the terminal showed **✔ Deployed agent**, refresh the app page, then start a **New session** |
# MAGIC
# MAGIC ### Key takeaways
# MAGIC - A handful of commands took you from nothing to a deployed agent with governed tools, per-person memory, saved conversations and tracing.
# MAGIC - `agent.toml` declares it all, and `deploy` builds it.
# MAGIC - The agent answers **as the person asking**, remembers each person separately, forgets on request, and changes ship in seconds.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up (when you're done with the workshop)
# MAGIC The workshop leaves an app running, which uses compute until you delete it. To remove everything:
# MAGIC 1. Run the cell below. It prints three commands with your names filled in.
# MAGIC 2. Paste them into the web terminal: they delete your app (and its copy of the project), its memory store and its session store.
# MAGIC 3. In **Genie Agents**, delete the agent you built in notebook 01 (and the one *Skip the clicks* made, if you ran it): **⋮** → **Delete**.
# MAGIC 4. To drop your schema with every table and function in it, set `DROP_MY_SCHEMA = True` in the cell and run it again.

# COMMAND ----------

DROP_MY_SCHEMA = False   # set to True, then run this cell again, to drop your schema and everything in it

print(f"""agentbricks deployments delete {APP_NAME} --delete-source -y
agentbricks memory stores delete {MEMORY_STORE} -y
agentbricks sessions stores delete {SESSION_STORE} -y""")
print(f"\nTrace experiment (delete it in Experiments if you want): {TRACE_EXPERIMENT}")
if DROP_MY_SCHEMA:
    spark.sql(f"DROP SCHEMA IF EXISTS {S} CASCADE")
    print(f"✅ Dropped {S}")
