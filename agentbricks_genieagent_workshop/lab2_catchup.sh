#!/usr/bin/env bash
# Notebook 02 catch-up: runs blocks A-D in one go, then deploys. Notebook 02 prints the exact command:
#   bash <kit folder>/lab2_catchup.sh <agent-name> <genie-space-id> <catalog.schema>
set -euo pipefail
AGENT="${1:?usage: lab2_catchup.sh <agent-name> <genie-space-id> <catalog.schema>}"
GENIE="${2:?usage: lab2_catchup.sh <agent-name> <genie-space-id> <catalog.schema>}"
SCHEMA="${3:?usage: lab2_catchup.sh <agent-name> <genie-space-id> <catalog.schema>}"
KIT="$(cd "$(dirname "$0")" && pwd)"
ME="$(databricks current-user me -o json | python -c 'import json,sys; print(json.load(sys.stdin)["userName"])')"

cd ~
pip install -q "databricks-agentbricks==0.9.0"
[ -d "$AGENT" ] || agentbricks init "$AGENT"
cd "$AGENT"
grep -q 'id = "clinic_ops"' agent.toml || agentbricks tools add genie-agent "$GENIE" --name clinic_ops
for fn in clinic_snapshot my_access search_docs; do
  grep -q "$SCHEMA.$fn" agent.toml || agentbricks tools add uc-function "$SCHEMA.$fn"
done
agentbricks memory bind "$AGENT-memory"
agentbricks sessions bind "$AGENT-sessions"
agentbricks tracing bind --experiment-name "/Users/$ME/agentbricks_traces/$AGENT"
python "$KIT/lab2_customize.py"
databricks apps get "agent-bricks-$AGENT" >/dev/null 2>&1 || databricks api delete "/api/2.0/agents/runtime-stores/agent-bricks-$AGENT" >/dev/null 2>&1 || true
agentbricks deploy "$AGENT"
