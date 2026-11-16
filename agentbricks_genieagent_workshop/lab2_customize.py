"""Notebook 02 helper: turn a fresh `agentbricks init` project into the Kasterwell ops agent.

Run from inside the project folder (in the web terminal):
    python <kit folder>/lab2_customize.py                     # set the instructions, remove sample tools
    python <kit folder>/lab2_customize.py --add-rule "..."    # add one rule to the instructions (block H)

The template's sample send_message tool asks for human approval, which isn't supported when tools run on
behalf of the signed-in user (our Genie Agent and UC function tools do), so it's removed.
"""
import argparse
import re
from pathlib import Path

SYSTEM_PROMPT = '''You are the Kasterwell Ops Assistant for Kasterwell Health, a virtual-care platform used by about 120 clinics.
- Numbers and trends: ask the clinic_ops Genie Agent tool. It answers with the user's own data permissions. Pass the user's question in their own words (keep terms like "respiratory surge" or "renewal risk") so its definitions apply.
- A specific clinic: call clinic_snapshot first. Judge its health only by cs02_at_risk and its red signals, and compare rates only with the network_* values it returns.
- WHY something happened or WHAT to do: call search_docs and cite the document id, for example (OPS-07).
- If a clinic or result is missing, call my_access. When the user sees fewer clinics than the total, say the clinic is outside their access and how many of the total they can see. Don't say it doesn't exist or ask them to check the spelling.
- Numbers: use only numbers your tools returned, copied exactly, with the period they cover. Never calculate or estimate one yourself; ask the Genie Agent again instead.
- Memory: save the user's role, focus and answer preferences. Never save patient information.
- Lead with the answer and the key number, then 2-4 short bullets. The data runs through 2026-10-31.'''

parser = argparse.ArgumentParser()
parser.add_argument("--add-rule", help="one more instruction line to append")
args = parser.parse_args()

agent_py = Path("agent/agent.py")
if not agent_py.exists():
    raise SystemExit("Run this from your project folder (the one with agent.toml): cd ~/<your-agent-name>")

src = agent_py.read_text()
if "SYSTEM_PROMPT" not in src:
    src = src.replace('REQUIRE_APPROVAL = {"send_message": True}',
                      f'REQUIRE_APPROVAL: dict = {{}}\n\nSYSTEM_PROMPT = """{SYSTEM_PROMPT}"""')
    src = src.replace("        tools=tools,\n        middleware=middleware,",
                      "        tools=tools,\n        system_prompt=SYSTEM_PROMPT,\n        middleware=middleware,")
    assert "system_prompt=SYSTEM_PROMPT" in src and "REQUIRE_APPROVAL: dict = {}" in src, \
        "The agentbricks template changed; ask a presenter for help."
    print("✅ Instructions set.")

if args.add_rule:
    rule = args.add_rule.strip().replace('"', "'")
    src = re.sub(r'(SYSTEM_PROMPT = """.*?)"""', lambda m: f'{m.group(1)}\n- {rule}"""', src, count=1, flags=re.S)
    print(f"✅ Added rule: {rule}")

agent_py.write_text(src)
for sample in ("agent/tools/sample_tool.py", "agent/tools/send_message.py"):
    Path(sample).unlink(missing_ok=True)
print("Next: agentbricks deploy <your-agent-name>")
