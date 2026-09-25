"""Two fields, every MCP tool: data_check, labelling with skips, synthetic rows re-checked blind, finetune, evaluate,
save and ds.load."""

import csv
from pathlib import Path

import decisionsmith as ds
from decisionsmith import mcp

EXAMPLES = Path(__file__).parents[2]
DATA = EXAMPLES / "data" / "toy.csv"
TOY = str(DATA)
SCHEMA = "%s:Ticket" % (EXAMPLES / "schema.py")  # team (billing/technical/sales) + wants_refund (yes/no)

# 1. look at the data (the agent proposes the schema from this and the user approves it)
check = mcp.data_check(TOY, SCHEMA)
print("data:", check["rows"], "rows ·", check["advice"][0])

# 2. pick 150 rows into a session; about 20% are held out as split=test
started = mcp.golden_start(TOY, SCHEMA, n=150, strategy="diverse", agent="claude-code")
session = started["session"]

# 3. label: a scripted stand-in for the agent (a real agent reads each text); it skips texts it can't tell apart
rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
known = {r["text"]: r for r in rows}


def agent(text):
    r = known.get(text)
    if r is None:  # a text the data-writer wrote: judge it from its words
        team = "sales" if "price" in text or "plan" in text else "technical"
        return {"team": team, "wants_refund": "refund" in text}
    return {"team": r["team"], "wants_refund": r["wants_refund"]}


def label_until(session, stop_at_pass):
    while (batch := mcp.golden_batch(session))["items"] and batch["pass"] != stop_at_pass:
        answers = [
            {"id": i["id"], "skip": "too short to tell"}
            if len(i["text"].split()) < 5
            else {"id": i["id"], "answers": agent(i["text"])}
            for i in batch["items"]
        ]
        result = mcp.golden_submit(session, answers, agent="claude-code")
        assert not result["rejected"], result["rejected"]


label_until(session, stop_at_pass=2)  # pass 1 only
status = mcp.golden_status(session)
print("labelled", status["labelled"], "· skipped", status["skipped"], "· balance", status["balance"]["team"])

# 4. short of sales rows? the data-writer adds a few; they are re-checked blind with the rest in pass 2
written = ["What is the price of the team plan?", "Is there a cheaper plan for students?", "Can I get the price list?"]
mcp.golden_add(session, [{"text": t, "answers": {"team": "sales", "wants_refund": False}} for t in written])
label_until(session, stop_at_pass=None)  # pass 2: the blind re-check
status = mcp.golden_status(session)
print("re-checked", status["rechecked"], "· agreement", status["agreement"], "· disagreements", status["disagreements"])

# 5. golden.csv, train, evaluate on the held-out rows, save a versioned folder
print(mcp.golden_finish(session)["message"])
trained = mcp.finetune("golden.csv", SCHEMA, out="runs/v1")
report = mcp.evaluate("runs/v1", "golden.csv", SCHEMA, save="models/ticket")
print("go:", report["go"], "·", report["reasons"][0])
print("accuracy by labeller:", report["details"]["labelled_by"])

# 6. what the production code loads
model = ds.load(report["saved"])
print(mcp.model_info(report["saved"])["load"], "->", model.predict("I was charged twice, please refund me"))
