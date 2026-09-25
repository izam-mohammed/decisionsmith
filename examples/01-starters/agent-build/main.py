"""The minimal agent flow: ds.golden(teacher="agent") -> the agent labels through the MCP tools -> train -> evaluate."""

import csv
from pathlib import Path

import decisionsmith as ds
from decisionsmith import mcp

TOY = Path(__file__).parents[2] / "data" / "toy.csv"
rows = list(csv.DictReader(TOY.open(encoding="utf-8")))
labels = ["billing", "technical", "sales"]

# pick the rows worth labelling into a session file instead of calling an LLM
ds.golden([r["text"] for r in rows], teacher="agent:claude-code", schema=labels, strategy="diverse", n=120)

# a scripted stand-in for the coding agent: it answers from the toy file's labels, where a real agent reads the text
known = {r["text"]: r["team"] for r in rows}
session = "golden.session.json"
while (batch := mcp.golden_batch(session))["items"]:  # pass 1, then the blind re-check (pass 2)
    mcp.golden_submit(session, [{"id": i["id"], "answers": {"label": known[i["text"]]}} for i in batch["items"]])
print(mcp.golden_finish(session)["message"])  # writes golden.csv

model = ds.model(labels)
model.train("golden.csv")  # split=test rows are held out
print(model.evaluate("golden.csv"))  # accuracy by who labelled the rows is at the end
