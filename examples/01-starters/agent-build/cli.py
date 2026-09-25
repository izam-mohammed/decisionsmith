"""The same flow from the command line: `decisionsmith golden --teacher agent`, the agent labels, `--finish`, train,
evaluate and save, then `decisionsmith eval`."""

import csv
import json
import subprocess
import sys
from pathlib import Path

from decisionsmith import mcp

TOY = Path(__file__).parents[2] / "data" / "toy.csv"
rows = list(csv.DictReader(TOY.open(encoding="utf-8")))
Path("texts.txt").write_text("\n".join(r["text"] for r in rows), encoding="utf-8")
labels = "billing,technical,sales"


def decisionsmith(*args: str) -> str:
    print("$ decisionsmith", " ".join(args))
    done = subprocess.run([sys.executable, "-m", "decisionsmith.cli", *args], capture_output=True, text=True)
    print(done.stdout, done.stderr, "exit code %d" % done.returncode, sep="")
    return done.stdout


started = json.loads(
    decisionsmith(
        "golden",
        "texts.txt",
        "--labels",
        labels,
        "--teacher",
        "agent:codex",
        "--strategy",
        "random",
        "-n",
        "100",
        "--json",
    )
)

# the agent labels through the MCP tools (a scripted stand-in here: answers from the toy file's labels)
known = {r["text"]: r["team"] for r in rows}
while (batch := mcp.golden_batch(started["session"]))["items"]:
    mcp.golden_submit(
        started["session"], [{"id": i["id"], "answers": {"label": known[i["text"]]}} for i in batch["items"]]
    )

decisionsmith("golden", "--finish", started["session"])
decisionsmith("train", "golden.csv", "--labels", labels, "--out", "runs/v1")
saved = mcp.evaluate("runs/v1", "golden.csv", labels=labels.split(","), save="models/team")["saved"]
decisionsmith("eval", saved, "golden.csv")
