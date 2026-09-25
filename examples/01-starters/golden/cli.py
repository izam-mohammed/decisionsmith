"""The same loop from the command line: `decisionsmith golden` from a log or a texts file, then `decisionsmith eval`."""

import csv
import subprocess
import sys
from pathlib import Path

import decisionsmith as ds

TOY = Path(__file__).parents[2] / "data" / "toy.csv"
texts = [r["text"] for r in csv.DictReader(TOY.open(encoding="utf-8"))]
Path("texts.txt").write_text("\n".join(texts), encoding="utf-8")
with ds.harness(ds.model(["billing", "technical", "sales"]), teacher="claude-haiku-4-5", mode="shadow") as h:
    h.many(texts[:100])


def decisionsmith(*args: str) -> None:
    print("$ decisionsmith", " ".join(args))
    done = subprocess.run([sys.executable, "-m", "decisionsmith.cli", *args], capture_output=True, text=True)
    print(done.stdout, done.stderr, "exit code %d (2 = done, but not ready yet)" % done.returncode, sep="")


labels = "billing,technical,sales"
decisionsmith("golden", "--log", "decisions.db", "--labels", labels, "--teacher", "claude-opus-5", "-n", "80")
decisionsmith(
    "golden",
    "texts.txt",
    "--labels",
    labels,
    "--teacher",
    "claude-haiku-4-5",
    "--strategy",
    "random",
    "-n",
    "120",
    "--out",
    "golden-random.csv",
)

model = ds.model(labels.split(","))
model.train("golden-random.csv")
path = model.save("models/ticket")
decisionsmith("eval", path, "golden.csv")
decisionsmith("eval", path, "golden-random.csv", "--target", "0.9", "--out", "report.html")
