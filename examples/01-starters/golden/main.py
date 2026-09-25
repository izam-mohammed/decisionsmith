"""Close the loop: real samples from a harness log -> a golden dataset labelled by the main LLM -> train -> evaluate."""

import csv
from pathlib import Path

import decisionsmith as ds

TOY = Path(__file__).parents[2] / "data" / "toy.csv"
texts = [r["text"] for r in csv.DictReader(TOY.open(encoding="utf-8"))]
labels = ["billing", "technical", "sales"]

# production: the LLM answers, Laya is measured in shadow mode, and the texts are logged
model = ds.model(labels)
with ds.harness(model, teacher="claude-haiku-4-5", mode="shadow", log="decisions.db") as h:
    h.many(texts)

# developer: pick the 200 texts the student was least sure about and have the main LLM label them
rows = ds.golden("decisions.db", teacher="claude-opus-4-5", n=200, schema=labels)  # writes golden.csv

model.train("golden.csv")  # rows marked split=test are never trained on
print(model.evaluate("golden.csv"))  # ...and they are the only ones evaluated
