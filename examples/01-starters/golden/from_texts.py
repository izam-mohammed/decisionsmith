"""Build a golden dataset from a plain list of texts: the model scores them, `diverse` spreads picks across labels."""

import csv
from pathlib import Path

import decisionsmith as ds

TOY = Path(__file__).parents[2] / "data" / "toy.csv"
texts = [r["text"] for r in csv.DictReader(TOY.open(encoding="utf-8"))]

model = ds.model(["billing", "technical", "sales"])
rows = ds.golden(texts, teacher="claude-haiku-4-5", n=60, strategy="diverse", schema=model, out="golden.csv")
print(rows[0])  # {'id': ..., 'text': ..., 'answers': {...}, 'split': 'train' or 'test', 'labelled_by': ...}
print(sum(r["split"] == "test" for r in rows), "rows held out for evaluation")
