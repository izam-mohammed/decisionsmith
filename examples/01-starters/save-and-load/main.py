"""Train, evaluate, save a versioned model folder, and load it back with ds.load: the end of the developer flow."""

import csv
from pathlib import Path

import decisionsmith as ds

TOY = Path(__file__).parents[2] / "data" / "toy.csv"
rows = [(r["text"], r["team"]) for r in csv.DictReader(TOY.open(encoding="utf-8"))]
train, test = rows[:220], rows[220:]

model = ds.model(["billing", "technical", "sales"])
model.train(train)
print(model.evaluate(test))  # per field: accuracy, macro-F1, ECE, threshold, coverage, then go/no-go

path = model.save("models/team")  # models/team-v1 the first time, then team-v2, ...
print("saved", path)

m = ds.load(path)  # labels, calibration and thresholds come with the folder
print(m.predict("I was charged twice this month"))
