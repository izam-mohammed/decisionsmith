# Evaluate, save and load

model.evaluate on held-out rows, model.save to a versioned folder, ds.load in the app that serves it.

```python
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
```

| file | what it shows |
|---|---|
| [`load_with_class.py`](load_with_class.py) | Save a model with several fields and load it back with your own Pydantic class (it must match the saved one). |
| [`main.py`](main.py) | Train, evaluate, save a versioned model folder, and load it back with ds.load: the end of the developer flow. |

## Run

```bash
uv add "decisionsmith[laya]"
uv run python examples/01-starters/save-and-load/load_with_class.py
uv run python examples/01-starters/save-and-load/main.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
