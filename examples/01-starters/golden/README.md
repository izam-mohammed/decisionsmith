# Golden dataset

ds.golden picks the texts most worth labelling (from a harness log or a list), the main LLM labels them, and a held-out split keeps evaluation honest.

```python
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
rows = ds.golden("decisions.db", teacher="claude-opus-5", n=200, schema=labels)  # writes golden.csv

model.train("golden.csv")  # rows marked split=test are never trained on
print(model.evaluate("golden.csv"))  # ...and they are the only ones evaluated
```

| file | what it shows |
|---|---|
| [`cli.py`](cli.py) | The same loop from the command line: `decisionsmith golden` from a log or a texts file, then `decisionsmith eval`. |
| [`from_texts.py`](from_texts.py) | Build a golden dataset from a plain list of texts: the model scores them, `diverse` spreads picks across labels. |
| [`main.py`](main.py) | Close the loop: real samples from a harness log -> a golden dataset labelled by the main LLM -> train -> evaluate. |

## Run

```bash
uv add "decisionsmith[laya,anthropic]"
uv run python examples/01-starters/golden/cli.py
uv run python examples/01-starters/golden/from_texts.py
uv run python examples/01-starters/golden/main.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
