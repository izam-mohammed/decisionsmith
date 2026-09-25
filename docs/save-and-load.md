# Save and load a model

`model.save()` writes one versioned folder with everything the model needs; `ds.load(path)` brings it back ready
to use, with no need to define the labels or the class again.

**Flow:** developer flow, step 6 (save), and the start of the production flow (load).

```python
import decisionsmith as ds

model = ds.model(["billing", "technical", "sales"])
model.train("tickets.csv")
model.evaluate("test.csv")
path = model.save("models/ticket")  # models/ticket-v1, then models/ticket-v2 next time

m = ds.load(path)  # on the server
print(m.predict("I was charged twice"))
```

## Versions

Saved versions are never overwritten.

| call | writes |
|---|---|
| `model.save()` | `models/<name>-v1`, then `-v2`, ... (`<name>` is `label` for a labels model, else the class name) |
| `model.save("models/ticket")` | the first free `models/ticket-vN` |
| `model.save("models/ticket-v3")` | exactly that folder; an error if it exists |

## The folder

| file | what |
|---|---|
| `model.safetensors` `rl_agent_config.json` `tokenizer/` `encoder/` | the Laya checkpoint; `laya.load(path)` works too |
| `decisionsmith.json` | labels or schema, question text, calibration, per-field thresholds, version, created date, base model, data hash, decisionsmith version |
| `report.json` | the latest `model.evaluate()`; without one, the training report is kept as `train_report.json` |
| `MODEL_CARD.md` | what the model decides, how it was evaluated, how to load it, credits |

## Loading with your own class

A labels model comes back as labels. A class model comes back with a class rebuilt from the saved schema. To
get your own Pydantic class back (with its methods and enums), pass it; it must ask the same questions:

```python
from typing import Literal

from pydantic import BaseModel


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"]
    wants_refund: bool


model = ds.model(Ticket)
model.train("tickets.csv")
m = ds.load(model.save("models/ticket"), Ticket)
print(m.predict("Refund my double charge"))  # Ticket(team=..., wants_refund=...)
```

## Options

| option | default | change it when |
|---|---|---|
| `schema` (second argument of `ds.load`) | rebuilt from the folder | you want your own class back |
| `device` | auto | you want `"cpu"`, `"cuda"` or `"mps"` explicitly |

## What can go wrong

| message | fix |
|---|---|
| `nothing to save yet` | train first (`model.train(...)`), or save a model you loaded |
| `... already exists and saved versions are never overwritten` | use a new version number, or `model.save()` for the next free one |
| `... is a plain Laya checkpoint` | the folder came from `ds.finetune`; use `ds.model(labels_or_class, path)` |
| `Ticket does not match the saved model` | the class has different fields or options than the saved one; load without it |

**Next:** use it in production with `ds.harness(m, teacher=...)` ([guide](guide.md)).
