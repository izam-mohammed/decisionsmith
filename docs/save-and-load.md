# Save and load a model

`model.save()` writes one versioned folder with everything the model needs; `ds.load(path)` brings it back ready
to use, with no need to define the labels or the class again.

**Flow:** developer flow, step 6 (save), and the start of the production flow (load).

> **Upgrading from an earlier decisionsmith?** The path you give `save` is now a name: `model.save("my-model")`
> writes `my-model-v1`, then `my-model-v2`, and prints `saved to ...`. Use the path it returns.

```python
import decisionsmith as ds

model = ds.model(["billing", "technical", "sales"])
report = model.train("tickets.csv")
model.evaluate("test.csv")
if report.switched:  # False: the new weights scored worse on the test split, so the old model was kept
    path = model.save("models/ticket")  # models/ticket-v1, then models/ticket-v2 next time

    m = ds.load(path)  # on the server
    print(m.predict("I was charged twice"))
```

`tickets.csv` and `test.csv` need a `text` column and a `label` column (for a labels model) or one column per field
(for a class). `path` is what `save` returns: the name you pass gets a version number.

`model.train()` switches to the new weights unless they score worse on its held-out test split (a tie counts as
switched), and `report.switched` says which happened. After a training run that kept the old model, `save` still
writes whatever the model had before: an earlier trained or loaded model, a local checkpoint, and any calibration and
thresholds from `model.evaluate()` or `h.adapt()`. A model that started from the downloaded base and never switched
has no local folder, so `save` says training kept the old model instead.

## Versions

Saved versions are never overwritten.

| call | writes |
|---|---|
| `model.save()` | `models/<name>-v1`, then `-v2`, ... (`<name>` is the labels joined, e.g. `billing-technical-sales`, or the class name) |
| `model.save("models/ticket")` | `models/ticket-vN`, one above the highest version already there |
| `model.save("models/ticket-v3")` | exactly that folder; an error if it exists |
| `ds.load("models/ticket-v2").save()` | the next version of the same name, `models/ticket-v3` |

The folder is written under a temporary name and renamed when it is complete, so a crash never leaves a half
written version. If another process takes the same number first, `save` moves on to the next one. `~` in a path
means your home folder.

## The folder

| file | what |
|---|---|
| `model.safetensors` `rl_agent_config.json` `tokenizer/` `encoder/` | the Laya checkpoint; `laya.load(path)` works too |
| `decisionsmith.json` | labels or schema, question text, calibration, per-field thresholds, version, created date, base model, data hash, decisionsmith version |
| `report.json` | the latest `model.evaluate()`; without one, the training report is kept as `train_report.json` |
| `train_log.jsonl` | loss per training epoch |

The saved reports hold numbers only: no texts (the worst examples stay in memory as `report.details["worst"]`),
no training row ids and no local paths. `rl_agent_config.json` holds one-way fingerprints of the training texts
(see [evaluate](evaluate.md#same-text)). Treat the folder like the data it was trained on before you share it.
| `MODEL_CARD.md` | what the model decides, how it was evaluated, how to load it, credits |

## Loading with your own class

A labels model comes back as labels. A class model comes back with a class rebuilt from the saved schema:
`Literal` values keep their type (`Literal[1, 2, 3]` comes back as ints), and an `Enum` field comes back as its
values' strings. To get your own Pydantic class back (with its methods and enums), pass it. It must ask exactly the
same questions: the same fields and options, and the same descriptions, option descriptions and class docstring.
The error lists every difference.

```python
from typing import Literal

from pydantic import BaseModel

import decisionsmith as ds


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"]
    wants_refund: bool


model = ds.model(Ticket)
if model.train("tickets.csv").switched:
    m = ds.load(model.save("models/ticket"), Ticket)
    print(m.predict("Refund my double charge"))  # Ticket(team=..., wants_refund=...)
```

## Options

| option | default | change it when |
|---|---|---|
| `schema` (second argument of `ds.load`) | rebuilt from the folder | you want your own class back |
| `verbose` (on `save`) | `True` | you don't want the `saved to ...` line |
| `device` | auto | you want `"cpu"`, `"cuda"` or `"mps"` explicitly |

## What can go wrong

| message | fix |
|---|---|
| `nothing to save yet` | train first (`model.train(...)`), or save a model you loaded |
| `nothing to save: training ran but the new model scored worse on its test split` | `report.switched` was False and the model is still the downloaded base; add more (and more varied) rows and train again |
| `... already exists and saved versions are never overwritten` | use a new version number, or `model.save()` for the next free one |
| `... is a plain Laya checkpoint` | the folder came from `ds.finetune`; use `ds.model(labels_or_class, path)` |
| `Ticket does not ask the same questions as the saved model (team.criteria: ...)` | the class differs from the saved one where the message says; load without it, or change it to match |
| `decisionsmith.json is missing ...` / `is not valid JSON` | the folder was edited or copied halfway; save the model again |

## What a loaded model carries

`m.path` is the folder it came from and `m.meta` is its `decisionsmith.json` (name, version, labels or schema,
calibration, thresholds, provenance). `m.calibration` and `m.report` are the saved calibration and evaluation.
`ds.model(labels_or_class, "models/ticket-v2")` on a saved folder is the same as `ds.load` (and checks the labels or
class match).

In a harness, a loaded model's thresholds decide when the student answers alone (cascade). Per field, the first of
these that exists wins: what `h.adapt()` fitted on the harness log, a `threshold=` you pass to `ds.harness`, the
saved thresholds, then 0.8.

Only load folders you trust. Loading never runs code from the folder (weights are safetensors, settings are JSON),
but a folder without its `encoder/` config can make the transformers library look for that config on the Hugging
Face Hub when you are online.

**Next:** use it in production with `ds.harness(m, teacher=...)` ([guide](guide.md)).
