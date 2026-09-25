# Evaluate a model

`model.evaluate(data)` tests a model on labelled rows it never trained on and says whether it is ready to answer
behind the harness (go / no-go), with the reasons.

**Flow:** developer flow, step 5 (define, data, golden, train, **evaluate**, save).

```python
import decisionsmith as ds

model = ds.model(["billing", "technical", "sales"])
model.train("tickets.csv")
report = model.evaluate("test.csv")  # same formats as .train(): CSV with text + label, JSONL, or rows
print(report)  # per field: decisions, accuracy, macro-F1, ECE, threshold, coverage
print(report.go, report.reasons)
```

Test on rows the model never trained on. A file with a `split` column (such as the `golden.csv` that `ds.golden`
writes) is handled for you: `model.train()` never trains on `split=test` rows, and `model.evaluate()` uses only
them. A trained model also records one-way fingerprints of its training texts (short hashes, never the texts), so
`evaluate` counts any test text it trained on, lists it first in `report.reasons` and sets `go` to false.

## What you get

| name | what it means |
|---|---|
| accuracy | share of decisions that match the label |
| macro-F1 | accuracy that counts every option equally, so a rare option can't hide |
| ECE | calibration error: how far the model's confidence is from how often it is right (0 is perfect) |
| threshold | the lowest confidence at which the model reached `target` accuracy on this data; `-` means none did |
| coverage | share of decisions at or above that threshold, so the share the student would answer in cascade |
| `report.details["fields"][name]["confusions"]` | wrong answers counted as `gold -> predicted` |
| `report.details["worst"]` | the ten most confident wrong answers, with their text |
| `report.details["ms_per_text"]` | average latency per text on this machine |

`evaluate` also stores each field's threshold on the model. `model.save()` writes it with the report, and a
harness built from the model uses it to decide when the student is sure (cascade: the student answers when sure,
else the LLM does).

## Options

| option | default | change it when |
|---|---|---|
| `target` | `0.97` | you can accept more (lower) or fewer (higher) mistakes on the answers the student gives alone |

## Go / no-go

`go` is true only when all of these hold; every failed one is in `report.reasons`:
- no test text was in the training data
- at least 100 test decisions, and every option has at least 10 test rows
- calibration error (ECE) is at most 0.10 per field
- every field has a threshold that reaches `target` accuracy

## What can go wrong

| message | fix |
|---|---|
| `only 40 test decisions; want at least 100` | label more held-out rows; the numbers are too rough to trust |
| `no confidence level reaches 97% accuracy` | train on more (or cleaner) data, or lower `target`; until then cascade sends that field to the LLM |
| `12 of 200 test texts were in the training data` | evaluate on rows the model never saw; with `ds.golden`, keep the `split` column |
| `no labelled rows to evaluate` | the file needs a `text` column and a label column per field (`label` for a labels model) |

**Next:** [save and load](save-and-load.md).
