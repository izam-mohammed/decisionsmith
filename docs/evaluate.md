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

`tickets.csv` and `test.csv` need a `text` column and a `label` column (for a labels model) or one column per field
(for a class, e.g. `team` and `wants_refund`).

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
| threshold | the lowest confidence at which the model reached `target` accuracy; `-` means none did |
| coverage | share of decisions at or above that threshold, so the share the student would answer in cascade |
| accuracy_when_sure | accuracy on the decisions at or above the threshold |

The threshold is picked on one half of the rows and coverage and accuracy_when_sure are measured on the other half
(the halves come from a hash of each text), so the numbers aren't graded on the rows that chose the threshold.
With a single row both come from that row.
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

<a id="same-text"></a>
## What counts as the same text

Two texts are the same when they match after Unicode NFKC normalisation, case folding, dropping punctuation and
collapsing spaces: `"Refund, please!"` and `"refund please"` are the same; `"refund"` and `"refunds"` are not. A
model retrained from a saved model keeps the fingerprints of both rounds of training. The fingerprints are 16 hex
characters of a SHA-256 hash; see [collect](collect.md) for what that means for privacy.

**Next:** [save and load](save-and-load.md).
