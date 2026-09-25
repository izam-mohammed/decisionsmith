# Golden dataset

`ds.golden(source, teacher)` picks the texts most worth labelling, has your main LLM label them, and writes
`golden.csv` for you to review before training.

**Flow:** developer flow, step 3 (define, data, **golden**, train, evaluate, save). Its input is often the real
samples the production flow collected.

<!-- no-test: needs a decisions.db from a harness; examples/01-starters/golden runs the same flow offline -->

```python
import decisionsmith as ds

rows = ds.golden("decisions.db", teacher="claude-opus-4-5", n=500, schema=["billing", "technical", "sales"])
# writes golden.csv: id, text, one column per field, split (train or test), labelled_by

model = ds.model(["billing", "technical", "sales"])
model.train("golden.csv")  # skips split=test rows
print(model.evaluate("golden.csv"))  # uses only split=test rows
```

## Where the texts come from

| `source` | what |
|---|---|
| a harness, or its log (`"decisions.db"`) | real samples from production, with the student's and teacher's answers already logged |
| a file: `.txt` (one text per line), `.csv` (a `text` column), `.jsonl` (`text` fields) | texts you have |
| a list of texts | the same, from Python |

## Strategies

| `strategy` | picks | needs |
|---|---|---|
| `uncertain` (default) | the texts the student was least sure about | student confidences: a harness log, or `schema=ds.model(...)` to score the texts |
| `disagree` | texts where the student and the teacher answered differently first, then the least sure | a harness log (for texts without a teacher answer it falls back to `uncertain`) |
| `diverse` | an even spread across the student's predicted labels and across text lengths | nothing (with no student it spreads by length only) |
| `random` | a random sample | nothing |

Duplicates (same text, ignoring case and spacing) are dropped first.

## Options

| option | default | change it when |
|---|---|---|
| `n` | `500` | you want more or fewer rows labelled (each one is an LLM call) |
| `schema` | from the harness | the source is a file or a list: pass labels, your class, or a `ds.model` (which also scores the texts) |
| `test` | `0.2` | the share of rows marked `split=test`, held out for `evaluate` |
| `out` | `"golden.csv"` | another path, or `None` to only return the rows |

Review the CSV before you train: fix a wrong label in place, or delete the row. Rows the LLM could not label are
left out and counted in the summary line.

## From the command line

```bash
uv run decisionsmith golden --log decisions.db --labels billing,technical,sales --teacher claude-opus-4-5 -n 500
uv run decisionsmith golden texts.txt --schema app.py:Ticket --teacher claude-haiku-4-5 --strategy random
uv run decisionsmith golden texts.txt --labels billing,technical,sales --teacher claude-haiku-4-5 --score --strategy uncertain
```

`--score` scores a texts file with `--base` (default `laya`) so `uncertain` and `diverse` have confidences to use.

## What can go wrong

| message | fix |
|---|---|
| `strategy='uncertain' ranks texts by the student's confidence, and there is none here` | use a harness log, pass `schema=ds.model(...)` (CLI: `--score`), or `strategy='random'` |
| `ds.golden needs to know the answers` | pass `schema=`: a list of labels, your class, or a `ds.model` |
| `... has no Ticket decisions with text` | the harness ran with `collect=0`, so no text was kept; see [collect](collect.md) |
| `no log at decisions.db` | run a harness with `log="decisions.db"` first, or point at the right file |

**Next:** [evaluate](evaluate.md) the model you train on it.
