# Collect real samples in production

`collect=` on the harness decides which real texts stay in the decision log, so the next
[golden dataset](golden.md) is built from what your users actually send.

**Flow:** production flow. The samples feed step 3 of the developer flow.

```python
import decisionsmith as ds

model = ds.model(["billing", "technical", "sales"])  # in production: ds.load("models/ticket-v2")
h = ds.harness(model, teacher="claude-haiku-4-5", collect=0.1)
r = h.decide("I was charged twice")
print(r.value, r.sure)
h.forget(r.id)  # delete one decision on request
h.close()
```

Every decision is always logged (answers, confidences, which engine answered), because `status()` and `adapt()`
need them. `collect` only decides whether the **text** is stored with it.

| `collect` | keeps the text of |
|---|---|
| `1.0` (default) | every decision |
| `0.1` | a 10% share, plus every decision where the student was unsure or disagreed with the LLM |
| `0` | none: nothing the user wrote is stored |

Unsure and disputed texts are the ones worth labelling, so they are always kept when `collect` is above 0. What
that means per mode:

| mode | kept besides the share |
|---|---|
| `cascade` | every text the student was unsure of (those are the ones the LLM answered) and every audited one it got wrong |
| `shadow` | every text the student was unsure of or answered differently from the LLM |
| `teacher` (no student) | nothing: only the share |
| `student` | every text the student was unsure of |

The share is picked from each decision's id (a stable hash, not a fresh random draw), so the same decision is always
either in or out.

## Privacy, retention and deletion

- Text is stored only for collected rows, in the SQLite file you pass as `log=` (default `decisions.db`), on your
  machine or server. Nothing is sent anywhere by the log.
- Delete one decision (text, answers and labels): `h.forget(decision_id)`. The log overwrites deleted bytes and
  truncates its write-ahead file, so the text is gone from the database files, not only hidden. If another program
  is reading the same log at that moment, the write-ahead file can't be truncated yet: `forget` warns, and the text
  may stay there until the next checkpoint (close the other reader and call `forget` again).
- Retention: `h.forget(older_than_days=30)` deletes everything logged more than 30 days ago; run it on a schedule.
- `forget` only reaches the log. Copies made earlier stay where they are: an export (`h.export`), a `golden.csv`,
  and anything a model was trained on. Delete or regenerate those too.
- A trained model stores short one-way fingerprints of its training texts (see [evaluate](evaluate.md)). They
  don't contain the text, but someone holding the model can check whether a text they can guess exactly was
  trained on. Keep models trained on personal data as private as the data.
- Personal information: if texts can hold it, use a low `collect` (or `0`), and follow the privacy law that applies
  to you (in Australia, the Privacy Act 1988 and its Australian Privacy Principles).

## Options

| option | default | change it when |
|---|---|---|
| `collect` | `1.0` | you want fewer texts stored (privacy, disk) |
| `log` | `"decisions.db"` | another path; `None` logs nothing (no status, no golden) |

## What can go wrong

| message | fix |
|---|---|
| `collect must be in [0, 1]` | use a share such as `0.1` |
| `give a decision id or older_than_days` | `h.forget(r.id)` or `h.forget(older_than_days=30)` |
| `no decision with id ...` | it was already forgotten, or it is in another log file |
| `... their text may stay in its write-ahead file` | another connection was reading the log; close it and run `forget` again |
| `older_than_days must be 0 or more` | pass a number of days, such as `30` |
| `exported 0 rows: the logged decisions have no text` | raise `collect=` so texts are kept for training |

**Next:** turn the samples into a [golden dataset](golden.md).
