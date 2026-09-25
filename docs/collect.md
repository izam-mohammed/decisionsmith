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
| `0.1` | a random 10%, plus every decision the student was unsure of and every one the LLM answered |
| `0` | none: nothing the user wrote is stored |

Unsure and LLM-answered texts are the ones worth labelling, so they are always kept when `collect` is above 0.

## Privacy, retention and deletion

- Text is stored only for collected rows, in the SQLite file you pass as `log=` (default `decisions.db`), on your
  machine or server. Nothing is sent anywhere by the log.
- Delete one decision (text, answers and labels): `h.forget(decision_id)`.
- Retention: `h.forget(older_than_days=30)` deletes everything logged more than 30 days ago; run it on a schedule.
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

**Next:** turn the samples into a [golden dataset](golden.md).
