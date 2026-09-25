# Collect real samples

collect= keeps a sample of real texts in the log for the next golden dataset; h.forget deletes on request.

```python
"""collect=: keep a 10% sample of real texts (plus every unsure or disputed one) and forget on request."""

import decisionsmith as ds

texts = ["I was charged twice %d" % i for i in range(100)]

model = ds.model(["billing", "technical", "sales"])
# mode="teacher": the LLM answers, so only the random 10% share keeps its text; every decision is still logged.
with ds.harness(model, teacher="claude-haiku-4-5", mode="teacher", collect=0.1) as h:
    results = [h.decide(t) for t in texts]
    rows = h.log.rows("Label")
    print(len(rows), "decisions logged,", sum(bool(r["text"]) for r in rows), "kept their text")

    h.forget(results[0].id)  # a customer asked to be forgotten: text, answers and labels are deleted
    print(h.forget(older_than_days=30), "decisions older than 30 days deleted")
```

| file | what it shows |
|---|---|
| [`local_only.py`](local_only.py) | collect=0: log every decision (for status and drift) but never store a single text. |
| [`main.py`](main.py) | collect=: keep a 10% sample of real texts (plus every unsure or disputed one) and forget on request. |

## Run

```bash
uv add "decisionsmith[laya,anthropic]"
uv run python examples/01-starters/collect/local_only.py
uv run python examples/01-starters/collect/main.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
