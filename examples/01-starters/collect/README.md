# Collect real samples

collect= keeps a sample of real texts in the log for the next golden dataset; h.forget deletes on request.

```python
"""collect=: keep a 10% sample of real texts plus every unsure or LLM-answered one, and forget on request."""

import decisionsmith as ds

model = ds.model(["billing", "technical", "sales"])
texts = ["I was charged twice", "The app crashes on login", "Do you have a nonprofit discount?"] * 20

with ds.harness(model, teacher="claude-haiku-4-5", collect=0.1, log="decisions.db") as h:
    results = [h.decide(t) for t in texts]
    kept = [r for r in h.log.rows("Label") if r["text"]]
    print(len(kept), "of", len(texts), "decisions kept their text")

    h.forget(results[0].id)  # a customer asked to be forgotten
    print(h.forget(older_than_days=30), "decisions older than 30 days deleted")
```

| file | what it shows |
|---|---|
| [`local_only.py`](local_only.py) | collect=0: log every decision (for status and drift) but never store a single text. |
| [`main.py`](main.py) | collect=: keep a 10% sample of real texts plus every unsure or LLM-answered one, and forget on request. |

## Run

```bash
uv add "decisionsmith[laya,anthropic]"
uv run python examples/01-starters/collect/local_only.py
uv run python examples/01-starters/collect/main.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
