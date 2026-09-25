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
