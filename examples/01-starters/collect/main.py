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
