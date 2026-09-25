"""LM Studio as the teacher: load a model, start the local server.

The teacher labels the texts; the rows train Laya (`model.train(rows)`) or go to review first.
"""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=ds.LLM("qwen/qwen3-8b", url="http://localhost:1234/v1"))
for row in rows:
    print(row["text"], "->", {k: max(v, key=v.get) for k, v in row["answers"].items()})
