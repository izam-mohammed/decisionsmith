"""Fireworks AI as the teacher (FIREWORKS_API_KEY).

The teacher labels the texts; the rows train Laya (`model.train(rows)`) or go to review first.
"""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct")
for row in rows:
    print(row["text"], "->", {k: max(v, key=v.get) for k, v in row["answers"].items()})
