"""Llama 3.3 70B on Groq as the teacher (GROQ_API_KEY).

The teacher labels the texts; the rows train Laya (`model.train(rows)`) or go to review first.
"""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="groq/llama-3.3-70b-versatile")
for row in rows:
    print(row["text"], "->", {k: max(v, key=v.get) for k, v in row["answers"].items()})
