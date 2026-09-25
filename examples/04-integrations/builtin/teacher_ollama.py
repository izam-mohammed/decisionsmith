"""A local Ollama model as the teacher: no key, nothing leaves the machine (ollama pull qwen3).

The teacher labels the texts; the rows train Laya (`model.train(rows)`) or go to review first.
"""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="ollama/qwen3")
for row in rows:
    print(row["text"], "->", {k: max(v, key=v.get) for k, v in row["answers"].items()})
