"""Fully local: an Ollama model as the teacher, Laya as the student. No data leaves the machine.

ollama pull qwen3 && python examples/05_local_only.py
"""

from _common import toy_rows
from schema import Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher="ollama/qwen3", student="laya", mode="shadow")
for row in toy_rows()[:20]:
    print(h.decide(row["text"]).value)
print(h.status())
