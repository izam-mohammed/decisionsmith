"""Claude through DSPy (ANTHROPIC_API_KEY)."""

import dspy
from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=dspy.LM("anthropic/claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
