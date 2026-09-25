"""OpenAI through DSPy (OPENAI_API_KEY)."""

import dspy
from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=dspy.LM("openai/gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
