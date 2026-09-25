"""Gemini through DSPy (GEMINI_API_KEY)."""

import dspy
from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=dspy.LM("gemini/gemini-2.5-flash"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
