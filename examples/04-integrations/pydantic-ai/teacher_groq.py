"""Groq through Pydantic AI (GROQ_API_KEY)."""

from _schema import TEXTS, Ticket
from pydantic_ai.models.groq import GroqModel

import decisionsmith as ds

h = ds.harness(Ticket, teacher=GroqModel("llama-3.3-70b-versatile"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
