"""Groq through LlamaIndex (GROQ_API_KEY; uv add llama-index-llms-groq)."""

from _schema import TEXTS, Ticket
from llama_index.llms.groq import Groq

import decisionsmith as ds

h = ds.harness(Ticket, teacher=Groq(model="llama-3.3-70b-versatile"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
