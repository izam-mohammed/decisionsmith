"""Groq through LiteLLM (GROQ_API_KEY)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="litellm/groq/llama-3.3-70b-versatile")
print(len(rows), "rows labelled; LiteLLM reports the cost of each call in the log")
