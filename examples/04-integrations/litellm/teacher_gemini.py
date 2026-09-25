"""Gemini (Google AI Studio) through LiteLLM (GEMINI_API_KEY)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="litellm/gemini/gemini-2.5-flash")
print(len(rows), "rows labelled; LiteLLM reports the cost of each call in the log")
