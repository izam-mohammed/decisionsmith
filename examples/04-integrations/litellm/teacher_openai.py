"""OpenAI through LiteLLM (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="litellm/gpt-5-mini")
print(len(rows), "rows labelled; LiteLLM reports the cost of each call in the log")
