"""OpenRouter through LiteLLM (OPENROUTER_API_KEY)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ds.LLM("litellm/openrouter/openai/gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
