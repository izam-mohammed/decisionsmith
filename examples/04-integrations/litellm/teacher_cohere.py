"""Cohere Command A through LiteLLM (COHERE_API_KEY)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ds.LLM("litellm/cohere_chat/command-a-03-2025"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
