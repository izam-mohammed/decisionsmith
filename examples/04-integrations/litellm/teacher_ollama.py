"""A local Ollama model through LiteLLM (ollama pull qwen3)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ds.LLM("litellm/ollama_chat/qwen3"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
