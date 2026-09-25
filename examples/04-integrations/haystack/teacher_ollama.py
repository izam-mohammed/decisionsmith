"""A local Ollama model through Haystack (ollama pull qwen3; pip install ollama-haystack)."""

from _schema import TEXTS, Ticket
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OllamaChatGenerator(model="qwen3"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
