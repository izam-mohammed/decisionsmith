"""A local Ollama model through AutoGen (ollama pull qwen3; uv add "autogen-ext[ollama]")."""

from _schema import TEXTS, Ticket
from autogen_ext.models.ollama import OllamaChatCompletionClient

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OllamaChatCompletionClient(model="qwen3"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
