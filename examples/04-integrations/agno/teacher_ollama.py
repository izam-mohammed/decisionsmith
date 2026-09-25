"""A local Ollama model through Agno (ollama pull qwen3; pip install ollama)."""

from _schema import TEXTS, Ticket
from agno.models.ollama import Ollama

import decisionsmith as ds

h = ds.harness(Ticket, teacher=Ollama(id="qwen3"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
