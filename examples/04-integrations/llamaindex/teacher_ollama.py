"""A local Ollama model through LlamaIndex (ollama pull qwen3; pip install llama-index-llms-ollama)."""

from _schema import TEXTS, Ticket
from llama_index.llms.ollama import Ollama

import decisionsmith as ds

h = ds.harness(Ticket, teacher=Ollama(model="qwen3", request_timeout=120.0), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
