"""A local Ollama model through ChatOllama (ollama pull qwen3); no key, nothing leaves the machine."""

from _schema import TEXTS, Ticket
from langchain_ollama import ChatOllama

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ChatOllama(model="qwen3"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
