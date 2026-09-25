"""A local Ollama model through DSPy (ollama pull qwen3)."""

import dspy
from _schema import TEXTS, Ticket

import decisionsmith as ds

lm = dspy.LM("ollama_chat/qwen3", api_base="http://localhost:11434", api_key="")
h = ds.harness(Ticket, teacher=lm, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
