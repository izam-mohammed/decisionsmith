"""A local Ollama model through CrewAI (ollama pull qwen3)."""

from _schema import TEXTS, Ticket
from crewai import LLM

import decisionsmith as ds

h = ds.harness(
    Ticket, teacher=LLM(model="ollama/qwen3", base_url="http://localhost:11434"), student="laya", mode="shadow"
)
h.many(TEXTS)
print(h.status())
