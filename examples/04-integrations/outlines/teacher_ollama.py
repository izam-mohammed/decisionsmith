"""A local Ollama model as the teacher, with Outlines structured generation (ollama pull qwen3).

Outlines constrains the reply to the answer schema, so small local models always return valid labels.
"""

import ollama
import outlines
from _schema import TEXTS, Ticket

import decisionsmith as ds

teacher = outlines.from_ollama(ollama.Client(), "qwen3")
h = ds.harness(Ticket, teacher=teacher, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
