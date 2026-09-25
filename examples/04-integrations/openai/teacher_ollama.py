"""A local Ollama model through the OpenAI SDK (ollama pull qwen3); no key, nothing leaves the machine."""

from _schema import TEXTS, Ticket
from openai import OpenAI

import decisionsmith as ds
from decisionsmith.integrations.openai import teacher

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
h = ds.harness(Ticket, teacher=teacher(client, "qwen3"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
