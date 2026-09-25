"""A local Ollama model through Pydantic AI (ollama pull qwen3); no key, nothing leaves the machine."""

from _schema import TEXTS, Ticket
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

import decisionsmith as ds

model = OpenAIChatModel("qwen3", provider=OllamaProvider(base_url="http://localhost:11434/v1"))
h = ds.harness(Ticket, teacher=model, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
