"""A local Ollama model through Semantic Kernel (ollama pull qwen3; uv add "semantic-kernel[ollama]")."""

from _schema import TEXTS, Ticket
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OllamaChatCompletion(ai_model_id="qwen3"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
