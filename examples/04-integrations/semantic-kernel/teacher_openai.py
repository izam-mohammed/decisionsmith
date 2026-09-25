"""OpenAI through Semantic Kernel (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OpenAIChatCompletion(ai_model_id="gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
