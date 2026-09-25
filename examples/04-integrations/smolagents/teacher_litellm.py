"""Claude through smolagents' LiteLLMModel, or any LiteLLM model id (ANTHROPIC_API_KEY)."""

from _schema import TEXTS, Ticket
from smolagents import LiteLLMModel

import decisionsmith as ds

h = ds.harness(Ticket, teacher=LiteLLMModel(model_id="anthropic/claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
