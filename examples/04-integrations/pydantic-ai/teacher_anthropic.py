"""Claude through Pydantic AI (ANTHROPIC_API_KEY)."""

from _schema import TEXTS, Ticket
from pydantic_ai.models.anthropic import AnthropicModel

import decisionsmith as ds

h = ds.harness(Ticket, teacher=AnthropicModel("claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
