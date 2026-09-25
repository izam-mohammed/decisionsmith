"""Claude through Haystack (ANTHROPIC_API_KEY; pip install anthropic-haystack)."""

from _schema import TEXTS, Ticket
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

import decisionsmith as ds

h = ds.harness(Ticket, teacher=AnthropicChatGenerator(model="claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
