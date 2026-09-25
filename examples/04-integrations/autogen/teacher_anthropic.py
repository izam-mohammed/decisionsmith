"""Claude through AutoGen (ANTHROPIC_API_KEY; uv add "autogen-ext[anthropic]")."""

from _schema import TEXTS, Ticket
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

import decisionsmith as ds

h = ds.harness(Ticket, teacher=AnthropicChatCompletionClient(model="claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
