"""Claude through Semantic Kernel (ANTHROPIC_API_KEY; uv add "semantic-kernel[anthropic]")."""

from _schema import TEXTS, Ticket
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion

import decisionsmith as ds

h = ds.harness(Ticket, teacher=AnthropicChatCompletion(ai_model_id="claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
