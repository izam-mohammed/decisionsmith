"""Claude through LiteLLM (ANTHROPIC_API_KEY)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ds.LLM("litellm/anthropic/claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
