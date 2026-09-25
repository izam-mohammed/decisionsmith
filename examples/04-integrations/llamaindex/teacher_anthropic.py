"""Claude through LlamaIndex (ANTHROPIC_API_KEY; uv add llama-index-llms-anthropic)."""

from _schema import TEXTS, Ticket
from llama_index.llms.anthropic import Anthropic

import decisionsmith as ds

h = ds.harness(Ticket, teacher=Anthropic(model="claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
