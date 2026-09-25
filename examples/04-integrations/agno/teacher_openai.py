"""OpenAI through Agno (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from agno.models.openai import OpenAIChat

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OpenAIChat(id="gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
