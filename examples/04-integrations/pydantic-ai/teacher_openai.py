"""OpenAI through Pydantic AI (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from pydantic_ai.models.openai import OpenAIChatModel

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OpenAIChatModel("gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
