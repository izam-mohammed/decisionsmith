"""OpenAI through AutoGen (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from autogen_ext.models.openai import OpenAIChatCompletionClient

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OpenAIChatCompletionClient(model="gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
