"""OpenAI through Haystack's OpenAIChatGenerator (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from haystack.components.generators.chat import OpenAIChatGenerator

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OpenAIChatGenerator(model="gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
