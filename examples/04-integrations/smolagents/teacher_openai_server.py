"""OpenAI (or any OpenAI-compatible server: pass api_base=) through smolagents' OpenAIServerModel (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from smolagents import OpenAIServerModel

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OpenAIServerModel(model_id="gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
