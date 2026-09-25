"""OpenAI through ChatOpenAI (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from langchain_openai import ChatOpenAI

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ChatOpenAI(model="gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
