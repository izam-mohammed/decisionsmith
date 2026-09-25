"""OpenAI through CrewAI (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from crewai import LLM

import decisionsmith as ds

h = ds.harness(Ticket, teacher=LLM(model="gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
