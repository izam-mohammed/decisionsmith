"""Claude through CrewAI (ANTHROPIC_API_KEY; pip install "crewai[anthropic]")."""

from _schema import TEXTS, Ticket
from crewai import LLM

import decisionsmith as ds

h = ds.harness(Ticket, teacher=LLM(model="anthropic/claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
