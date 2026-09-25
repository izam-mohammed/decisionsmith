"""Gemini through CrewAI (GEMINI_API_KEY; uv add "crewai[google-genai]")."""

from _schema import TEXTS, Ticket
from crewai import LLM

import decisionsmith as ds

h = ds.harness(Ticket, teacher=LLM(model="gemini/gemini-2.5-flash"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
