"""Gemini through Agno (GOOGLE_API_KEY; pip install google-genai)."""

from _schema import TEXTS, Ticket
from agno.models.google import Gemini

import decisionsmith as ds

h = ds.harness(Ticket, teacher=Gemini(id="gemini-2.5-flash"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
