"""Gemini through Pydantic AI (GOOGLE_API_KEY)."""

from _schema import TEXTS, Ticket
from pydantic_ai.models.google import GoogleModel

import decisionsmith as ds

h = ds.harness(Ticket, teacher=GoogleModel("gemini-2.5-flash"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
