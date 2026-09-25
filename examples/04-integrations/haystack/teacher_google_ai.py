"""Gemini through Haystack's Google GenAI generator (GEMINI_API_KEY or GOOGLE_API_KEY; uv add
google-genai-haystack)."""

from _schema import TEXTS, Ticket
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=GoogleGenAIChatGenerator(model="gemini-2.5-flash"))
print(len(rows), "rows labelled")
