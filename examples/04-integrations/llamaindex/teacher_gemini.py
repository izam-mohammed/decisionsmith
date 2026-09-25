"""Gemini through LlamaIndex's Google GenAI LLM (GOOGLE_API_KEY; uv add llama-index-llms-google-genai).

Passing max_tokens and context_window skips the model metadata lookup at construction.
"""

from _schema import TEXTS, Ticket
from llama_index.llms.google_genai import GoogleGenAI

import decisionsmith as ds

llm = GoogleGenAI(model="gemini-2.5-flash", max_tokens=1024, context_window=1_000_000)
model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=llm)
print(len(rows), "rows labelled by", llm.model)
