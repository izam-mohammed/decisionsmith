"""Gemini on Vertex AI through LiteLLM (gcloud auth application-default login; VERTEXAI_PROJECT, VERTEXAI_LOCATION)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ds.LLM("litellm/vertex_ai/gemini-2.5-flash"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
