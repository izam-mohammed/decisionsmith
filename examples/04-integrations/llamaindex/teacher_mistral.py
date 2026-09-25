"""Mistral through LlamaIndex (MISTRAL_API_KEY; pip install llama-index-llms-mistralai)."""

from _schema import TEXTS, Ticket
from llama_index.llms.mistralai import MistralAI

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=MistralAI(model="mistral-small-latest"))
print(len(rows), "rows labelled")
