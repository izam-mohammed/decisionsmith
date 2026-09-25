"""Mistral through ChatMistralAI (MISTRAL_API_KEY)."""

from _schema import TEXTS, Ticket
from langchain_mistralai import ChatMistralAI

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=ChatMistralAI(model="mistral-small-latest"))
print(len(rows), "rows labelled; train Laya on them with model.train(rows)")
