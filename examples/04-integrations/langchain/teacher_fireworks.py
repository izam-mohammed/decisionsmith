"""Fireworks through ChatFireworks (FIREWORKS_API_KEY)."""

from _schema import TEXTS, Ticket
from langchain_fireworks import ChatFireworks

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=ChatFireworks(model="accounts/fireworks/models/llama-v3p3-70b-instruct"))
print(len(rows), "rows labelled; train Laya on them with model.train(rows)")
