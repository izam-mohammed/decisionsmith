"""Groq through ChatGroq (GROQ_API_KEY)."""

from _schema import TEXTS, Ticket
from langchain_groq import ChatGroq

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=ChatGroq(model="llama-3.3-70b-versatile"))
print(len(rows), "rows labelled; train Laya on them with model.train(rows)")
