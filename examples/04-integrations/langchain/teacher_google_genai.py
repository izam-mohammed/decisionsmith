"""Gemini (Google AI Studio) through ChatGoogleGenerativeAI (GOOGLE_API_KEY)."""

from _schema import TEXTS, Ticket
from langchain_google_genai import ChatGoogleGenerativeAI

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=ChatGoogleGenerativeAI(model="gemini-2.5-flash"))
print(len(rows), "rows labelled; train Laya on them with model.train(rows)")
