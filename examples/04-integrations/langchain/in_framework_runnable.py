"""A DecisionRunnable in a chain: invoke, batch and `|` like any LangChain Runnable (OPENAI_API_KEY for the teacher)."""

from _schema import TEXTS, Ticket
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

import decisionsmith as ds
from decisionsmith.integrations.langchain import DecisionRunnable

h = ds.harness(Ticket, teacher=ChatOpenAI(model="gpt-5-mini"), student="laya")
route = RunnableLambda(lambda ticket: ticket["body"].strip()) | DecisionRunnable(h, "team")

print(route.invoke({"body": "  I was charged twice, please refund me  "}))
for text, team in zip(TEXTS, DecisionRunnable(h, "team").batch(TEXTS)):
    print(team, "<-", text)
