"""A `dspy.Module` backed by the harness: use it wherever a DSPy program goes, or inside a bigger one.

program(text=...) returns a dspy.Prediction with one output per field (team, wants_refund).
"""

import asyncio

from _schema import TEXTS, Ticket

import decisionsmith as ds
from decisionsmith.integrations.dspy import DecisionModule

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
route = DecisionModule(h)

for text in TEXTS[:3]:
    p = route(text=text)
    print(p.team, p.wants_refund, "<-", text)
print(asyncio.run(route.acall(text=TEXTS[3])))
