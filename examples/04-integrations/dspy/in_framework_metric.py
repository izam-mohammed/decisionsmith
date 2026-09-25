"""Score a program with `dspy.Evaluate` and a decisionsmith metric (per-field match with the gold labels).

The devset is synthetic; use your own labelled examples. The same metric works for any DSPy program that
outputs `team`, e.g. dspy.Predict("text -> team") on a big LLM, so both can be compared on one devset.
"""

import dspy
from _schema import Ticket

import decisionsmith as ds
from decisionsmith.integrations.dspy import DecisionModule, metric

GOLD = [
    ("I was charged twice this month, please refund me", "billing", True),
    ("The app crashes every time I log in", "technical", False),
    ("Can I get a quote for 20 seats?", "sales", False),
    ("My invoice shows the wrong company name", "billing", False),
]
devset = [dspy.Example(text=t, team=team, wants_refund=r).with_inputs("text") for t, team, r in GOLD]

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
evaluate = dspy.Evaluate(devset=devset, metric=metric("team"), num_threads=1)
print("team match: %.0f%%" % evaluate(DecisionModule(h)).score)
print("all fields: %.0f%%" % dspy.Evaluate(devset=devset, metric=metric(), num_threads=1)(DecisionModule(h)).score)
