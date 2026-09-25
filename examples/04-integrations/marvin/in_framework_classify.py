"""Marvin's `classify()` answered by Laya instead of an LLM: same call, no API key, runs on your machine.

Swap `from marvin import classify` for the import below; pass `model=` to use a model you trained.
"""

import enum

from decisionsmith.integrations.marvin import classify


class Team(enum.Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"


print(classify("I was charged twice this month", ["billing", "technical", "sales"]))
print(classify("The app crashes every time I log in", Team, instructions="Which team should handle this?"))
