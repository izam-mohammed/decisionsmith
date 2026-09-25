"""Claude through ChatAnthropic (ANTHROPIC_API_KEY)."""

from _schema import TEXTS, Ticket
from langchain_anthropic import ChatAnthropic

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ChatAnthropic(model="claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
