"""Claude through Agno (ANTHROPIC_API_KEY; pip install anthropic)."""

from _schema import TEXTS, Ticket
from agno.models.anthropic import Claude

import decisionsmith as ds

h = ds.harness(Ticket, teacher=Claude(id="claude-haiku-4-5"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
