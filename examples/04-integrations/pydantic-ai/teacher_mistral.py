"""Mistral through Pydantic AI (MISTRAL_API_KEY)."""

from _schema import TEXTS, Ticket
from pydantic_ai.models.mistral import MistralModel

import decisionsmith as ds

h = ds.harness(Ticket, teacher=MistralModel("mistral-small-latest"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
