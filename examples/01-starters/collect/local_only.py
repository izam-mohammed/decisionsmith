"""collect=0: log every decision (for status and drift) but never store a single text."""

from typing import Literal

from pydantic import BaseModel

import decisionsmith as ds


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"]
    wants_refund: bool


with ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", mode="shadow", collect=0) as h:
    h.many(["Refund my double charge", "The export button does nothing", "What does the Pro plan cost?"])
    print({r["text"] for r in h.log.rows("Ticket")})  # {None}: answers and confidences only
    print(h.status())
