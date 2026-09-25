"""Start with an LLM teacher, measure Laya in shadow mode, calibrate it, and see where each field stands."""

import csv
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

import decisionsmith as ds

TOY = Path(__file__).parents[2] / "data" / "toy.csv"


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


texts = [row["text"] for row in csv.DictReader(TOY.read_text().splitlines())]

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya", mode="shadow")
print(h(texts[2]), "<-", texts[2])
h.many(texts)
print(h.status())
print(h.adapt())
