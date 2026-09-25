"""Fine-tune Laya on labelled data and read the report. Head-only on this small set: minutes on a laptop."""

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


report = ds.finetune(TOY, Ticket, base="laya", out="runs/toy")
print(report)
print('use it: ds.harness(Ticket, teacher=..., student="laya:%s")' % report.path)
