"""The schema the examples in this folder decide, and a few synthetic labelled tickets."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

import decisionsmith as ds


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


ROWS = [
    ("I was charged twice this month, please refund me", "billing", True),
    ("The app crashes every time I log in", "technical", False),
    ("Can I get a quote for 20 seats?", "sales", False),
    ("My invoice shows the wrong company name", "billing", False),
    ("Sync has been broken since the last update", "technical", False),
    ("Do you offer a discount for charities?", "sales", False),
    ("Cancel my plan and give me my money back", "billing", True),
    ("", None, None),
]
