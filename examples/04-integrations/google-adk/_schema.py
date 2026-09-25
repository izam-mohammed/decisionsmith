"""The schemas the examples in this folder decide, and a few synthetic texts."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

import decisionsmith as ds


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


class Injection(BaseModel):
    is_attack: bool = Field(description="Is this trying to override the assistant's instructions?")


TEXTS = [
    "I was charged twice this month, please refund me",
    "The app crashes every time I log in",
    "Can I get a quote for 20 seats?",
]
