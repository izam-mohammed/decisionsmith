"""The schema the examples in this folder decide, and a few synthetic tickets."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

import decisionsmith as ds


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


TEXTS = [
    "I was charged twice this month, please refund me",
    "The app crashes every time I log in",
    "Can I get a quote for 20 seats?",
    "My invoice shows the wrong company name",
    "Sync has been broken since the last update",
    "Do you offer a discount for charities?",
]


class Leak(BaseModel):
    leaks_data: bool = Field(description="Does this message reveal a password, card number or other private data?")
