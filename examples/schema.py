from typing import Annotated, Literal

from pydantic import BaseModel, Field

import decisionsmith as ds


class Ticket(BaseModel):
    """A customer support ticket."""

    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(
            billing="payments, invoices, refunds", technical="bugs, outages, errors", sales="pricing, plans, buying"
        ),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")
