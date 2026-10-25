"""The schemas the examples in this folder decide, a few synthetic documents, and a toy embedding.

`embed` is a stand-in so the examples run offline: use your real embedding model instead.
"""

import hashlib
import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field

import decisionsmith as ds


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


DOCS = [
    "Refunds are paid back to the original card within 5 business days.",
    "To request a refund, reply to your receipt email.",
    "Our office dog is called Biscuit.",
    "If the app crashes on login, update to the latest version.",
    "The team plan costs less per seat above 20 seats.",
]


def embed(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    return vec
