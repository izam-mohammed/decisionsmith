"""Fully local: an Ollama model teaches, Laya learns. No data leaves the machine.

ollama pull qwen3 && python main.py
"""

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


texts = [row["text"] for row in csv.DictReader(TOY.read_text().splitlines())][:20]

h = ds.harness(Ticket, teacher="ollama/qwen3", student="laya", mode="shadow")
for text in texts:
    print(h.decide(text).value)
print(h.status())
