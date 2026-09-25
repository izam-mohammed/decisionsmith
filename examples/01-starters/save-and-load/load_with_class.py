"""Save a model with several fields and load it back with your own Pydantic class (it must match the saved one)."""

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


rows = list(csv.DictReader(TOY.open(encoding="utf-8")))
model = ds.model(Ticket)
model.train(rows[:220])
report = model.evaluate(rows[220:])
print("go" if report.go else "no-go:", *report.reasons, sep="\n  ")

path = model.save()  # models/ticket-v1
m = ds.load(path, Ticket)  # your class back, checked against the saved schema
print(m.predict("Please refund the double charge on my card"))

rebuilt = ds.load(path)  # no class: one is rebuilt from decisionsmith.json
print(type(rebuilt.predict("The app crashes on login")).__name__)
