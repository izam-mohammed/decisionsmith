"""A production-style server: an LLM behind the model for unsure cases, collect=0.1, an API key, a Jev client, labels,
status and metrics."""

import csv
import secrets
from pathlib import Path
from typing import Annotated, Literal

from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

import decisionsmith as ds
from decisionsmith.serve import app

TOY = Path(__file__).parents[1] / "data" / "toy.csv"


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"],
        ds.Options(billing="payments, invoices, refunds", technical="bugs, outages", sales="pricing, plans"),
    ] = Field(description="Which team should handle this ticket?")
    wants_refund: bool = Field(description="Does the customer ask for their money back?")


model = ds.model(Ticket)
model.train(list(csv.DictReader(TOY.open(encoding="utf-8")))[:220])
path = model.save("models/ticket", verbose=False)

# The CLI reads the key from DECISIONSMITH_API_KEY; in code, pass it (from your secret manager, not the source).
key = secrets.token_hex(16)
auth = {"Authorization": "Bearer " + key}

# Same as: DECISIONSMITH_API_KEY=... decisionsmith serve models/ticket-v1 --teacher claude-haiku-4-5 --collect 0.1 \
#   --log served.db
api = app(path, teacher="claude-haiku-4-5", collect=0.1, log="served.db", api_key=key)
with TestClient(api) as client:
    print(client.get("/health").json()["fields"])  # no key needed for the health check
    print(client.post("/v1/decide", json={"text": "hi"}).json()["error"]["fix"])  # no key: 401 with the fix

    r = client.post("/v1/decide", json={"text": "Please refund the double charge"}, headers=auth).json()
    print(r["value"], r["source"], "sure" if r["sure"] else "unsure")

    # A Jev-style client: same request shape as TypeSafe's /v1/systemone, only the base URL changes.
    jev = client.post(
        "/v1/systemone",
        json={"state": "The app crashes on login", "questions": {"team": {"type": "choice"}}},
        headers=auth,
    ).json()
    print(jev["answers"]["team"]["choice"], jev["answers"]["team"]["confidence"], jev["usage"])

    # A human fixes a decision; status() and the next golden dataset use it.
    client.post("/v1/label", json={"id": r["id"], "labels": {"team": "billing"}}, headers=auth)
    status = client.get("/v1/status", headers=auth).json()
    print({name: f["advice"] for name, f in status["fields"].items()})

    metrics = client.get("/metrics", headers=auth).text
    print("\n".join(line for line in metrics.splitlines() if line.startswith("decisionsmith_decisions_total")))
