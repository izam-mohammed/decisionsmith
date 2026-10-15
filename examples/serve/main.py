"""Save a model, serve it, and ask it for a decision over HTTP: the smallest production server."""

import csv
from pathlib import Path

from fastapi.testclient import TestClient

import decisionsmith as ds
from decisionsmith.serve import app

TOY = Path(__file__).parents[1] / "data" / "toy.csv"
rows = [(r["text"], r["team"]) for r in csv.DictReader(TOY.open(encoding="utf-8"))]

model = ds.model(["billing", "technical", "sales"])
model.train(rows[:220])
path = model.save("models/team", verbose=False)  # models/team-v1

# For real: `uv run decisionsmith serve models/team-v1` (Uvicorn on 127.0.0.1:8000).
# Here TestClient calls the same app in-process, with no port.
with TestClient(app(path)) as client:
    r = client.post("/v1/decide", json={"text": "I was charged twice this month"})
    print(r.json())  # {"id": ..., "value": {"label": ...}, "source": {"label": "student"}, "sure": ...}
