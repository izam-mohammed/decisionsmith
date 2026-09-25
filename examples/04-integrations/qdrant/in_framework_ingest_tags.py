"""Tag points with decisions as you upsert them into Qdrant, then filter on the tags in the query."""

from _schema import DOCS, Ticket, embed
from qdrant_client import QdrantClient, models

import decisionsmith as ds
from decisionsmith.integrations.qdrant import points

client = QdrantClient(":memory:")  # or QdrantClient(url="http://localhost:6333")
client.create_collection("tickets", vectors_config=models.VectorParams(size=32, distance=models.Distance.COSINE))

model = ds.model(Ticket)  # base Laya; train it on your data for real use
client.upsert("tickets", points=points(model, list(range(len(DOCS))), [embed(d) for d in DOCS], DOCS))

print(client.retrieve("tickets", [0])[0].payload)
team_filter = models.Filter(must=[models.FieldCondition(key="team", match=models.MatchValue(value="billing"))])
hits = client.query_points("tickets", query=embed("refund"), query_filter=team_filter, limit=2).points
print("billing team hits:", [h.payload["text"] for h in hits])
