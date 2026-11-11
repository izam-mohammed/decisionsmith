"""A RAG relevance filter: after Qdrant retrieves, the model drops the points that don't help the question.

Train the judge on your own question/passage pairs first; base Laya is only a starting point.
"""

from _schema import DOCS, embed
from qdrant_client import QdrantClient, models

import decisionsmith as ds
from decisionsmith.integrations.qdrant import filter_points

client = QdrantClient(":memory:")
client.create_collection("help-centre", vectors_config=models.VectorParams(size=32, distance=models.Distance.COSINE))
client.upsert(
    "help-centre",
    points=[models.PointStruct(id=i, vector=embed(d), payload={"text": d}) for i, d in enumerate(DOCS)],
)

judge = ds.model(["relevant", "off-topic"], question="Does the document help answer the query?")
question = "How do I get a refund?"
hits = client.query_points("help-centre", query=embed(question), limit=4)
kept = filter_points(judge, hits, keep=["relevant"], query=question)
print("retrieved:", len(hits.points), "kept:", len(kept))
for p in kept:
    print(" -", p.payload["text"])
