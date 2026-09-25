# Qdrant

Tag points with decisions at ingest (`points` / `tag`, then filter with a payload `Filter`), and drop off-topic hits after retrieval (`filter_points`). Uses Qdrant's in-memory mode and a toy embedding here.

```python
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
billing = models.Filter(must=[models.FieldCondition(key="team", match=models.MatchValue(value="billing"))])
hits = client.query_points("tickets", query=embed("refund"), query_filter=billing, limit=2).points
print("billing hits:", [h.payload["text"] for h in hits])
```

| file | what it shows |
|---|---|
| [`in_framework_ingest_tags.py`](in_framework_ingest_tags.py) | Tag points with decisions as you upsert them into Qdrant, then filter on the tags in the query. |
| [`in_framework_rag_filter.py`](in_framework_rag_filter.py) | A RAG relevance filter: after Qdrant retrieves, the model drops the points that don't help the question. |

## Run

```bash
uv add "decisionsmith[qdrant,laya]"
uv run python examples/04-integrations/qdrant/in_framework_ingest_tags.py
uv run python examples/04-integrations/qdrant/in_framework_rag_filter.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
