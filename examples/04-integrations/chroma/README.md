# Chroma

Tag documents with decisions at ingest (`add` / `tag`, then filter with `where=`), and drop off-topic hits after retrieval (`filter_results`). Uses Chroma's in-memory client and a toy embedding here.

```python
"""Tag documents with decisions as you add them to Chroma, then filter on the tags with `where=`."""

import chromadb
from _schema import DOCS, Ticket, embed

import decisionsmith as ds
from decisionsmith.integrations.chroma import add

client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
tickets = client.get_or_create_collection("tickets", embedding_function=None)

model = ds.model(Ticket)  # base Laya; train it on your data for real use
ids = [str(i) for i in range(len(DOCS))]
add(tickets, model, ids=ids, documents=DOCS, embeddings=[embed(d) for d in DOCS])

print(tickets.get(ids=["0"])["metadatas"][0])
billing = tickets.query(query_embeddings=[embed("refund")], n_results=2, where={"team": "billing"})
print("billing hits:", billing["documents"][0])
```

| file | what it shows |
|---|---|
| [`in_framework_ingest_tags.py`](in_framework_ingest_tags.py) | Tag documents with decisions as you add them to Chroma, then filter on the tags with `where=`. |
| [`in_framework_rag_filter.py`](in_framework_rag_filter.py) | A RAG relevance filter: after Chroma retrieves, the model drops the passages that don't help the question. |

## Run

```bash
uv add "decisionsmith[chroma,laya]"
uv run python examples/04-integrations/chroma/in_framework_ingest_tags.py
uv run python examples/04-integrations/chroma/in_framework_rag_filter.py
```

No keys yet? `DS_OFFLINE=1` swaps every LLM for a local stand-in so you can walk through the flow.

---

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations. decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
