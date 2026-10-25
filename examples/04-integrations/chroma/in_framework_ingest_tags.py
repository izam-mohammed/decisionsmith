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
found = tickets.query(query_embeddings=[embed("refund")], n_results=2, where={"team": "billing"})
print("billing team hits:", found["documents"][0])
