"""A RAG relevance filter: after Chroma retrieves, the model drops the passages that don't help the question.

Train the judge on your own question/passage pairs first; base Laya is only a starting point.
"""

import chromadb
from _schema import DOCS, embed

import decisionsmith as ds
from decisionsmith.integrations.chroma import filter_results

client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
docs = client.get_or_create_collection("help-centre", embedding_function=None)
docs.upsert(ids=[str(i) for i in range(len(DOCS))], documents=DOCS, embeddings=[embed(d) for d in DOCS])

judge = ds.model(["relevant", "off-topic"], question="Does the document help answer the query?")
question = "How do I get a refund?"
hits = docs.query(query_embeddings=[embed(question)], n_results=4)
kept = filter_results(judge, hits, keep=["relevant"], query=question)
print("retrieved:", len(hits["ids"][0]), "kept:", len(kept["ids"][0]))
for doc in kept["documents"][0]:
    print(" -", doc)
