"""A RAG relevance filter: retrieved documents the model marks off-topic never reach the generator.

Put it between the retriever and the prompt builder. Train the model on your own query/document pairs first;
base Laya is only a starting point.
"""

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

import decisionsmith as ds
from decisionsmith.integrations.haystack import document_filter

store = InMemoryDocumentStore()
store.write_documents(
    [
        Document(content="Refunds are paid back to the original card within 5 business days."),
        Document(content="Our office dog is called Biscuit and refuses to give refunds."),
        Document(content="To request a refund, reply to your receipt email."),
    ]
)
judge = ds.model(["relevant", "off-topic"], question="Does the document help answer the query?")

pipe = Pipeline()
pipe.add_component("retrieve", InMemoryBM25Retriever(store, top_k=3))
pipe.add_component("keep", document_filter(judge, keep=["relevant"]))
pipe.connect("retrieve.documents", "keep.documents")

query = "How do I get a refund?"
kept = pipe.run({"retrieve": {"query": query}, "keep": {"query": query}})["keep"]["documents"]
print(len(kept), "of 3 documents kept")
for doc in kept:
    print("kept:", doc.content)
