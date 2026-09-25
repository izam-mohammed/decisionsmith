"""RAG relevance filtering: a document compressor keeps the retrieved documents the model finds relevant.

Use it as `ContextualCompressionRetriever(base_compressor=keep, base_retriever=retriever)` (langchain-classic),
or call it on any list of documents as below (OPENAI_API_KEY for the teacher).
"""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import decisionsmith as ds
from decisionsmith.integrations.langchain import compressor


class Relevance(BaseModel):
    relevant: bool = Field(description="Does the document help answer the query?")


h = ds.harness(Relevance, teacher=ChatOpenAI(model="gpt-5-mini"), student="laya")
keep = compressor(h, "relevant", keep=[True])

retrieved = [
    Document("Refunds are paid back to the original card within 5 business days."),
    Document("Our office is closed on public holidays."),
    Document("Duplicate charges are reversed automatically after review."),
]
for doc in keep.compress_documents(retrieved, "I was charged twice, when do I get my money back?"):
    print("keep:", doc.page_content)
