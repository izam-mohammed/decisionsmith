"""Hugging Face Inference Providers through ChatHuggingFace (HF_TOKEN)."""

from _schema import TEXTS, Ticket
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import decisionsmith as ds

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.3-70B-Instruct", provider="auto")
model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=ChatHuggingFace(llm=llm))
print(len(rows), "rows labelled; train Laya on them with model.train(rows)")
