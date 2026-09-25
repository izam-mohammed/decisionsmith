"""Hugging Face Inference Providers through LlamaIndex (HF_TOKEN; pip install llama-index-llms-huggingface-api)."""

import os

from _schema import TEXTS, Ticket
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

import decisionsmith as ds

llm = HuggingFaceInferenceAPI(model="meta-llama/Llama-3.3-70B-Instruct", token=os.environ["HF_TOKEN"])
model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=llm)
print(len(rows), "rows labelled")
