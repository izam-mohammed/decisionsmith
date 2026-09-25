"""Hugging Face Inference Providers through Haystack (HF_TOKEN; pip install huggingface-api-haystack)."""

from _schema import TEXTS, Ticket
from haystack_integrations.components.generators.huggingface_api import HuggingFaceAPIChatGenerator

import decisionsmith as ds

llm = HuggingFaceAPIChatGenerator(
    api_type="serverless_inference_api", api_params={"model": "meta-llama/Llama-3.3-70B-Instruct"}
)
model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=llm)
print(len(rows), "rows labelled")
