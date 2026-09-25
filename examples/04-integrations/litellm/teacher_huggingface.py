"""Hugging Face Inference Providers through LiteLLM (HF_TOKEN)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="litellm/huggingface/meta-llama/Llama-3.3-70B-Instruct")
print(len(rows), "rows labelled; LiteLLM reports the cost of each call in the log")
