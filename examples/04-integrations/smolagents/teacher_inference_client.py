"""An open model on Hugging Face Inference Providers through smolagents' InferenceClientModel (HF_TOKEN)."""

from _schema import TEXTS, Ticket
from smolagents import InferenceClientModel

import decisionsmith as ds

llm = InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Instruct")
h = ds.harness(Ticket, teacher=llm, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
