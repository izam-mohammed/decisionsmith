"""A local Hugging Face model through smolagents' TransformersModel (downloads Qwen/Qwen3-0.6B on first run)."""

from _schema import TEXTS, Ticket
from smolagents import TransformersModel

import decisionsmith as ds

llm = TransformersModel(model_id="Qwen/Qwen3-0.6B", max_new_tokens=256)
h = ds.harness(Ticket, teacher=llm, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
