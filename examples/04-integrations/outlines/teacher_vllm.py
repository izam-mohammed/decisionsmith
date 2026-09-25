"""A vLLM server as the teacher through Outlines (vllm serve Qwen/Qwen3-8B); vLLM enforces the answer schema."""

import openai
import outlines
from _schema import TEXTS, Ticket

import decisionsmith as ds

teacher = outlines.from_vllm(openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none"), "Qwen/Qwen3-8B")
model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=teacher)
print(len(rows), "rows labelled; train with model.train(rows)")
