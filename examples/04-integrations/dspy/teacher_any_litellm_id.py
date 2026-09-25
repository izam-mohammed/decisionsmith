"""Any LiteLLM model id through DSPy, here Llama 3.3 on Groq (GROQ_API_KEY).

bedrock/..., vertex_ai/..., mistral/... and every other LiteLLM prefix work the same way.
"""

import dspy
from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=dspy.LM("groq/llama-3.3-70b-versatile"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
