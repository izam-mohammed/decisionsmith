"""OpenAI through LlamaIndex (OPENAI_API_KEY)."""

from _schema import TEXTS, Ticket
from llama_index.llms.openai import OpenAI

import decisionsmith as ds

h = ds.harness(Ticket, teacher=OpenAI(model="gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
