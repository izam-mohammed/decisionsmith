"""Any OpenAI-compatible server as the teacher (llama.cpp, SGLang, a gateway, ...): set LLM_URL and LLM_API_KEY.

Shadow mode: the teacher answers, Laya answers too, and status() shows how often they agree.
"""

import os

from _schema import TEXTS, Ticket

import decisionsmith as ds

url = os.environ.get("LLM_URL", "http://localhost:8080/v1")
teacher = ds.LLM(os.environ.get("LLM_MODEL", "my-model"), url=url, api_key=os.environ.get("LLM_API_KEY"))
h = ds.harness(Ticket, teacher=teacher, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
