"""An Azure OpenAI deployment through CrewAI (AZURE_API_KEY, AZURE_ENDPOINT).

uv add "crewai[azure-ai-inference]"; the model id is `azure/<your deployment name>`.
"""

from _schema import TEXTS, Ticket
from crewai import LLM

import decisionsmith as ds

h = ds.harness(Ticket, teacher=LLM(model="azure/gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
