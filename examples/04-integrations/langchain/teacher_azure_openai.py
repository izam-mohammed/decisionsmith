"""Azure OpenAI through AzureChatOpenAI (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT).

azure_deployment is your deployment name.
"""

from _schema import TEXTS, Ticket
from langchain_openai import AzureChatOpenAI

import decisionsmith as ds

llm = AzureChatOpenAI(azure_deployment="gpt-5-mini", api_version="2024-10-21")
h = ds.harness(Ticket, teacher=llm, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
