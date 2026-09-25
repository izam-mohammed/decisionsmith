"""Azure OpenAI through LlamaIndex (AZURE_OPENAI_API_KEY; pip install llama-index-llms-azure-openai).

`engine` is your deployment name; set azure_endpoint to your resource URL.
"""

from _schema import TEXTS, Ticket
from llama_index.llms.azure_openai import AzureOpenAI

import decisionsmith as ds

llm = AzureOpenAI(
    engine="gpt-5-mini",
    model="gpt-5-mini",
    azure_endpoint="https://my-resource.openai.azure.com/",
    api_version="2024-10-21",
)
h = ds.harness(Ticket, teacher=llm, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
