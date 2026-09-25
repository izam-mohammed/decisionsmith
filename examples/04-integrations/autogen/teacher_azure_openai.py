"""Azure OpenAI through AutoGen (AZURE_OPENAI_API_KEY).

`azure_deployment` is your deployment name; set azure_endpoint to your resource URL.
"""

import os

from _schema import TEXTS, Ticket
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

import decisionsmith as ds

client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-5-mini",
    model="gpt-5-mini",
    azure_endpoint="https://my-resource.openai.azure.com/",
    api_version="2024-10-21",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
h = ds.harness(Ticket, teacher=client, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
