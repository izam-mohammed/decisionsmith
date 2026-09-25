"""Azure OpenAI through Semantic Kernel (AZURE_OPENAI_API_KEY).

`deployment_name` is your deployment; set endpoint to your resource URL.
"""

from _schema import TEXTS, Ticket
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

import decisionsmith as ds

service = AzureChatCompletion(
    deployment_name="gpt-5-mini", endpoint="https://my-resource.openai.azure.com/", api_version="2024-10-21"
)
h = ds.harness(Ticket, teacher=service, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
