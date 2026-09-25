"""Azure OpenAI through LiteLLM (AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION).

The id after azure/ is your deployment name.
"""

from _schema import TEXTS, Ticket

import decisionsmith as ds

h = ds.harness(Ticket, teacher=ds.LLM("litellm/azure/gpt-5-mini"), student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
