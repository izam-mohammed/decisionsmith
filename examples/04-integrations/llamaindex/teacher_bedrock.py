"""Amazon Nova on Bedrock through LlamaIndex (AWS credentials; pip install llama-index-llms-bedrock-converse)."""

from _schema import TEXTS, Ticket
from llama_index.llms.bedrock_converse import BedrockConverse

import decisionsmith as ds

llm = BedrockConverse(model="amazon.nova-lite-v1:0", region_name="us-east-1")
h = ds.harness(Ticket, teacher=llm, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
