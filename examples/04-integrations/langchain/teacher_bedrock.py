"""Amazon Nova on Bedrock through ChatBedrockConverse (AWS credentials; pick your region)."""

from _schema import TEXTS, Ticket
from langchain_aws import ChatBedrockConverse

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher=ChatBedrockConverse(model="amazon.nova-lite-v1:0", region_name="us-east-1"))
print(len(rows), "rows labelled; train Laya on them with model.train(rows)")
