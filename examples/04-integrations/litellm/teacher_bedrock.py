"""Amazon Nova on Bedrock through LiteLLM (AWS credentials and AWS_REGION_NAME; pip install boto3)."""

from _schema import TEXTS, Ticket

import decisionsmith as ds

model = ds.model(Ticket)
rows = model.label(TEXTS, teacher="litellm/bedrock/amazon.nova-lite-v1:0")
print(len(rows), "rows labelled; LiteLLM reports the cost of each call in the log")
