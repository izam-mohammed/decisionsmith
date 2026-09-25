"""Amazon Nova on Bedrock through Pydantic AI (AWS credentials with Bedrock access)."""

from _schema import TEXTS, Ticket
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

import decisionsmith as ds

model = BedrockConverseModel("us.amazon.nova-lite-v1:0", provider=BedrockProvider(region_name="us-east-1"))
h = ds.harness(Ticket, teacher=model, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
