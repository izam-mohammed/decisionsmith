"""An output validator: your model reads every reply, and the agent answers again when it promises a refund.

Uses OPENAI_API_KEY; offline (DS_OFFLINE=1) the agent runs on Pydantic AI's TestModel with a fixed reply.
"""

import os

from pydantic import BaseModel, Field
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

import decisionsmith as ds
from decisionsmith.integrations.pydantic_ai import output_validator


class Reply(BaseModel):
    promises_refund: bool = Field(description="Does this reply promise the customer a refund?")


check = ds.harness(Reply, teacher="gpt-5-mini", student="laya")
llm = TestModel(custom_output_text="Thanks, our team will look into it.") if os.environ.get("DS_OFFLINE") else None
agent = Agent(llm or "openai:gpt-5-mini", instructions="You are a support agent. Never promise refunds.", retries=2)
agent.output_validator(output_validator(check, "promises_refund", message="Do not promise a refund; answer again."))

try:
    print(agent.run_sync("I was charged twice, I want my money back").output)
except UnexpectedModelBehavior:
    print("every reply promised a refund; escalate to a human")
