"""A Pydantic AI agent that calls your model as a tool to route tickets (OPENAI_API_KEY).

Offline (DS_OFFLINE=1) the agent runs on Pydantic AI's TestModel, which calls every tool once.
"""

import os

from _schema import TEXTS, Ticket
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import decisionsmith as ds
from decisionsmith.integrations.pydantic_ai import tool

llm = TestModel() if os.environ.get("DS_OFFLINE") else "openai:gpt-5-mini"
route = tool(ds.model(Ticket), name="route_ticket", description="Which team handles this ticket, and is it a refund?")
agent = Agent(llm, tools=[route], instructions="Route the ticket with route_ticket, then say which team gets it.")

for text in TEXTS[:3]:
    print(text, "->", agent.run_sync(text).output)
