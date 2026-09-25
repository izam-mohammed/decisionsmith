"""An Agno agent with a `route_ticket` tool: the agent calls it and the harness answers (OPENAI_API_KEY).

With DS_OFFLINE=1 the tool is called directly instead of running the agent.
"""

import os

from _schema import TEXTS, Ticket
from agno.agent import Agent
from agno.models.openai import OpenAIChat

import decisionsmith as ds
from decisionsmith.integrations.agno import tool

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
route = tool(h, name="route_ticket", description="Which team should handle this support ticket, and is it a refund?")
agent = Agent(model=OpenAIChat(id="gpt-5-mini"), tools=[route], instructions="Route each ticket with route_ticket.")

for text in TEXTS[:3]:
    if os.environ.get("DS_OFFLINE"):
        print(route.entrypoint(text=text), "<-", text)
    else:
        print(agent.run(text).content)
