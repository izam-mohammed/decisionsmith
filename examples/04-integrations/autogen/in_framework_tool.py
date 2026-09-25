"""An AssistantAgent with a `route_ticket` FunctionTool: the agent calls it and the harness answers
(OPENAI_API_KEY).

Offline (DS_OFFLINE=1) a replay client plays the agent's model, so the real agent and tool still run.
"""

import asyncio

from _offline import client
from _schema import TEXTS, Ticket
from autogen_agentchat.agents import AssistantAgent

import decisionsmith as ds
from decisionsmith.integrations.autogen import tool

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
route = tool(h, name="route_ticket", description="Which team should handle this support ticket?")


async def main():
    agent = AssistantAgent(
        "support",
        model_client=client([], tool="route_ticket", text=TEXTS[0]),
        tools=[route],
        system_message="Route each ticket with route_ticket.",
    )
    result = await agent.run(task=TEXTS[0])
    print(result.messages[-1].to_text())


asyncio.run(main())
