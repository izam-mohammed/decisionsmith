"""An agent that calls your model as a function tool to route tickets (OPENAI_API_KEY)."""

import asyncio

from _offline import llm
from _schema import TEXTS, Ticket
from agents import Agent, Runner

import decisionsmith as ds
from decisionsmith.integrations.openai_agents import function_tool

route = function_tool(ds.model(Ticket), name="route_ticket", description="Which team handles this ticket?")


async def main() -> None:
    for text in TEXTS[:3]:
        agent = Agent(
            name="support",
            instructions="Route the ticket with route_ticket, then say which team gets it.",
            model=llm("(offline) routed", tool="route_ticket"),
            tools=[route],
        )
        print(text, "->", (await Runner.run(agent, text)).final_output)


asyncio.run(main())
