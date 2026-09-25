"""Hand each ticket to a team agent with your model instead of a triage LLM call (OPENAI_API_KEY)."""

import asyncio

from _offline import llm
from _schema import TEXTS, Ticket
from agents import Agent

import decisionsmith as ds
from decisionsmith.integrations.openai_agents import router

triage = ds.harness(Ticket, teacher="gpt-5-mini", student="laya")


async def main() -> None:
    for text in TEXTS[:3]:
        agents = {
            team: Agent(name=team, instructions="You are the %s support team." % team, model=llm("(%s reply)" % team))
            for team in ["billing", "technical", "sales"]
        }
        result = await router(triage, "team", agents).run(text)
        print(text, "->", result.last_agent.name, "->", result.final_output)


asyncio.run(main())
