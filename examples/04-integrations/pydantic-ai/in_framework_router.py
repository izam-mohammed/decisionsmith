"""Route each ticket to a team agent with your model instead of a triage LLM call (OPENAI_API_KEY).

Offline (DS_OFFLINE=1) the team agents run on Pydantic AI's TestModel.
"""

import os

from _schema import TEXTS, Ticket
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import decisionsmith as ds
from decisionsmith.integrations.pydantic_ai import router


def team_agent(team: str) -> Agent:
    llm = TestModel(custom_output_text="(%s agent reply)" % team) if os.environ.get("DS_OFFLINE") else None
    return Agent(llm or "openai:gpt-5-mini", name=team, instructions="You are the %s support team." % team)


agents = {team: team_agent(team) for team in ["billing", "technical", "sales"]}
triage = router(ds.harness(Ticket, teacher="gpt-5-mini", student="laya"), "team", agents)

for text in TEXTS[:3]:
    agent = triage.pick(text)
    print(text, "->", agent.name, "->", agent.run_sync(text).output)
