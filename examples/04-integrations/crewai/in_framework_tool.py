"""A CrewAI tool: the agent calls `route_ticket` and your harness answers (OPENAI_API_KEY).

With DS_OFFLINE=1 the tool is run directly instead of kicking off the crew.
"""

import os

from _schema import TEXTS, Ticket
from crewai import Agent, Crew, Task

import decisionsmith as ds
from decisionsmith.integrations.crewai import tool

h = ds.harness(Ticket, teacher="gpt-5-mini", student="laya")
route = tool(h, name="route_ticket", description="Which team handles a support ticket, and is it a refund?")

if os.environ.get("DS_OFFLINE"):
    for text in TEXTS[:3]:
        print(text, "->", route.run(text=text))
else:
    agent = Agent(
        role="Support router",
        goal="Send each ticket to the right team",
        backstory="You triage support.",
        llm="gpt-5-mini",
        tools=[route],
    )
    task = Task(description="Route this ticket: {ticket}", expected_output="The team name", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    for text in TEXTS[:3]:
        print(text, "->", crew.kickoff(inputs={"ticket": text}).raw)
