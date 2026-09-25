"""A smolagents `CodeAgent` with a `route_ticket` tool: the agent writes code that calls it and the harness answers
(OPENAI_API_KEY).

With DS_OFFLINE=1 the tool is called directly instead of running the agent's LLM.
"""

import os

from _schema import TEXTS, Ticket
from smolagents import CodeAgent, OpenAIServerModel

import decisionsmith as ds
from decisionsmith.integrations.smolagents import tool

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
route = tool(h, name="route_ticket", description="Which team handles a support ticket, and is it a refund?")
agent = CodeAgent(tools=[route], model=OpenAIServerModel(model_id="gpt-5-mini"), max_steps=3)

print(route.to_code_prompt().splitlines()[0])
for text in TEXTS[:3]:
    if os.environ.get("DS_OFFLINE"):
        print(route(text=text), "<-", text)
    else:
        print(agent.run("Route this ticket with route_ticket and give the team: %s" % text))
