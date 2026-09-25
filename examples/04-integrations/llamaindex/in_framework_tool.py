"""A `FunctionTool` for LlamaIndex agents: the agent calls the harness to route a ticket.

agent = FunctionAgent(tools=[route], llm=OpenAI(model="gpt-5-mini")); await agent.run("...")
"""

from _schema import TEXTS, Ticket

import decisionsmith as ds
from decisionsmith.integrations.llamaindex import as_tool

h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")
route = as_tool(h, name="route_ticket", description="Which team handles a support ticket, and is it a refund?")

print(route.metadata.name, route.metadata.get_parameters_dict()["properties"])
for text in TEXTS[:3]:
    print(text, "->", route.call(text=text).raw_output)
