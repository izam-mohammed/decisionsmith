"""A decision as a LangChain tool: bind it to any chat model that calls tools (OPENAI_API_KEY for the teacher)."""

from _schema import Ticket
from langchain_openai import ChatOpenAI

import decisionsmith as ds
from decisionsmith.integrations.langchain import as_tool

h = ds.harness(Ticket, teacher=ChatOpenAI(model="gpt-5-mini"), student="laya")
tool = as_tool(h, "route_ticket", "Decide which team handles a support ticket and whether it asks for a refund.")

agent_llm = ChatOpenAI(model="gpt-5-mini").bind_tools([tool])  # the agent calls route_ticket(text=...)
print(tool.name, tool.args)
print(tool.invoke({"text": "My invoice shows the wrong company name"}))
