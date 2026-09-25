"""Route each ticket to a team node: `route_on(h, "team")` is the path of a conditional edge (ANTHROPIC_API_KEY)."""

from typing import TypedDict

from _schema import TEXTS, Ticket
from langgraph.graph import END, START, StateGraph

import decisionsmith as ds
from decisionsmith.integrations.langgraph import route_on


class State(TypedDict, total=False):
    text: str
    handled_by: str


h = ds.harness(Ticket, teacher="claude-haiku-4-5", student="laya")

graph = StateGraph(State)
graph.add_node("intake", lambda state: {})
for team in ("billing", "technical", "sales"):
    graph.add_node(team, lambda state, team=team: {"handled_by": team})
    graph.add_edge(team, END)
graph.add_edge(START, "intake")
graph.add_conditional_edges("intake", route_on(h, "team", key="text"), ["billing", "technical", "sales"])
app = graph.compile()

for text in TEXTS:
    print(app.invoke({"text": text})["handled_by"], "<-", text)
