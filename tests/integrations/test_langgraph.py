import asyncio
from typing import TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

import decisionsmith as ds
from decisionsmith.integrations.langgraph import guard_node, route_on, text_of
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth


def harness():
    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


class Intake(TypedDict, total=False):
    text: str
    handled_by: str


def reply(team):
    return lambda state: {"handled_by": team}


def test_route_on_drives_conditional_edges():
    g = StateGraph(Intake)
    g.add_node("intake", lambda s: {})
    for team in ("billing", "technical", "sales"):
        g.add_node(team, reply(team))
        g.add_edge(team, END)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_on(harness(), "team", key="text"), ["billing", "technical", "sales"])
    app = g.compile()
    assert app.invoke({"text": "you charged me twice"})["handled_by"] == "billing"
    assert asyncio.run(app.ainvoke({"text": "sync is broken"}))["handled_by"] == "technical"
    with pytest.raises(ValueError, match="no text"):
        app.invoke({"text": " "})


def test_route_on_a_labels_model_with_messages_state():
    m = ds.model(["easy", "hard"], FakeEngine(lambda t: {"label": "hard" if "why" in t else "easy"}))
    g = StateGraph(MessagesState)
    g.add_node("intake", lambda s: {})
    g.add_node("cheap", lambda s: {"messages": [AIMessage("cheap")]})
    g.add_node("strong", lambda s: {"messages": [AIMessage("strong")]})
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_on(m), {"easy": "cheap", "hard": "strong"})
    app = g.compile()
    out = app.invoke({"messages": [HumanMessage("why is the sky blue?"), AIMessage("hm")]})
    assert out["messages"][-1].content == "strong"
    assert asyncio.run(app.ainvoke({"messages": [("user", "hello")]}))["messages"][-1].content == "cheap"


class Guarded(MessagesState, total=False):
    blocked: bool


def guarded_app(block):
    g = StateGraph(Guarded)
    g.add_node("guard", guard_node(harness(), "team", block=block))
    g.add_node("agent", lambda s: {"messages": [AIMessage("answered")]})
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", lambda s: s["blocked"], {True: END, False: "agent"})
    g.add_edge("agent", END)
    return g.compile()


def test_guard_node_blocks_and_passes():
    app = guarded_app(["billing"])
    out = app.invoke({"messages": [HumanMessage("my invoice is wrong")]})
    assert out["blocked"] is True and out["messages"][-1].content == "my invoice is wrong"
    out = asyncio.run(app.ainvoke({"messages": [HumanMessage("the app crashes on login")]}))
    assert out["blocked"] is False and out["messages"][-1].content == "answered"
    assert asyncio.run(app.ainvoke({"messages": [HumanMessage("refund my card charge")]}))["blocked"] is True
    assert app.invoke({"messages": [AIMessage("only the assistant spoke")]})["blocked"] is False


def test_text_of_shapes():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, "c"]
    assert text_of({"messages": [{"role": "user", "content": parts}]}) == "a b c"
    assert text_of({"messages": [{"role": "user", "content": None}]}) == ""
    assert text_of({"messages": [HumanMessage("hi")]}) == "hi"

    class State:
        text = "plain"

    assert text_of(State(), "text") == "plain" and text_of(State()) == "" and text_of({}) == ""
