import asyncio

import pytest

pytest.importorskip("smolagents")

from decisionsmith.engines import from_string
from decisionsmith.integrations.smolagents import _text
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import openai_client

Q = compile_schema(Ticket).questions()


def test_openai_server_model_is_a_teacher():
    from smolagents import OpenAIServerModel

    m = OpenAIServerModel(model_id="gpt-4o-mini", api_key="sk-test")
    m.client, seen = openai_client(is_async=False)
    e = from_string(m)
    assert e.name == "smolagents:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert '"role":"system"' in seen[0].content.decode().replace(" ", "")


def test_content_parts():
    assert _text([{"type": "text", "text": "a"}, "x", {"text": "b"}]) == "ab" and _text(None) == ""


def harness():
    import decisionsmith as ds
    from decisionsmith.testing import FakeEngine
    from tests.conftest import truth

    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def test_tool_is_a_smolagents_tool():
    from smolagents import Tool

    import decisionsmith as ds
    from decisionsmith.integrations import smolagents as integration
    from decisionsmith.testing import FakeEngine

    t = integration.tool(harness(), "route_ticket")
    assert isinstance(t, Tool) and isinstance(t, integration.DecisionTool)
    assert t.name == "route_ticket" and t.output_type == "object" and "team, wants_refund" in t.description
    assert t(text="refund my card charge") == {"team": "billing", "wants_refund": True}
    assert t({"text": "sync is broken"})["team"] == "technical"
    spam = integration.tool(ds.model(["yes", "no"], FakeEngine(lambda text: {"label": "yes"})), description="Spam?")
    assert spam("x") == "yes" and spam.output_type == "string" and spam.description == "Spam?"
    assert "def route_ticket(text: string) -> object" in t.to_code_prompt()
    with pytest.raises(AttributeError):
        integration.Missing


def test_code_agent_calls_the_tool():
    from smolagents import ChatMessage, CodeAgent, Model

    from decisionsmith.integrations.smolagents import tool

    class Scripted(Model):
        def generate(self, messages, stop_sequences=None, **kwargs):
            code = 'r = route_ticket(text="my invoice is wrong")\nfinal_answer(r["team"])'
            return ChatMessage(role="assistant", content="Thought: route it.\n<code>\n%s\n</code>" % code)

    agent = CodeAgent(tools=[tool(harness(), "route_ticket")], model=Scripted(), max_steps=2, verbosity_level=0)
    assert agent.run("route this ticket") == "billing"


def test_tool_calling_agent_calls_the_tool():
    from smolagents import ChatMessage, ChatMessageToolCall, Model, ToolCallingAgent
    from smolagents.models import ChatMessageToolCallFunction

    from decisionsmith.integrations.smolagents import tool

    class Scripted(Model):
        def generate(self, messages, stop_sequences=None, **kwargs):
            seen = str(messages)
            done = "technical" in seen
            name, args = ("final_answer", {"answer": "done"}) if done else ("route", {"text": "sync is broken"})
            call = ChatMessageToolCall(ChatMessageToolCallFunction(name=name, arguments=args), id="c1", type="function")
            return ChatMessage(role="assistant", content="", tool_calls=[call])

    agent = ToolCallingAgent(tools=[tool(harness(), "route")], model=Scripted(), max_steps=3, verbosity_level=0)
    assert agent.run("route it") == "done"
    assert "technical" in str(agent.memory.steps[1].observations)
