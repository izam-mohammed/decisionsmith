import asyncio

import pytest

pytest.importorskip("autogen_core")

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth
from tests.integrations.openai_mock import openai_client

Q = compile_schema(Ticket).questions()


def test_openai_client_is_a_teacher():
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    c = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="sk-test")
    c._client, seen = openai_client()
    e = from_string(c)
    assert e.name == "autogen:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0

    async def inside_a_loop():
        return e.ask("x", Q)

    assert asyncio.run(inside_a_loop())["answers"]["team"]["choice"] == "billing"
    assert '"role":"system"' in seen[0].content.decode().replace(" ", "")


def harness():
    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def spam():
    return ds.model(["spam", "ok"], FakeEngine(lambda t: {"label": "spam" if "free" in t else "ok"}))


def test_function_tool_runs_in_an_assistant_agent():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import ToolCallSummaryMessage
    from autogen_core import CancellationToken, FunctionCall
    from autogen_core.models import CreateResult, ModelFamily, ModelInfo, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient

    from decisionsmith.integrations.autogen import tool

    route = tool(harness(), name="route_ticket")
    assert route.name == "route_ticket" and "team, wants_refund" in route.description
    assert route.schema["parameters"]["required"] == ["text"]
    call = FunctionCall(id="1", name="route_ticket", arguments='{"text": "refund my card charge"}')
    info = ModelInfo(vision=False, function_calling=True, json_output=False, family=ModelFamily.UNKNOWN)
    client = ReplayChatCompletionClient(
        [CreateResult(finish_reason="function_calls", content=[call], usage=RequestUsage(0, 0), cached=False)],
        model_info=info,
    )
    agent = AssistantAgent("support", model_client=client, tools=[route])
    result = asyncio.run(agent.run(task="route this"))
    summary = result.messages[-1]
    assert isinstance(summary, ToolCallSummaryMessage) and '"team": "billing"' in summary.content
    label = tool(spam(), description="Is it spam?")
    assert label.description == "Is it spam?"
    assert asyncio.run(label.run_json({"text": "free money"}, CancellationToken())) == "spam"


def test_stop_on_ends_a_team_run():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import TerminatedException
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.replay import ReplayChatCompletionClient

    from decisionsmith.integrations.autogen import stop_on

    replies = ["how much is the plan", "can I get a quote", "sync is broken", "never said"]
    agent = AssistantAgent("writer", model_client=ReplayChatCompletionClient(replies))
    stop = stop_on(harness(), "team", block=["technical"], sources=["writer"])
    team = RoundRobinGroupChat([agent], termination_condition=stop | MaxMessageTermination(10))
    result = asyncio.run(team.run(task="the app crashes"))
    assert [m.content for m in result.messages[1:]] == replies[:3]
    assert result.stop_reason == "blocked by decisionsmith: team='technical' in a message from writer"
    assert not stop.terminated
    assert asyncio.run(stop([TextMessage(content="sync is broken", source="user")])) is None
    assert asyncio.run(stop([TextMessage(content="sync is broken", source="writer")])) is not None
    assert stop.terminated
    with pytest.raises(TerminatedException):
        asyncio.run(stop([]))
    asyncio.run(stop.reset())
    assert not stop.terminated
    labels = stop_on(spam(), block=["spam"])
    assert asyncio.run(labels([TextMessage(content="free money", source="user")])).source == "DecisionTermination"
