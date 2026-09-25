import asyncio
import json

import pytest

pytest.importorskip("crewai")

import httpx
import respx

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth
from tests.integrations.openai_mock import reply

Q = compile_schema(Ticket).questions()
URL = "https://api.openai.com/v1/chat/completions"


def harness():
    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


@respx.mock
def test_crewai_llm_is_a_teacher():
    from crewai import LLM

    route = respx.post(URL).mock(side_effect=reply())
    e = from_string(LLM(model="gpt-4o-mini", api_key="sk-test"))
    assert e.name == "crewai:gpt-4o-mini"
    assert e.ask("you charged me twice", Q)["answers"]["team"]["choice"] == "billing"
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert '"role":"system"' in route.calls[0].request.content.decode().replace(" ", "")


def test_tool_runs_sync_and_async():
    from crewai.tools import BaseTool

    from decisionsmith.integrations.crewai import tool

    t = tool(harness(), name="route_ticket")
    assert isinstance(t, BaseTool) and t.name == "route_ticket" and "team, wants_refund" in t.description
    assert json.loads(t.run(text="refund my card charge")) == {"team": "billing", "wants_refund": True}
    assert asyncio.run(t.arun(text="sync is broken")) == '{"team":"technical","wants_refund":false}'
    assert t.to_structured_tool().args_schema.model_json_schema()["required"] == ["text"]
    labels = ds.model(["spam", "ham"], FakeEngine(lambda x: {"label": "spam"}))
    assert tool(labels, description="Is it spam?").run(text="x") == "spam"
    assert tool(harness(), field="team").run(text="can I get a quote") == "sales"


def test_guardrail_on_a_real_task_output():
    from crewai import Task
    from crewai.tasks.task_output import TaskOutput
    from crewai.utilities.guardrail import process_guardrail

    from decisionsmith.integrations.crewai import guardrail

    check = guardrail(harness(), "team", block=["technical"])
    task = Task(description="d", expected_output="e", guardrail=check)
    assert task._guardrail is check
    ok = TaskOutput(description="d", raw="my invoice is wrong", agent="a")
    assert process_guardrail(ok, check, retry_count=0).result is ok
    bad = process_guardrail(TaskOutput(description="d", raw="sync is broken", agent="a"), check, retry_count=0)
    assert not bad.success and bad.error == "rejected by decisionsmith: team='technical'; answer again"
    custom = guardrail(ds.model(["yes", "no"], FakeEngine(lambda x: {"label": "yes"})), block=["yes"], message="no")
    assert custom(ok) == (False, "no")


def test_tool_and_guardrail_in_a_crew():
    from crewai import LLM, Agent, Crew, Task

    from decisionsmith.integrations.crewai import guardrail, tool

    call = {"name": "route_ticket", "arguments": json.dumps({"text": "refund my card charge"})}
    replies = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": call}]},
        {"role": "assistant", "content": "sync is broken"},
        {"role": "assistant", "content": "billing: refund my card charge"},
    ]
    bodies = []

    def answer(request):
        bodies.append(json.loads(request.content))
        choice = {"index": 0, "finish_reason": "stop", "message": replies[len(bodies) - 1]}
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "choices": [choice],
                "model": "gpt-4o-mini",
                "usage": usage,
            },
        )

    h = harness()
    with respx.mock:
        respx.post(URL).mock(side_effect=answer)
        agent = Agent(
            role="router",
            goal="route tickets",
            backstory="support",
            llm=LLM(model="gpt-4o-mini", api_key="sk-test"),
            tools=[tool(h, name="route_ticket")],
        )
        task = Task(
            description="Route: refund my card charge",
            expected_output="the team",
            agent=agent,
            guardrail=guardrail(h, "team", block=["technical"]),
        )
        out = Crew(agents=[agent], tasks=[task]).kickoff()
    assert out.raw == "billing: refund my card charge" and len(bodies) == 3
    assert bodies[1]["messages"][-1]["content"] == '{"team":"billing","wants_refund":true}'
    assert "rejected by decisionsmith: team='technical'" in json.dumps(bodies[2]["messages"])
