import asyncio
import json

import pytest

pytest.importorskip("agno")

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth
from tests.integrations.openai_mock import http_lib, openai_client, reply

Q = compile_schema(Ticket).questions()


def test_openai_chat_is_a_teacher():
    from agno.models.openai import OpenAIChat

    sync, seen = openai_client(is_async=False)
    async_, _ = openai_client()
    e = from_string(OpenAIChat(id="gpt-4o-mini", client=sync, async_client=async_))
    assert e.name == "agno:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    sent = json.loads(seen[0].content)["messages"]
    assert [m["role"] for m in sent] == ["developer", "user"] and "<text>" in sent[1]["content"]


def agent_client(is_async, calls):
    """An openai client that asks for one `route_ticket` call, then answers with the tool's result."""
    import openai

    lib = http_lib()

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        results = [m["content"] for m in body["messages"] if m["role"] == "tool"]
        if results:
            return reply(content="routed: %s" % results[0], http=lib)(request)
        args = json.dumps({"text": "refund my card charge"})
        call = {"id": "c1", "type": "function", "function": {"name": "route_ticket", "arguments": args}}
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        choice = {"index": 0, "finish_reason": "tool_calls", "message": message}
        body = {"id": "1", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice]}
        return lib.Response(200, json=body)

    http = (lib.AsyncClient if is_async else lib.Client)(transport=lib.MockTransport(handler))
    return (openai.AsyncOpenAI if is_async else openai.OpenAI)(api_key="sk-test", http_client=http)


def test_tool_runs_inside_an_agent():
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    from decisionsmith.integrations.agno import tool

    h = ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)
    calls = []
    model = OpenAIChat(id="gpt-4o-mini", client=agent_client(False, calls), async_client=agent_client(True, calls))
    agent = Agent(model=model, tools=[tool(h, name="route_ticket")])
    out = agent.run("route this")
    assert out.content == 'routed: {"team": "billing", "wants_refund": true}'
    spec = calls[0]["tools"][0]["function"]
    assert spec["name"] == "route_ticket" and "team, wants_refund" in spec["description"]
    assert spec["parameters"]["required"] == ["text"]
    assert asyncio.run(agent.arun("again")).content.startswith("routed: ")


def test_tool_with_a_labels_model():
    from decisionsmith.integrations.agno import tool

    t = tool(ds.model(["yes", "no"], FakeEngine(lambda t: {"label": "yes"})), description="Is it spam?")
    assert t.name == "decide" and t.description == "Is it spam?"
    assert json.loads(t.entrypoint(text="win a prize")) == {"label": "yes"}
