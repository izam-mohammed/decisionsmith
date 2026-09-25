import asyncio

import pytest

pytest.importorskip("pydantic_ai")

from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import GOOD, openai_client

Q = compile_schema(Ticket).questions()


def test_openai_model_is_a_teacher():
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client, seen = openai_client()
    e = from_string(OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=client)))
    assert e.name == "pydantic-ai:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert "<text>" in seen[0].content.decode() and seen[0].url.path == "/v1/chat/completions"


def test_test_model_is_a_teacher():
    from pydantic_ai.models.test import TestModel

    e = from_string(TestModel(custom_output_text=GOOD))
    assert e.ask("x", Q)["answers"]["team"]["choice"] == "billing" and e.name.startswith("pydantic-ai:")


def harness():
    import decisionsmith as ds
    from decisionsmith.testing import FakeEngine
    from tests.conftest import truth

    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def test_tool_runs_inside_an_agent():
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    from decisionsmith.integrations.pydantic_ai import tool

    returned = []

    def model(messages, info):
        assert [t.name for t in info.function_tools] == ["route"] and "team" in info.function_tools[0].description
        done = [p for msg in messages for p in msg.parts if isinstance(p, ToolReturnPart)]
        if not done:
            return ModelResponse(parts=[ToolCallPart("route", {"text": "refund my card charge"})])
        returned.append(done[0].content)
        return ModelResponse(parts=[TextPart("routed to %s" % done[0].content["team"])])

    agent = Agent(FunctionModel(model), tools=[tool(harness(), name="route")])
    assert agent.run_sync("help").output == "routed to billing"
    assert returned == [{"team": "billing", "wants_refund": True}]


def test_tool_with_a_labels_model():
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    import decisionsmith as ds
    from decisionsmith.integrations.pydantic_ai import tool
    from decisionsmith.testing import FakeEngine

    m = ds.model(["yes", "no"], FakeEngine(lambda t: {"label": "yes"}))
    t = tool(m, description="Is it spam?")
    assert t.description == "Is it spam?"
    result = Agent(TestModel(), tools=[t]).run_sync("x")
    assert '"decide":"yes"' in result.output.replace(" ", "")


def test_output_validator_retries_blocked_replies():
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart
    from pydantic_ai.models.function import FunctionModel

    from decisionsmith.integrations.pydantic_ai import output_validator

    retries = []

    def model(messages, info):
        retry = [p for msg in messages for p in msg.parts if isinstance(p, RetryPromptPart)]
        retries.extend(r.content for r in retry)
        return ModelResponse(parts=[TextPart("how much is the plan" if retry else "sync is broken")])

    agent = Agent(FunctionModel(model))
    agent.output_validator(output_validator(harness(), "team", block=["technical"]))
    assert agent.run_sync("hi").output == "how much is the plan"
    assert "team='technical'" in retries[0]


def test_output_validator_judges_structured_output():
    from pydantic import BaseModel
    from pydantic_ai import Agent, UnexpectedModelBehavior
    from pydantic_ai.models.test import TestModel

    from decisionsmith.integrations.pydantic_ai import output_validator

    class Reply(BaseModel):
        text: str

    agent = Agent(TestModel(custom_output_args={"text": "sync is broken"}), output_type=Reply, retries=1)
    agent.output_validator(output_validator(harness(), "team", block=["billing"]))
    assert agent.run_sync("x").output.text == "sync is broken"
    strict = Agent(TestModel(custom_output_args={"text": "sync is broken"}), output_type=Reply, retries=1)
    strict.output_validator(output_validator(harness(), "team", block=["technical"], message="no"))
    with pytest.raises(UnexpectedModelBehavior):
        strict.run_sync("x")


def test_router_picks_the_agent():
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from decisionsmith.integrations.pydantic_ai import router

    billing = Agent(TestModel(custom_output_text="billing agent"))
    other = Agent(TestModel(custom_output_text="other agent"))
    r = router(harness(), "team", {"billing": billing}, default=other)
    assert r.pick("my invoice is wrong") is billing
    assert r.run_sync("my invoice is wrong").output == "billing agent"
    assert asyncio.run(r.run("sync is broken")).output == "other agent"
    assert asyncio.run(r.apick("refund my card charge")) is billing
    with pytest.raises(KeyError, match="no agent for team='sales'"):
        router(harness(), "team", {"billing": billing}).pick("how much is the plan")
