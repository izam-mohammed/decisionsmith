import asyncio
import json

import pytest

pytest.importorskip("agents")

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    set_tracing_disabled,
)
from agents.testing import ScriptedModel, assistant_message, function_call

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.integrations import openai_agents as integration
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth
from tests.integrations.openai_mock import GOOD

set_tracing_disabled(True)
Q = compile_schema(Ticket).questions()


def harness():
    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def run(agent, input):
    return asyncio.run(Runner.run(agent, input))


def test_input_guardrail_trips_on_blocked_input():
    g = integration.input_guardrail(harness(), "team", block=["billing"])
    agent = Agent(name="a", model=ScriptedModel([[assistant_message("ok")]]), input_guardrails=[g])
    with pytest.raises(InputGuardrailTripwireTriggered) as e:
        run(agent, [{"role": "user", "content": [{"type": "input_text", "text": "my invoice is wrong"}]}])
    assert e.value.guardrail_result.output.output_info == {"team": "billing", "wants_refund": False}
    ok = Agent(name="a", model=ScriptedModel([[assistant_message("ok")]]), input_guardrails=[g])
    assert run(ok, "sync is broken").final_output == "ok"
    empty = Agent(name="a", model=ScriptedModel([[assistant_message("ok")]]), input_guardrails=[g])
    assert run(empty, [{"role": "assistant", "content": "hi"}]).final_output == "ok"


def test_output_guardrail_trips_on_blocked_output():
    from pydantic import BaseModel

    class Reply(BaseModel):
        text: str

    g = integration.output_guardrail(harness(), "wants_refund")
    agent = Agent(name="a", model=ScriptedModel([[assistant_message("refund my card charge")]]), output_guardrails=[g])
    with pytest.raises(OutputGuardrailTripwireTriggered):
        run(agent, "hi")
    reply = json.dumps({"text": "sync is broken"})
    fine = Agent(name="a", model=ScriptedModel([[assistant_message(reply)]]), output_guardrails=[g], output_type=Reply)
    assert run(fine, "hi").final_output.text == "sync is broken"


def test_function_tool_runs_in_the_runner():
    tool = integration.function_tool(harness(), name="route")
    assert tool.name == "route" and "team" in tool.description
    model = ScriptedModel(
        [[function_call("route", {"text": "refund my card charge"}, call_id="c1")], [assistant_message("done")]]
    )
    result = run(Agent(name="a", model=model, tools=[tool]), "help")
    assert result.final_output == "done"
    outputs = [i.output for i in result.new_items if i.type == "tool_call_output_item"]
    assert json.loads(outputs[0]) == {"team": "billing", "wants_refund": True}
    labels = ds.model(["spam", "ham"], FakeEngine(lambda t: {"label": "spam"}))
    assert integration.function_tool(labels, description="spam?").description == "spam?"


def test_router_hands_off_without_a_triage_call():
    billing = Agent(name="billing", model=ScriptedModel([[assistant_message("billing here")]]))
    other = Agent(name="other", model=ScriptedModel([[assistant_message("other here")]]))
    r = integration.router(harness(), "team", {"billing": billing}, default=other)
    assert asyncio.run(r.run("my invoice is wrong")).final_output == "billing here"
    assert asyncio.run(r.run([{"role": "user", "content": "sync is broken"}])).final_output == "other here"
    with pytest.raises(KeyError, match="no agent for team='sales'"):
        asyncio.run(integration.router(harness(), "team", {"billing": billing}).pick("how much is the plan"))


def test_agents_model_is_a_teacher():
    e = integration.teacher(ScriptedModel([[assistant_message(GOOD)], [assistant_message(GOOD)]]))
    assert e.name == "openai-agents:ScriptedModel"
    assert e.ask("you charged me twice", Q)["answers"]["team"]["choice"] == "billing"
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert from_string(integration.teacher("gpt-5-mini")).name == "openai-agents:gpt-5-mini"
