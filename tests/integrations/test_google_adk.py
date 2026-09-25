import asyncio
import json

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

import decisionsmith as ds
from decisionsmith.integrations.google_adk import guard, last_user_text, tool
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth


class ScriptedLlm(BaseLlm):
    """Calls the first tool with the user's text, then replies with the tool's result (no network)."""

    model: str = "scripted"
    calls: int = 0

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        self.calls += 1
        last = llm_request.contents[-1].parts[0]
        if last.function_response:
            text = json.dumps(last.function_response.response, sort_keys=True)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
            return
        name = next(iter(llm_request.tools_dict), None)
        if name is None:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="model reply")]))
            return
        call = types.FunctionCall(name=name, args={"text": last_user_text(llm_request)})
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))


def harness():
    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


async def run(agent, text):
    runner = InMemoryRunner(agent=agent, app_name="test")
    session = await runner.session_service.create_session(app_name="test", user_id="u")
    message = types.Content(role="user", parts=[types.Part(text=text)])
    events = [e async for e in runner.run_async(user_id="u", session_id=session.id, new_message=message)]
    return [p.text for e in events if e.content for p in e.content.parts or [] if p.text]


def test_function_tool_runs_in_an_agent():
    t = tool(harness(), name="route_ticket")
    assert t.name == "route_ticket" and "team, wants_refund" in t.description
    agent = LlmAgent(name="support", model=ScriptedLlm(), tools=[t])
    out = asyncio.run(run(agent, "refund my card charge"))
    assert json.loads(out[-1]) == {"team": "billing", "wants_refund": True}
    labels = ds.model(["easy", "hard"], FakeEngine(lambda text: {"label": "hard"}))
    lt = tool(labels, description="Is this hard?")
    assert lt.name == "decide" and lt.description == "Is this hard?"
    out = asyncio.run(run(LlmAgent(name="a", model=ScriptedLlm(), tools=[lt]), "why?"))
    assert json.loads(out[-1]) == {"label": "hard"}


def test_before_model_guard_skips_the_model():
    llm = ScriptedLlm()
    agent = LlmAgent(name="a", model=llm, before_model_callback=guard(harness(), "team", ["billing"], message="no"))
    assert asyncio.run(run(agent, "my invoice is wrong")) == ["no"] and llm.calls == 0
    assert asyncio.run(run(agent, "sync is broken")) == ["model reply"] and llm.calls == 1


def test_last_user_text():
    fn = types.Part(function_response=types.FunctionResponse(name="x", response={"a": 1}))
    contents = [
        types.Content(role="user", parts=[types.Part(text="first"), types.Part(text="ask")]),
        types.Content(role="model", parts=[types.Part(text="reply")]),
        types.Content(role="user", parts=[fn]),
    ]
    assert last_user_text(LlmRequest(contents=contents)) == "first ask"
    assert last_user_text(LlmRequest(contents=[])) == ""
    assert asyncio.run(guard(harness(), "team")(None, LlmRequest(contents=[]))) is None
