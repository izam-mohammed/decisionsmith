import asyncio

import pytest

litellm = pytest.importorskip("litellm")

import decisionsmith as ds  # noqa: E402
from decisionsmith.engines import EngineError, from_string  # noqa: E402
from decisionsmith.integrations import litellm as integration  # noqa: E402
from decisionsmith.schema import compile_schema  # noqa: E402
from decisionsmith.testing import FakeEngine  # noqa: E402
from tests.conftest import Ticket, truth  # noqa: E402

Q = compile_schema(Ticket).questions()
GOOD = '{"q0": "billing", "q1": true}'


def test_teacher_uses_litellm_completion():
    e = integration.teacher("gpt-4o-mini", mock_response=GOOD)
    out = e.ask("you charged me twice, refund", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["answers"]["wants_refund"]["noul"] == 1.0
    assert out["usage"]["input_tokens"] and "cost_usd" in out["usage"]
    assert asyncio.run(e.aask("x", Q))["answers"]["team"]["choice"] == "billing"
    w = integration.teacher("gpt-4o-mini", mock_response='{"texts": ["a", "b"]}')
    assert w.write("write 2", 2) == ["a", "b"]
    assert ds.LLM("litellm/gpt-4o-mini", mock_response=GOOD).ask("x", Q)["model"] == "litellm/gpt-4o-mini"
    assert from_string("litellm/gpt-4o-mini").name == "litellm/gpt-4o-mini"


def test_teacher_errors_and_cost(monkeypatch):
    e = integration.teacher("gpt-4o-mini", mock_response=litellm.RateLimitError("slow", "openai", "gpt-4o-mini"))
    with pytest.raises(EngineError, match="RateLimitError"):
        e.ask("x", Q)
    monkeypatch.setattr(litellm, "completion_cost", lambda **k: (_ for _ in ()).throw(ValueError()))
    assert integration.teacher("gpt-4o-mini", mock_response=GOOD).ask("x", Q)["usage"]["cost_usd"] is None


def test_teacher_trains_a_model_end_to_end():
    m = ds.model(Ticket, FakeEngine(truth, name="student"))
    rows = m.label(["my invoice is wrong", "sync is broken"], integration.teacher("gpt-4o-mini", mock_response=GOOD))
    assert [r["answers"]["team"]["billing"] for r in rows] == [1.0, 1.0]


def guard(block, hook="pre_call", field="team"):
    h = ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)
    cls = integration.guardrail(h, field=field, block=block)
    return cls(guardrail_name="ds", event_hook=hook, default_on=True)


def test_guardrail_pre_call():
    g = guard(["billing"])
    data = {"messages": [{"role": "system", "content": "hi"}, {"role": "user", "content": "my invoice is wrong"}]}
    with pytest.raises(ValueError, match="team='billing'"):
        asyncio.run(g.async_pre_call_hook(None, None, data, "completion"))
    ok = {"messages": [{"role": "user", "content": [{"type": "text", "text": "the app crashes on login"}]}]}
    assert asyncio.run(g.async_pre_call_hook(None, None, ok, "completion")) is ok
    assert asyncio.run(g.async_pre_call_hook(None, None, {"messages": []}, "completion")) == {"messages": []}
    off = guard(["billing"], hook="post_call")
    assert asyncio.run(off.async_pre_call_hook(None, None, data, "completion")) is data


def test_guardrail_post_call():
    g = guard(["technical"], hook="post_call")
    reply = litellm.completion(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], mock_response="sync is broken"
    )
    with pytest.raises(ValueError, match="blocked by decisionsmith"):
        asyncio.run(g.async_post_call_success_hook({}, None, reply))
    fine = litellm.completion(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], mock_response="how much is the plan"
    )
    assert asyncio.run(g.async_post_call_success_hook({}, None, fine)) is fine
    pre = guard(["technical"])
    assert asyncio.run(pre.async_post_call_success_hook({}, None, reply)) is reply


def test_tier_picks_a_router_model():
    m = ds.model(["easy", "hard"], FakeEngine(lambda t: {"label": "hard" if "why" in t else "easy"}))
    pick = integration.tier(m, None, {"hard": "strong"}, default="cheap")
    assert pick([{"role": "user", "content": "why is the sky blue"}]) == "strong" and pick("hi") == "cheap"
    router = litellm.Router(
        model_list=[
            {"model_name": "strong", "litellm_params": {"model": "gpt-4o", "mock_response": "big"}},
            {"model_name": "cheap", "litellm_params": {"model": "gpt-4o-mini", "mock_response": "small"}},
        ]
    )
    messages = [{"role": "user", "content": "why?"}]
    assert router.completion(model=pick(messages), messages=messages).choices[0].message.content == "big"
    with pytest.raises(KeyError, match="no model"):
        integration.tier(m, None, {"hard": "strong"})("hello")
    assert integration.last_text([{"role": "assistant", "content": "x"}]) == ""
