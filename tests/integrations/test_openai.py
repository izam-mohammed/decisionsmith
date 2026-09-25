import asyncio
import json
from typing import Literal

import pytest

openai = pytest.importorskip("openai")

from pydantic import BaseModel  # noqa: E402

import decisionsmith as ds  # noqa: E402
from decisionsmith.integrations import openai as integration  # noqa: E402
from decisionsmith.schema import compile_schema  # noqa: E402
from decisionsmith.testing import FakeEngine  # noqa: E402
from tests.conftest import Ticket, truth  # noqa: E402
from tests.integrations.openai_mock import openai_client  # noqa: E402

Q = compile_schema(Ticket).questions()
USER = [{"role": "system", "content": "route it"}, {"role": "user", "content": "my invoice is wrong"}]


class Team(BaseModel):
    team: Literal["billing", "technical", "sales"]


class Summary(BaseModel):
    summary: str


def sure_harness(confidence=1.0):
    return ds.harness(Ticket, student=FakeEngine(truth, confidence=confidence, name="student"), log=None)


def test_teacher_sync_and_async_clients():
    client, seen = openai_client(is_async=False)
    e = integration.teacher(client, "llama-3.3-70b-versatile", temperature=0)
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    body = json.loads(seen[0].content)
    assert body["model"] == "llama-3.3-70b-versatile" and body["temperature"] == 0 and "<text>" in str(body)
    assert e.name == "openai-sdk:llama-3.3-70b-versatile"
    aclient, _ = openai_client()
    a = integration.teacher(aclient, "gpt-5-mini")
    assert asyncio.run(a.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert a.ask("x", Q)["answers"]["team"]["choice"] == "billing"
    m = ds.model(Ticket, FakeEngine(truth, name="student"))
    assert len(m.label(["sync is broken"], integration.teacher(client, "gpt-5-mini"))) == 1


def test_wrap_parse_answers_from_a_sure_harness():
    client, seen = openai_client(is_async=False)
    wrapped = integration.wrap(client, sure_harness())
    r = wrapped.chat.completions.parse(model="gpt-5-mini", messages=USER, response_format=Team)
    assert r.id == "decisionsmith" and r.choices[0].message.parsed == Team(team="billing") and not seen
    full = wrapped.chat.completions.parse(model="gpt-5-mini", messages=USER, response_format=Ticket)
    assert full.choices[0].message.parsed.wants_refund is False
    fmt = {"type": "json_schema", "json_schema": {"name": "t", "schema": Team.model_json_schema()}}
    c = wrapped.chat.completions.create(model="gpt-5-mini", messages=USER, response_format=fmt)
    assert json.loads(c.choices[0].message.content) == {"team": "billing"} and not seen
    p = wrapped.chat.completions.parse(model="gpt-5-mini", messages=USER, response_format=fmt)
    assert p.choices[0].message.parsed is None and p.id == "decisionsmith"
    assert wrapped.api_key == "sk-test" and wrapped.chat.completions.with_raw_response is not None


def test_wrap_falls_through_to_the_client(monkeypatch):
    monkeypatch.delenv("DS_OFFLINE", raising=False)
    client, seen = openai_client(is_async=False, content='{"team": "sales"}')
    wrapped = integration.wrap(client, sure_harness(confidence=0.3))
    r = wrapped.chat.completions.parse(model="gpt-5-mini", messages=USER, response_format=Team)
    assert r.id == "chatcmpl-1" and r.choices[0].message.parsed == Team(team="sales") and len(seen) == 1
    sure = integration.wrap(client, sure_harness())
    for fmt in (None, {"type": "json_object"}):
        kw = {"response_format": fmt} if fmt else {}
        assert sure.chat.completions.create(model="m", messages=USER, **kw).id == "chatcmpl-1"
    other, _ = openai_client(is_async=False, content='{"summary": "a bill"}')
    parsed = integration.wrap(other, sure_harness()).chat.completions.parse(
        model="m", messages=USER, response_format=Summary
    )
    assert parsed.choices[0].message.parsed == Summary(summary="a bill")
    assert sure.chat.completions.create(model="m", messages=[{"role": "system", "content": "x"}]).id == "chatcmpl-1"
    narrow = {"type": "json_schema", "json_schema": {"schema": {"properties": {"team": {"enum": ["sales"]}}}}}
    assert sure.chat.completions.create(model="m", messages=USER, response_format=narrow).id == "chatcmpl-1"

    class Only(BaseModel):
        team: Literal["sales", "technical"]

    assert sure.chat.completions.parse(model="m", messages=USER, response_format=Only).id == "chatcmpl-1"


def test_wrap_offline_never_calls_the_client(monkeypatch):
    monkeypatch.setenv("DS_OFFLINE", "1")
    client, seen = openai_client(is_async=False)
    wrapped = integration.wrap(client, sure_harness(confidence=0.3))
    assert wrapped.chat.completions.parse(model="m", messages=USER, response_format=Team).id == "decisionsmith"
    assert not seen


def test_wrap_async_and_simple_models():
    client, seen = openai_client(content='{"label": "no"}')
    m = ds.model(["yes", "no"], FakeEngine(lambda t: {"label": "yes"}, confidence=1.0))

    class Answer(BaseModel):
        answer: Literal["yes", "no"]

    wrapped = integration.wrap(client, m)
    r = asyncio.run(wrapped.chat.completions.parse(model="m", messages=USER, response_format=Answer))
    assert r.choices[0].message.parsed == Answer(answer="yes") and not seen
    r = asyncio.run(wrapped.chat.completions.create(model="m", messages=USER))
    assert r.id == "chatcmpl-1" and len(seen) == 1
    with pytest.raises(TypeError):
        integration.wrap(client, "nope")
