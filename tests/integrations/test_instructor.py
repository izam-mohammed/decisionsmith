import asyncio
from typing import Literal

import pytest

instructor = pytest.importorskip("instructor")

from pydantic import BaseModel  # noqa: E402

import decisionsmith as ds  # noqa: E402
from decisionsmith.integrations import instructor as integration  # noqa: E402
from decisionsmith.testing import FakeEngine  # noqa: E402
from tests.conftest import Ticket, truth  # noqa: E402
from tests.integrations.openai_mock import openai_client  # noqa: E402

USER = [{"role": "user", "content": "the app crashes on login"}]


class Team(BaseModel):
    team: Literal["billing", "technical", "sales"]


class Summary(BaseModel):
    summary: str


def harness(confidence):
    return ds.harness(Ticket, student=FakeEngine(truth, confidence=confidence), log=None)


def test_sure_answers_skip_the_llm(monkeypatch):
    monkeypatch.delenv("DS_OFFLINE", raising=False)
    raw, seen = openai_client(is_async=False, content='{"team": "sales"}')
    client = integration.wrap(instructor.from_openai(raw, mode=instructor.Mode.JSON), harness(1.0))
    assert client.create(response_model=Team, messages=USER, model="gpt-5-mini") == Team(team="technical")
    assert client.chat.completions.create(response_model=Ticket, messages=USER, model="m").wants_refund is False
    assert client.messages.create(response_model=Team, messages=USER, model="m").team == "technical" and not seen
    assert not seen
    other, seen = openai_client(is_async=False, content='{"summary": "a crash"}')
    wrapped = integration.wrap(instructor.from_openai(other, mode=instructor.Mode.JSON), harness(1.0))
    assert wrapped.create(response_model=Summary, messages=USER, model="m").summary == "a crash"
    assert len(seen) == 1
    assert client.mode == instructor.Mode.JSON


def test_unsure_falls_back_to_instructor(monkeypatch):
    monkeypatch.delenv("DS_OFFLINE", raising=False)
    raw, seen = openai_client(is_async=False, content='{"team": "sales"}')
    client = integration.wrap(instructor.from_openai(raw, mode=instructor.Mode.JSON), harness(0.3))
    assert client.create(response_model=Team, messages=USER, model="gpt-5-mini").team == "sales"
    assert len(seen) == 1 and "gpt-5-mini" in seen[0].content.decode()


def test_async_client(monkeypatch):
    monkeypatch.delenv("DS_OFFLINE", raising=False)
    raw, seen = openai_client(content='{"team": "sales"}')
    client = integration.wrap(instructor.from_openai(raw, mode=instructor.Mode.JSON), harness(1.0))
    assert asyncio.run(client.create(response_model=Team, messages=USER, model="m")) == Team(team="technical")
    empty = [{"role": "system", "content": "x"}]
    assert asyncio.run(client.create(response_model=Team, messages=empty, model="m")).team == "sales"
    assert len(seen) == 1
