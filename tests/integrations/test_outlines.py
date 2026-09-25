import asyncio
import json

import pytest

outlines = pytest.importorskip("outlines")

import decisionsmith as ds  # noqa: E402
from decisionsmith.engines import from_string  # noqa: E402
from decisionsmith.integrations import outlines as integration  # noqa: E402
from decisionsmith.schema import compile_schema  # noqa: E402
from decisionsmith.testing import FakeEngine  # noqa: E402
from tests.conftest import Ticket, truth  # noqa: E402
from tests.integrations.openai_mock import openai_client  # noqa: E402

Q = compile_schema(Ticket).questions()


def test_openai_backed_model_uses_structured_output(monkeypatch):
    monkeypatch.delenv("DS_OFFLINE", raising=False)
    client, seen = openai_client(is_async=False)
    e = from_string(outlines.from_openai(client, "gpt-5-mini"))
    assert e.name == "outlines:gpt-5-mini"
    assert e.ask("you charged me twice", Q)["answers"]["team"]["choice"] == "billing"
    body = json.loads(seen[0].content)
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["q0"]["enum"] == ["billing", "technical", "sales"]
    assert body["messages"][0]["role"] == "system" and "<text>" in body["messages"][1]["content"]
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0


def test_async_model_and_writing(monkeypatch):
    monkeypatch.delenv("DS_OFFLINE", raising=False)
    client, _ = openai_client()
    e = integration.teacher(outlines.from_openai(client, "gpt-5-mini"), max_tokens=50)
    assert e.is_async and asyncio.run(e.aask("x", Q))["answers"]["team"]["choice"] == "billing"
    assert e.ask("x", Q)["answers"]["team"]["choice"] == "billing"
    w, _ = openai_client(is_async=False, content='{"texts": ["a", "b"]}')
    assert integration.teacher(outlines.from_openai(w, "gpt-5-mini")).write("write 2", 2) == ["a", "b"]


def test_offline_and_training(monkeypatch):
    client, seen = openai_client(is_async=False)
    e = integration.teacher(outlines.from_openai(client, "gpt-5-mini"))
    monkeypatch.setenv("DS_OFFLINE", "1")
    assert e.ask("my invoice is wrong: billing", Q)["answers"]["team"]["choice"] == "billing"
    assert len(e.write("write 3 texts", 3)) == 3 and not seen
    m = ds.model(Ticket, FakeEngine(truth))
    assert len(m.label(["sync is broken"], e)) == 1
