import asyncio

import pytest

pytest.importorskip("smolagents")

from decisionsmith.engines import from_string
from decisionsmith.integrations.smolagents import _text
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import openai_client

Q = compile_schema(Ticket).questions()


def test_openai_server_model_is_a_teacher():
    from smolagents import OpenAIServerModel

    m = OpenAIServerModel(model_id="gpt-4o-mini", api_key="sk-test")
    m.client, seen = openai_client(is_async=False)
    e = from_string(m)
    assert e.name == "smolagents:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert '"role":"system"' in seen[0].content.decode().replace(" ", "")


def test_content_parts():
    assert _text([{"type": "text", "text": "a"}, "x", {"text": "b"}]) == "ab" and _text(None) == ""
