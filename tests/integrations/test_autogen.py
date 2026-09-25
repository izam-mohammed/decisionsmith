import asyncio

import pytest

pytest.importorskip("autogen_core")

from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import openai_client

Q = compile_schema(Ticket).questions()


def test_openai_client_is_a_teacher():
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    c = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="sk-test")
    c._client, seen = openai_client()
    e = from_string(c)
    assert e.name == "autogen:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0

    async def inside_a_loop():
        return e.ask("x", Q)

    assert asyncio.run(inside_a_loop())["answers"]["team"]["choice"] == "billing"
    assert '"role":"system"' in seen[0].content.decode().replace(" ", "")
