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
