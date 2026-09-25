import asyncio

import pytest

pytest.importorskip("haystack")

from decisionsmith.engines import from_string
from decisionsmith.integrations.haystack import teacher
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import GOOD, openai_client

Q = compile_schema(Ticket).questions()


def test_openai_chat_generator_is_a_teacher():
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.utils import Secret

    g = OpenAIChatGenerator(model="gpt-4o-mini", api_key=Secret.from_token("sk-test"))
    g.client, seen = openai_client(is_async=False)
    g.async_client, _ = openai_client()
    e = from_string(g)
    assert e.name == "haystack:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert '"role":"system"' in seen[0].content.decode().replace(" ", "")


def test_custom_component_without_async():
    from haystack import component
    from haystack.dataclasses import ChatMessage

    @component
    class Echo:
        @component.output_types(replies=list[ChatMessage])
        def run(self, messages: list[ChatMessage]):
            return {"replies": [ChatMessage.from_assistant(GOOD)]}

    e = teacher(Echo())
    assert e.acomplete is None and asyncio.run(e.aask("x", Q))["answers"]["team"]["choice"] == "billing"
