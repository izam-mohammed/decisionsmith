import asyncio

import pytest

pytest.importorskip("semantic_kernel")

from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import openai_client

Q = compile_schema(Ticket).questions()


def test_openai_chat_completion_is_a_teacher():
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    client, seen = openai_client()
    e = from_string(OpenAIChatCompletion(ai_model_id="gpt-4o-mini", async_client=client))
    assert e.name == "semantic-kernel:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert '"role":"system"' in seen[0].content.decode().replace(" ", "")
