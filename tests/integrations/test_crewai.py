import asyncio

import pytest

pytest.importorskip("crewai")

import respx

from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import reply

Q = compile_schema(Ticket).questions()


@respx.mock
def test_crewai_llm_is_a_teacher():
    from crewai import LLM

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=reply())
    e = from_string(LLM(model="gpt-4o-mini", api_key="sk-test"))
    assert e.name == "crewai:gpt-4o-mini"
    assert e.ask("you charged me twice", Q)["answers"]["team"]["choice"] == "billing"
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert '"role":"system"' in route.calls[0].request.content.decode().replace(" ", "")
