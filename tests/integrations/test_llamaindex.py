import asyncio

import pytest

pytest.importorskip("llama_index.core")

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import http_client

Q = compile_schema(Ticket).questions()


def test_openai_llm_is_a_teacher():
    from llama_index.llms.openai import OpenAI

    sync, _ = http_client(is_async=False)
    async_, _ = http_client()
    e = from_string(OpenAI(model="gpt-4o-mini", api_key="sk-test", http_client=sync, async_http_client=async_))
    assert e.name == "llamaindex:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0


def test_any_llama_index_llm_works():
    from llama_index.core.llms import MockLLM

    h = ds.harness(Ticket, teacher=MockLLM(), log=None)
    assert isinstance(h("my invoice is wrong"), Ticket) and h.teacher.name == "llamaindex:MockLLM"
