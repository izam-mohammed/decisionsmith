import asyncio

import pytest

pytest.importorskip("langchain_core")

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket
from tests.integrations.openai_mock import GOOD, http_client

Q = compile_schema(Ticket).questions()


def test_chat_openai_is_a_teacher():
    from langchain_openai import ChatOpenAI

    sync, seen = http_client(is_async=False)
    async_, _ = http_client()
    e = from_string(ChatOpenAI(model="gpt-4o-mini", api_key="sk-test", http_client=sync, http_async_client=async_))
    assert e.name == "langchain:gpt-4o-mini"
    out = e.ask("you charged me twice, refund", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    sent = seen[0].content.decode()
    assert '"role":"system"' in sent.replace(" ", "") and "<text>" in sent


def test_fake_chat_model_trains_labels():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    llm = FakeListChatModel(responses=["here you go: " + GOOD, '{"texts": ["a", "b"]}'])
    h = ds.harness(Ticket, teacher=llm, log=None)
    assert h("my invoice is wrong").team == "billing"
    assert h.teacher.write("write 2", 2) == ["a", "b"]
