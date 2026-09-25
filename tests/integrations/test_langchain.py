import asyncio

import pytest

pytest.importorskip("langchain_core")

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.integrations import langchain as integration
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
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


def harness():
    from decisionsmith.testing import FakeEngine
    from tests.conftest import truth

    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def test_decision_runnable_composes_and_batches():
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import Runnable, RunnableLambda

    from decisionsmith.integrations.langchain import DecisionRunnable

    r = DecisionRunnable(harness(), "team")
    assert isinstance(r, Runnable) and r.name == "decisionsmith:t"
    assert r.invoke("my invoice is wrong") == "billing"
    assert r.invoke(HumanMessage("the app crashes on login")) == "technical"
    assert r.invoke({"question": "can I get a quote"}) == "sales"
    assert r.batch(["sync is broken", [("human", "how much is the plan")]]) == ["technical", "sales"]
    chain = RunnableLambda(lambda t: t.upper().lower()) | r
    assert asyncio.run(chain.ainvoke("refund my card charge")) == "billing"
    assert asyncio.run(r.abatch(["sync is broken"])) == ["technical"]
    whole = DecisionRunnable(harness()).invoke("refund my card charge")
    assert isinstance(whole, Ticket) and whole.wants_refund
    labels = ds.model(["spam", "ham"], FakeEngine(lambda t: {"label": "spam" if "win" in t else "ham"}))
    assert DecisionRunnable(labels).invoke({"input": "win a prize"}) == "spam"
    for empty in ({"other": 1}, []):
        with pytest.raises(ValueError, match="non-empty"):
            DecisionRunnable(labels).invoke(empty)
    with pytest.raises(AttributeError):
        integration.Nope


def test_as_tool_is_a_langchain_tool():
    from langchain_core.tools import BaseTool

    from decisionsmith.integrations.langchain import as_tool

    tool = as_tool(harness(), "route_ticket", "Which team handles a support ticket")
    assert isinstance(tool, BaseTool) and tool.name == "route_ticket" and "text" in tool.args
    assert '"team":"billing"' in tool.invoke({"text": "my invoice is wrong"})
    team = as_tool(harness(), "team", "Team", field="team")
    assert asyncio.run(team.ainvoke({"text": "sync is broken"})) == "technical"


def test_output_parser_after_a_chat_model():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from decisionsmith.integrations.langchain import output_parser

    llm = FakeListChatModel(responses=["The customer says the app crashes on login.", "you charged me twice"])
    chain = llm | output_parser(harness(), "team")
    assert chain.invoke("summarise") == "technical"
    assert asyncio.run(chain.ainvoke("summarise")) == "billing"


def test_compressor_filters_retrieved_documents():
    from langchain_core.documents import BaseDocumentCompressor, Document

    from decisionsmith.integrations.langchain import compressor

    docs = [Document("refund my card charge"), Document("sync is broken"), Document("the payment failed")]
    billing = compressor(harness(), "team", keep=["billing"], with_query=False)
    assert isinstance(billing, BaseDocumentCompressor)
    assert [d.page_content for d in billing.compress_documents(docs, "q")] == [
        "refund my card charge",
        "the payment failed",
    ]
    seen = []
    m = ds.model(["yes", "no"], FakeEngine(lambda t: seen.append(t) or {"label": "yes" if "sync" in t else "no"}))
    relevant = compressor(m, keep=["yes"])
    out = asyncio.run(relevant.acompress_documents(docs, "why is it down"))
    assert [d.page_content for d in out] == ["sync is broken"]
    assert seen[0].startswith("Query: why is it down\n\nDocument: ")
