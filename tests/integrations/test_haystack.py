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


def harness():
    import decisionsmith as ds
    from decisionsmith.testing import FakeEngine
    from tests.conftest import truth

    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def rel():
    import decisionsmith as ds
    from decisionsmith.testing import FakeEngine

    def judge(text):
        return {"label": "relevant" if "refund" in text.split("Document: ")[-1] else "off-topic"}

    return ds.model(["relevant", "off-topic"], FakeEngine(judge))


def test_router_sends_text_to_its_value_in_a_pipeline():
    from haystack import Pipeline, component

    from decisionsmith.integrations.haystack import router

    @component
    class Team:
        def __init__(self, name):
            self.name = name

        @component.output_types(reply=str)
        def run(self, text: str):
            return {"reply": "%s: %s" % (self.name, text)}

    pipe = Pipeline()
    pipe.add_component("route", router(harness(), "team"))
    for team in ["billing", "technical", "sales"]:
        pipe.add_component(team, Team(team))
        pipe.connect("route.%s" % team, "%s.text" % team)
    assert pipe.run({"route": {"text": "sync is broken"}}) == {"technical": {"reply": "technical: sync is broken"}}
    r = router(harness(), "wants_refund")
    assert set(r.__haystack_output__._sockets_dict) == {"false", "true"}
    assert r.run(text="refund my card charge") == {"true": "refund my card charge"}
    assert asyncio.run(r.run_async(text="sync is broken")) == {"false": "sync is broken"}


def test_router_groups_documents():
    from haystack import Document, Pipeline

    from decisionsmith.integrations.haystack import router

    docs = [Document(content=t) for t in ["my invoice is wrong", "sync is broken", "refund my card charge"]]
    r = router(harness(), "team", documents=True)
    out = r.run(documents=docs)
    assert [d.content for d in out["billing"]] == ["my invoice is wrong", "refund my card charge"]
    assert [d.content for d in out["technical"]] == ["sync is broken"] and "sales" not in out
    pipe = Pipeline()
    pipe.add_component("route", router(rel(), documents=True))
    result = asyncio.run(pipe.run_async({"route": {"documents": docs}}))
    assert [d.content for d in result["route"]["relevant"]] == ["refund my card charge"]
    assert len(result["route"]["off-topic"]) == 2


def test_document_filter_after_a_retriever():
    from haystack import Document, Pipeline
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    from decisionsmith.integrations.haystack import document_filter

    store = InMemoryDocumentStore()
    store.write_documents([Document(content="refund my card charge"), Document(content="sync is broken, refund me")])

    def pipe(keep):
        p = Pipeline()
        p.add_component("retrieve", InMemoryBM25Retriever(store, top_k=2))
        p.add_component("keep", keep)
        p.connect("retrieve.documents", "keep.documents")
        return p

    ask = {"retrieve": {"query": "refund"}, "keep": {"query": "refund"}}
    billing = pipe(document_filter(harness(), "team", keep=["billing"], with_query=False))
    assert [d.content for d in billing.run(ask)["keep"]["documents"]] == ["refund my card charge"]
    technical = pipe(document_filter(harness(), "team", keep=["technical"], with_query=False))
    assert [d.content for d in asyncio.run(technical.run_async(ask))["keep"]["documents"]] == [
        "sync is broken, refund me"
    ]
    f = document_filter(rel(), keep=["relevant"])
    docs = [Document(content="refunds take 5 days"), Document(content="our office is in Perth")]
    assert [d.content for d in f.run(documents=docs, query="how do refunds work")["documents"]] == [docs[0].content]
    assert [d.content for d in f.run(documents=docs)["documents"]] == [docs[0].content]
    assert len(asyncio.run(f.run_async(documents=docs, query="q"))["documents"]) == 1
