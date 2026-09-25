import asyncio

import pytest

pytest.importorskip("llama_index.core")

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth
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


def rel():
    def judge(text):
        return {"label": "relevant" if "refund" in text.split("Passage: ")[-1] else "off-topic"}

    return ds.model(["relevant", "off-topic"], FakeEngine(judge))


def test_relevance_filter_keeps_matching_nodes():
    from llama_index.core.schema import NodeWithScore, TextNode

    from decisionsmith.integrations.llamaindex import relevance_filter

    nodes = [NodeWithScore(node=TextNode(text=t), score=1.0) for t in ["refunds take 5 days", "our office is in Perth"]]
    f = relevance_filter(rel())
    assert [n.node.text for n in f.postprocess_nodes(nodes, query_str="how do refunds work")] == ["refunds take 5 days"]
    assert len(f.postprocess_nodes(nodes)) == 1
    out = asyncio.run(f.apostprocess_nodes(nodes, query_str="q"))
    assert [n.node.text for n in out] == ["refunds take 5 days"]
    h = ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)
    tickets = [NodeWithScore(node=TextNode(text=t)) for t in ["my invoice is wrong", "sync is broken"]]
    assert [n.node.text for n in relevance_filter(h, "team", keep=["technical"]).postprocess_nodes(tickets)] == [
        "sync is broken"
    ]


def test_relevance_filter_in_a_query_engine():
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
    from llama_index.core.schema import TextNode

    from decisionsmith.integrations.llamaindex import relevance_filter

    Settings.embed_model, Settings.llm = MockEmbedding(embed_dim=8), MockLLM()
    index = VectorStoreIndex([TextNode(text="refunds take 5 days"), TextNode(text="our office is in Perth")])
    engine = index.as_query_engine(similarity_top_k=2, node_postprocessors=[relevance_filter(rel())])
    r = engine.query("how do refunds work")
    assert [n.node.text for n in r.source_nodes] == ["refunds take 5 days"]


def test_selector_routes_a_router_query_engine():
    from llama_index.core.llms import MockLLM
    from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
    from llama_index.core.tools import QueryEngineTool, ToolMetadata

    from decisionsmith.integrations.llamaindex import selector

    class Echo(CustomQueryEngine):
        team: str

        def custom_query(self, query_str):
            return "%s answered %s" % (self.team, query_str)

    h = ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)
    tools = [
        QueryEngineTool(query_engine=Echo(team=t), metadata=ToolMetadata(name=t, description="%s questions" % t))
        for t in ["billing", "technical", "sales"]
    ]
    router = RouterQueryEngine(selector=selector(h, "team"), query_engine_tools=tools, llm=MockLLM())
    assert str(router.query("sync is broken")) == "technical answered sync is broken"
    assert str(asyncio.run(router.aquery("my invoice is wrong"))) == "billing answered my invoice is wrong"
    s = selector(h, "team")
    assert s.select(["sales", "billing"], "can I get a quote").ind == 0 and s.get_prompts() == {}
    s.update_prompts({})
    with pytest.raises(ValueError, match="not the name of any choice"):
        s.select(["a", "b"], "sync is broken")


def test_function_tool_in_an_agent_call():
    from decisionsmith.integrations.llamaindex import as_tool

    h = ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)
    tool = as_tool(h, "route_ticket")
    assert tool.metadata.name == "route_ticket" and "team, wants_refund" in tool.metadata.description
    assert tool.call(text="refund my card charge").raw_output == {"team": "billing", "wants_refund": True}
    assert asyncio.run(tool.acall(text="sync is broken")).raw_output["team"] == "technical"
    assert as_tool(rel(), description="d").call(text="x").raw_output == "off-topic"
    assert tool.metadata.get_parameters_dict()["required"] == ["text"]
