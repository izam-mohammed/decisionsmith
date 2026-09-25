"""LlamaIndex (`pip install "decisionsmith[llamaindex]"`): any LlamaIndex LLM as the teacher, a relevance filter,
a selector and a `FunctionTool`.

ds.harness(Ticket, teacher=OpenAI(model="gpt-5-mini"))            # detected automatically
index.as_query_engine(node_postprocessors=[relevance_filter(ds.model(["relevant", "off-topic"]))])
RouterQueryEngine(selector=selector(ds.model(["billing", "technical"])), query_engine_tools=[...], llm=...)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..engines.structured import TextEngine
from ._base import model_name, resolve


def _messages(system: str, user: str) -> list[Any]:
    from llama_index.core.llms import ChatMessage

    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def _reply(response: Any) -> tuple[str, dict[str, Any]]:
    counts = getattr(response, "additional_kwargs", None) or {}
    usage = {"input_tokens": counts.get("prompt_tokens"), "output_tokens": counts.get("completion_tokens")}
    return str(response.message.content or ""), usage


def teacher(llm: Any) -> TextEngine:
    """A LlamaIndex LLM (OpenAI, Anthropic, Ollama, Bedrock, ...) as a decisionsmith teacher."""

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(llm.chat(_messages(system, user)))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(await llm.achat(_messages(system, user)))

    return TextEngine("llamaindex:%s" % model_name(llm), complete, acomplete)


def relevance_filter(x: Any, field: str | None = None, keep: Iterable[Any] = ("relevant",)) -> Any:
    """A node postprocessor that keeps the nodes whose decision `field` is one of `keep`. The model reads
    `Query: ...` and `Passage: ...` (just the passage when there is no query)."""
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

    d, kept = resolve(x), list(keep)

    def text(node: Any, query: Any) -> str:
        passage = node.node.get_content()
        return "Query: %s\n\nPassage: %s" % (query.query_str, passage) if query else passage

    class RelevanceFilter(BaseNodePostprocessor):
        def _postprocess_nodes(self, nodes: list[Any], query_bundle: Any = None) -> list[Any]:
            return [n for n in nodes if d.field(d.call(text(n, query_bundle)), field) in kept]

        async def _apostprocess_nodes(self, nodes: list[Any], query_bundle: Any = None) -> list[Any]:
            return [n for n in nodes if d.field(await d.acall(text(n, query_bundle)), field) in kept]

    return RelevanceFilter()


def selector(x: Any, field: str | None = None) -> Any:
    """A selector (for `RouterQueryEngine`, ...) that picks the choice whose name (or description) is the
    decision's `field` for the query."""
    from llama_index.core.base.base_selector import BaseSelector, SelectorResult, SingleSelection

    d = resolve(x)

    def pick(choices: Any, value: Any) -> Any:
        for i, c in enumerate(choices):
            if str(value) in (c.name, c.description):
                return SelectorResult(selections=[SingleSelection(index=i, reason="decisionsmith: %s" % value)])
        raise ValueError("the decision %r is not the name of any choice" % (value,))

    class DecisionSelector(BaseSelector):
        def _get_prompts(self) -> dict[str, Any]:
            return {}

        def _update_prompts(self, prompts: Any) -> None:
            pass

        def _select(self, choices: Any, query: Any) -> Any:
            return pick(choices, d.field(d.call(query.query_str), field))

        async def _aselect(self, choices: Any, query: Any) -> Any:
            return pick(choices, d.field(await d.acall(query.query_str), field))

    return DecisionSelector()


def _out(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def as_tool(x: Any, name: str = "decide", description: str | None = None) -> Any:
    """A LlamaIndex `FunctionTool` that decides a text and returns the decision (a dict, or the label)."""
    from llama_index.core.tools import FunctionTool

    d = resolve(x)

    def decide(text: str) -> Any:
        return _out(d.call(text))

    async def adecide(text: str) -> Any:
        return _out(await d.acall(text))

    desc = description or "Decide %s for a text; returns the answer." % ", ".join(d.fields)
    return FunctionTool.from_defaults(fn=decide, async_fn=adecide, name=name, description=desc)
