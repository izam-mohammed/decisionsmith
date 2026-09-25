"""LangChain (`uv add "decisionsmith[langchain]"`): chat models as teachers, and decisions inside chains.

ds.harness(Ticket, teacher=ChatOpenAI(model="gpt-5-mini"))            # any chat model, detected automatically
chain = prompt | llm | DecisionRunnable(h, "team")                    # invoke / ainvoke / batch, composes with |
as_tool(h, "route_ticket", "Which team handles a ticket"); output_parser(h); compressor(h, "relevant")
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name, resolve

_CLASSES: dict[str, type] = {}


def _text(value: Any) -> str:
    text = getattr(value, "text", None)
    if text is not None:
        return str(text() if callable(text) else text)
    if isinstance(value, dict):
        return str(next((value[k] for k in ("text", "input", "question", "query") if k in value), ""))
    if isinstance(value, (list, tuple)):
        return _text(value[-1]) if value else ""
    return str(value)


def _reply(message: Any) -> tuple[str, dict[str, Any]]:
    u = getattr(message, "usage_metadata", None) or {}
    return _text(message), {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")}


def teacher(llm: Any) -> TextEngine:
    """A LangChain chat model (ChatOpenAI, ChatAnthropic, ChatOllama, ...) as a decisionsmith teacher."""

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(llm.invoke([("system", system), ("human", user)]))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(await llm.ainvoke([("system", system), ("human", user)]))

    return TextEngine("langchain:%s" % model_name(llm), complete, acomplete)


def _runnable() -> type:
    from langchain_core.runnables import Runnable

    class DecisionRunnable(Runnable[Any, Any]):
        """A Runnable whose output is the decision (or one `field` of it) for a text, message or dict input."""

        def __init__(self, x: Any, field: str | None = None) -> None:
            self.decider, self.field = resolve(x), field
            self.name = "decisionsmith:%s" % self.decider.name

        def _decide(self, value: Any) -> Any:
            return self.decider.field(self.decider.call(_text(value)), self.field)

        async def _adecide(self, value: Any) -> Any:
            return self.decider.field(await self.decider.acall(_text(value)), self.field)

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            return self._call_with_config(self._decide, input, config)

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            return await self._acall_with_config(self._adecide, input, config)

    return DecisionRunnable


def __getattr__(name: str) -> Any:
    if name == "DecisionRunnable":
        return _CLASSES.setdefault(name, _runnable())
    raise AttributeError(name)


def _show(value: Any) -> str:
    return value.model_dump_json() if hasattr(value, "model_dump_json") else str(value)


def as_tool(x: Any, name: str, description: str, field: str | None = None) -> Any:
    """A LangChain `StructuredTool` taking `text` and returning the decision (JSON, a label, or one field)."""
    from langchain_core.tools import StructuredTool

    d = resolve(x)

    def decide(text: str) -> str:
        return _show(d.field(d.call(text), field))

    async def adecide(text: str) -> str:
        return _show(d.field(await d.acall(text), field))

    return StructuredTool.from_function(decide, adecide, name=name, description=description)


def output_parser(x: Any, field: str | None = None) -> Any:
    """An output parser (`llm | output_parser(h)`): the LLM's text in, the decision (or one field) out."""
    from langchain_core.output_parsers import BaseOutputParser

    d = resolve(x)

    class DecisionParser(BaseOutputParser[Any]):
        def parse(self, text: str) -> Any:
            return d.field(d.call(text), field)

        async def aparse(self, text: str) -> Any:
            return d.field(await d.acall(text), field)

        async def aparse_result(self, result: list[Any], *, partial: bool = False) -> Any:
            return await self.aparse(result[0].text)

    return DecisionParser()


def compressor(x: Any, field: str | None = None, keep: Iterable[Any] = (True,), with_query: bool = True) -> Any:
    """A `BaseDocumentCompressor` (for `ContextualCompressionRetriever`) keeping the documents whose decision's
    `field` is one of `keep`; the decided text is the query and the document, or the document alone."""
    from langchain_core.documents import BaseDocumentCompressor

    d, kept = resolve(x), list(keep)

    def text(doc: Any, query: str) -> str:
        return "Query: %s\n\nDocument: %s" % (query, doc.page_content) if with_query else doc.page_content

    class DecisionFilter(BaseDocumentCompressor):
        def compress_documents(self, documents: Any, query: str, callbacks: Any = None) -> Any:
            return [doc for doc in documents if d.field(d.call(text(doc, query)), field) in kept]

        async def acompress_documents(self, documents: Any, query: str, callbacks: Any = None) -> Any:
            values = await asyncio.gather(*(d.acall(text(doc, query)) for doc in documents))
            return [doc for doc, v in zip(documents, values) if d.field(v, field) in kept]

    return DecisionFilter()
