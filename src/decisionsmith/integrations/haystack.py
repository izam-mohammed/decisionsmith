"""Haystack (`uv add "decisionsmith[haystack]"`): any Haystack chat generator as the teacher, and decisions as
pipeline components: a router with one output per value, and a document filter.

ds.harness(Ticket, teacher=OpenAIChatGenerator(model="gpt-5-mini"))  # detected automatically
pipe.add_component("route", router(h, "team"))                   # outputs route.billing, route.technical, ...
pipe.add_component("keep", document_filter(judge, keep=["relevant"]))   # after a retriever
"""

import asyncio
from collections.abc import Iterable
from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name, resolve


def _messages(system: str, user: str) -> list[Any]:
    from haystack.dataclasses import ChatMessage

    return [ChatMessage.from_system(system), ChatMessage.from_user(user)]


def _reply(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    message = result["replies"][0]
    u = (message.meta or {}).get("usage") or {}
    return str(message.text or ""), {
        "input_tokens": u.get("prompt_tokens"),
        "output_tokens": u.get("completion_tokens"),
    }


def _name(generator: Any) -> str:
    v = getattr(generator, "_model", None) or getattr(generator, "_model_or_url", None)
    return v if isinstance(v, str) else model_name(generator)


def teacher(generator: Any) -> TextEngine:
    """A Haystack chat generator (OpenAI, Anthropic, Google, Ollama, Hugging Face, ...) as a decisionsmith teacher."""

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(generator.run(messages=_messages(system, user)))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(await generator.run_async(messages=_messages(system, user)))

    return TextEngine(
        "haystack:%s" % _name(generator), complete, acomplete if hasattr(generator, "run_async") else None
    )


def router(x: Any, field: str | None = None, documents: bool = False) -> Any:
    """A component with one output per value of the decision's `field` (`"true"` / `"false"` for a bool). It takes
    `text` and sends it to its value's output or, with `documents=True`, takes `documents` and sends each one
    (decided on its content) to its value's output; outputs with nothing on them are not emitted."""
    from haystack import Document, component

    d = resolve(x)
    f = x.schema.fields[field or "label"]
    names = dict(zip(f.values, f.labels))

    @component
    class DecisionRouter:
        def __init__(self) -> None:
            kind = list[Document] if documents else str
            component.set_input_types(self, **{"documents" if documents else "text": kind})
            component.set_output_types(self, **dict.fromkeys(f.labels, kind))

        def _group(self, items: list[Any], values: list[Any]) -> dict[str, Any]:
            if not documents:
                return {names[d.field(values[0], field)]: items[0]}
            out: dict[str, list[Any]] = {}
            for item, value in zip(items, values):
                out.setdefault(names[d.field(value, field)], []).append(item)
            return out

        def run(self, **kwargs: Any) -> dict[str, Any]:
            items = kwargs["documents"] if documents else [kwargs["text"]]
            return self._group(items, [d.call(i.content if documents else i) for i in items])

        async def run_async(self, **kwargs: Any) -> dict[str, Any]:
            items = kwargs["documents"] if documents else [kwargs["text"]]
            values = await asyncio.gather(*(d.acall(i.content if documents else i) for i in items))
            return self._group(items, list(values))

    return DecisionRouter()


def document_filter(x: Any, field: str | None = None, keep: Iterable[Any] = (True,), with_query: bool = True) -> Any:
    """A component that keeps the `documents` whose decision's `field` is one of `keep` (for RAG, after a
    retriever). With a `query` input the model reads `Query: ...` and `Document: ...`, else the document alone."""
    from haystack import Document, component

    d, kept = resolve(x), list(keep)

    def text(doc: Any, query: str | None) -> str:
        return "Query: %s\n\nDocument: %s" % (query, doc.content) if with_query and query else doc.content

    @component
    class DecisionFilter:
        @component.output_types(documents=list[Document])
        def run(self, documents: list[Document], query: str | None = None) -> dict[str, Any]:
            return {"documents": [doc for doc in documents if d.field(d.call(text(doc, query)), field) in kept]}

        @component.output_types(documents=list[Document])
        async def run_async(self, documents: list[Document], query: str | None = None) -> dict[str, Any]:
            values = await asyncio.gather(*(d.acall(text(doc, query)) for doc in documents))
            return {"documents": [doc for doc, v in zip(documents, values) if d.field(v, field) in kept]}

    return DecisionFilter()
