"""LangChain (`pip install "decisionsmith[langchain]"`): any LangChain chat model as the teacher.

from langchain_openai import ChatOpenAI
ds.harness(Ticket, teacher=ChatOpenAI(model="gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name


def _reply(message: Any) -> tuple[str, dict[str, Any]]:
    text = message.text
    text = text() if callable(text) else text
    u = getattr(message, "usage_metadata", None) or {}
    return str(text), {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")}


def teacher(llm: Any) -> TextEngine:
    """A LangChain chat model (ChatOpenAI, ChatAnthropic, ChatOllama, ...) as a decisionsmith teacher."""

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(llm.invoke([("system", system), ("human", user)]))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(await llm.ainvoke([("system", system), ("human", user)]))

    return TextEngine("langchain:%s" % model_name(llm), complete, acomplete)
