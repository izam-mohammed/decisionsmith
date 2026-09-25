"""LlamaIndex (`pip install "decisionsmith[llamaindex]"`): any LlamaIndex LLM as the teacher.

from llama_index.llms.openai import OpenAI
ds.harness(Ticket, teacher=OpenAI(model="gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name


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
