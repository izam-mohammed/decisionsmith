"""Haystack (`pip install "decisionsmith[haystack]"`): any Haystack chat generator as the teacher.

from haystack.components.generators.chat import OpenAIChatGenerator
ds.harness(Ticket, teacher=OpenAIChatGenerator(model="gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name


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


def teacher(generator: Any) -> TextEngine:
    """A Haystack chat generator (OpenAI, Anthropic, Google, Ollama, Hugging Face, ...) as a decisionsmith teacher."""

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(generator.run(messages=_messages(system, user)))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(await generator.run_async(messages=_messages(system, user)))

    return TextEngine(
        "haystack:%s" % model_name(generator), complete, acomplete if hasattr(generator, "run_async") else None
    )
