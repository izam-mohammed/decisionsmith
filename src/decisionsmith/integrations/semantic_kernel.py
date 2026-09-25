"""Semantic Kernel (`pip install "decisionsmith[semantic-kernel]"`): any chat completion service as the teacher.

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
ds.harness(Ticket, teacher=OpenAIChatCompletion(ai_model_id="gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name, run_sync


def teacher(service: Any) -> TextEngine:
    """A Semantic Kernel chat completion service (OpenAI, Azure OpenAI, Anthropic, Ollama, ...) as a teacher."""

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        from semantic_kernel.contents import ChatHistory

        history = ChatHistory()
        history.add_system_message(system)
        history.add_user_message(user)
        message = await service.get_chat_message_content(history, service.get_prompt_execution_settings_class()())
        u = (getattr(message, "metadata", None) or {}).get("usage")
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None),
        }
        return str(message.content if message is not None else ""), usage

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return run_sync(lambda: acomplete(system, user))

    return TextEngine("semantic-kernel:%s" % model_name(service), complete, acomplete)
