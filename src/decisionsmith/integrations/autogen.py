"""AutoGen (`pip install "decisionsmith[autogen]"`): any AutoGen `ChatCompletionClient` as the teacher.

from autogen_ext.models.openai import OpenAIChatCompletionClient
ds.harness(Ticket, teacher=OpenAIChatCompletionClient(model="gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name, run_sync


def teacher(client: Any) -> TextEngine:
    """An AutoGen (0.4+) model client (OpenAI, Azure OpenAI, Anthropic, Ollama, ...) as a decisionsmith teacher."""

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        from autogen_core.models import SystemMessage, UserMessage

        result = await client.create([SystemMessage(content=system), UserMessage(content=user, source="user")])
        u = result.usage
        return str(result.content), {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return run_sync(lambda: acomplete(system, user))

    info = getattr(client, "_raw_config", None) or {}
    return TextEngine("autogen:%s" % info.get("model", model_name(client)), complete, acomplete)
