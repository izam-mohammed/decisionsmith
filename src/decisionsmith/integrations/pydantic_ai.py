"""Pydantic AI (`pip install "decisionsmith[pydantic-ai]"`): any Pydantic AI model as the teacher.

from pydantic_ai.models.openai import OpenAIChatModel
ds.harness(Ticket, teacher=OpenAIChatModel("gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name


def _reply(result: Any) -> tuple[str, dict[str, Any]]:
    u = result.usage
    u = u() if callable(u) else u
    return str(result.output), {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}


def teacher(model: Any) -> TextEngine:
    """A Pydantic AI model (OpenAI, Anthropic, Gemini, Groq, Mistral, Bedrock, ...) as a decisionsmith teacher."""
    from pydantic_ai import Agent

    agents: dict[str, Any] = {}

    def agent(system: str) -> Any:
        if system not in agents:
            agents[system] = Agent(model, instructions=system)
        return agents[system]

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(agent(system).run_sync(user))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(await agent(system).run(user))

    return TextEngine("pydantic-ai:%s" % model_name(model), complete, acomplete)
