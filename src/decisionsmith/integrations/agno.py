"""Agno (`pip install "decisionsmith[agno]"`): any Agno model as the teacher, and a decision as an Agno tool.

ds.harness(Ticket, teacher=OpenAIChat(id="gpt-5-mini"))                 # detected automatically
Agent(model=OpenAIChat(id="gpt-5-mini"), tools=[tool(h, name="route_ticket")])
"""

from __future__ import annotations

import json
from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name, resolve


def _messages(system: str, user: str) -> list[Any]:
    from agno.models.message import Message

    return [Message(role="system", content=system), Message(role="user", content=user)]


def _reply(response: Any) -> tuple[str, dict[str, Any]]:
    u = response.response_usage
    usage = {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)}
    return str(response.content or ""), usage


def teacher(model: Any) -> TextEngine:
    """An Agno model (OpenAIChat, Claude, Gemini, Groq, Ollama, LiteLLM, ...) as a decisionsmith teacher."""

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(model.response(messages=_messages(system, user)))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(await model.aresponse(messages=_messages(system, user)))

    return TextEngine("agno:%s" % (getattr(model, "id", None) or model_name(model)), complete, acomplete)


def tool(x: Any, name: str = "decide", description: str | None = None) -> Any:
    """An Agno tool (`Function`) that takes `text` and returns the decision as JSON (`{"label": ...}` for a
    `ds.model(labels)`); works with `agent.run` and `agent.arun`."""
    from agno.tools import tool as agno_tool

    d = resolve(x)

    def decide(text: str) -> str:
        value = d.call(text)
        return json.dumps({"label": value} if d.simple else value.model_dump(mode="json"))

    about = description or "Decide %s for a text. Returns one value per field." % ", ".join(d.fields)
    return agno_tool(name=name, description=about)(decide)
