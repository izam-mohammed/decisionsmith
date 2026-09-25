"""Google ADK (`pip install "decisionsmith[google-adk]"`): a decision as a `FunctionTool`, and a
`before_model_callback` guard that answers instead of the model when a request is blocked.

agent = LlmAgent(model="gemini-2.5-flash", tools=[tool(ds.harness(Ticket, ...), name="route_ticket")],
                 before_model_callback=guard(ds.harness(Injection, ...), field="is_attack", block=[True]))
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._base import resolve


def tool(x: Any, name: str = "decide", description: str | None = None) -> Any:
    """A `FunctionTool` named `name` that takes `text` and returns the decision as a dict (`{"label": ...}` for a
    `ds.model(labels)`)."""
    from google.adk.tools import FunctionTool

    d = resolve(x)

    async def decide(text: str) -> dict[str, Any]:
        value = await d.acall(text)
        return {"label": value} if d.simple else value.model_dump(mode="json")

    decide.__name__ = decide.__qualname__ = name
    decide.__doc__ = description or "Decide %s for a text. Returns one value per field." % ", ".join(d.fields)
    return FunctionTool(decide)


def last_user_text(llm_request: Any) -> str:
    """The text of the last user turn in an `LlmRequest` (tool responses carry no text and are skipped)."""
    for content in reversed(list(llm_request.contents or [])):
        texts = [p.text for p in content.parts or [] if getattr(p, "text", None)]
        if content.role == "user" and texts:
            return " ".join(texts)
    return ""


def guard(
    x: Any, field: str | None = None, block: Iterable[Any] = (True,), message: str = "Sorry, I can't help with that."
) -> Any:
    """An async `before_model_callback(callback_context, llm_request)`: when the decision's `field` for the last
    user text is one of `block`, the model is skipped and `message` is the reply (`LlmResponse`); else `None`."""
    from google.adk.models import LlmResponse
    from google.genai import types

    d, blocked = resolve(x), list(block)

    async def before_model(callback_context: Any, llm_request: Any) -> Any:
        text = last_user_text(llm_request)
        if not text.strip() or d.field(await d.acall(text), field) not in blocked:
            return None
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=message)]))

    return before_model
