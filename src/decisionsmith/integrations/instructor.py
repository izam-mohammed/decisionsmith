"""Instructor (`pip install "decisionsmith[instructor]"`): answer from a model or harness first, fall back to the LLM.

client = wrap(instructor.from_openai(OpenAI()), ds.model(Ticket))
client.create(response_model=Ticket, messages=[...], model="gpt-5-mini")   # the model answers when it is sure
"""

from __future__ import annotations

from typing import Any

from .litellm import last_text
from .openai import Proxy, as_harness, fill, fits


def wrap(client: Any, x: Any) -> Any:
    """An Instructor client whose `create(response_model=...)` (also via `chat.completions` and `messages`) is
    answered by `x` (a `ds.model` or `ds.harness`) when `response_model` only asks for its fields and it is sure;
    otherwise Instructor makes its usual LLM call. Everything else goes to `client` unchanged."""
    import instructor

    h = as_harness(x)

    def text(response_model: Any, messages: Any) -> str:
        return last_text(messages) if fits(h, response_model) else ""

    def create(response_model: Any, messages: Any, **kw: Any) -> Any:
        t = text(response_model, messages)
        hit = fill(h, response_model, h.decide(t)) if t.strip() else None
        return hit[1] if hit else client.create(response_model=response_model, messages=messages, **kw)

    async def acreate(response_model: Any, messages: Any, **kw: Any) -> Any:
        t = text(response_model, messages)
        hit = fill(h, response_model, await h.adecide(t)) if t.strip() else None
        return hit[1] if hit else await client.create(response_model=response_model, messages=messages, **kw)

    wrapped = Proxy(client, create=acreate if isinstance(client, instructor.AsyncInstructor) else create)
    wrapped._overrides.update(chat=wrapped, completions=wrapped, messages=wrapped)
    return wrapped
