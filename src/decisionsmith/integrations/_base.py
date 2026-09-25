"""Shared by every integration: accept a `ds.model(...)` or a `ds.harness(...)`, and wrap LLM calls as teachers."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from ..engines.structured import Reply, TextEngine


@dataclass
class Decider:
    """What an integration needs from a model or harness: sync and async calls, and the schema."""

    call: Callable[[str], Any]
    acall: Callable[[str], Awaitable[Any]]
    fields: list[str]
    simple: bool
    name: str

    def field(self, value: Any, field: str | None = None) -> Any:
        """One field of a decision (`"billing"`), or the label for `ds.model(labels)`."""
        if self.simple or field is None:
            return value
        return getattr(value, field)


def resolve(x: Any) -> Decider:
    """A `ds.model(...)` or `ds.harness(...)` (or anything with `__call__`/`predict` and `schema`)."""
    from ..core import Harness
    from ..predictor import Model

    if isinstance(x, Model):
        return Decider(x.predict, x.apredict, list(x.schema.fields), x.simple, x.name)
    if isinstance(x, Harness):
        name = "+".join(e.name for e in (x.student, x.teacher) if e)
        return Decider(x, x.acall, list(x.schema.fields), x.simple, name)
    raise TypeError("expected ds.model(...) or ds.harness(...), got %r" % (x,))


def teacher(name: str, complete: Callable[[str, str], Reply], acomplete: Any = None) -> TextEngine:
    """A teacher engine from `complete(system, user) -> str` (and optionally its async twin)."""
    return TextEngine(name, complete, acomplete)


_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()


def run_sync(make: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run a coroutine from sync code on one shared background event loop, so async-only clients keep their
    connections on a single loop, and it works from notebooks and threads that already run a loop."""
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            threading.Thread(target=_LOOP.run_forever, name="decisionsmith-async", daemon=True).start()
    return asyncio.run_coroutine_threadsafe(make(), _LOOP).result()


def model_name(obj: Any) -> str:
    """A readable name for a framework LLM object: its model id when it has one."""
    for attr in ("model_name", "model", "model_id", "ai_model_id", "name"):
        v = getattr(obj, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(obj).__name__
