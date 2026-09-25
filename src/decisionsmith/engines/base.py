"""The engine contract: the Jev `/v1/systemone` shape, its errors, and engine strings."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol, overload, runtime_checkable


class EngineError(RuntimeError):
    """An engine could not answer. Carries the engine name and how to fix it."""

    def __init__(self, engine: str, message: str, fix: str = "") -> None:
        self.engine, self.message, self.fix = engine, message, fix
        super().__init__("engine %r: %s%s" % (engine, message, ("\n  fix: " + fix) if fix else ""))


@runtime_checkable
class Engine(Protocol):
    """Answers Jev-format questions about one text: returns `{"answers": {...}, "usage": {...}}`."""

    name: str

    def ask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]: ...


def need(engine: str, message: str, extra: str, *modules: str) -> list[Any]:
    """Import optional dependencies, or raise an EngineError that says which extra to install."""
    try:
        return [__import__(m) for m in modules]
    except ImportError:
        raise EngineError(engine, message, "pip install 'decisionsmith[%s]'" % extra) from None


def response(engine: Any, raw: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an engine's reply; accepts the full `{answers, usage}` response or a bare answers dict."""
    if not isinstance(raw, dict):
        raise EngineError(engine.name, "returned %s, expected a dict" % type(raw).__name__)
    if isinstance(raw.get("answers"), dict) and set(raw) <= {"answers", "usage", "model", "routing"}:
        answers, usage = raw["answers"], raw.get("usage") or {}
    else:
        answers, usage = raw, {}
    missing = [q for q in questions if q not in answers]
    if missing:
        raise EngineError(engine.name, "no answer for %s" % missing)
    return answers, usage if isinstance(usage, dict) else {}


def ask_many(engine: Engine, texts: Sequence[str], questions: dict[str, Any]) -> list[Any]:
    batch = getattr(engine, "ask_many", None)
    if callable(batch):
        return list(batch(list(texts), questions))
    return [engine.ask(t, questions) for t in texts]


@overload
def from_string(spec: None) -> None: ...


@overload
def from_string(spec: Any) -> Engine: ...


def from_string(spec: Any) -> Engine | None:
    """`"claude-sonnet-5"`, `"jev"`, `"laya:multilingual"`, `"systemone:http://..."`, `"fake"`, or an Engine."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        if not callable(getattr(spec, "ask", None)) or not isinstance(getattr(spec, "name", None), str):
            raise TypeError("an engine needs a `name` str and an `ask(text, questions)` method, got %r" % (spec,))
        return spec
    s = spec.strip()
    head, _, rest = s.partition(":")
    if s == "fake":
        from ..testing import FakeEngine

        return FakeEngine()
    if head == "jev":
        from .systemone import JEV_MODEL, JevEngine

        return JevEngine(rest or JEV_MODEL)
    if head == "systemone":
        from .systemone import SystemOneEngine

        if not rest:
            raise ValueError("systemone needs a URL: 'systemone:http://localhost:8000'")
        return SystemOneEngine(rest, api_key=os.environ.get("SYSTEMONE_API_KEY"))
    if head == "laya":
        from .laya import LayaEngine

        return LayaEngine(rest or "laya")
    if not s:
        raise ValueError("empty engine string")
    from .llm import LLMEngine

    return LLMEngine(s)
