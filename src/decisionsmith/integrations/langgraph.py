"""LangGraph (`pip install "decisionsmith[langgraph]"`): route conditional edges on a decision, and a guard node.

graph.add_conditional_edges("intake", route_on(h, "team"), {"billing": "billing", "technical": "tech", ...})
graph.add_node("guard", guard_node(ds.harness(Injection, ...), "is_attack", block=[True]))   # sets state["blocked"]
graph.add_conditional_edges("guard", lambda s: s["blocked"], {True: END, False: "agent"})
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._base import resolve


def _content(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def text_of(state: Any, key: str = "messages") -> str:
    """The text at `state[key]`: a string, or the last user message of a message list (LangChain messages,
    `{"role": "user", ...}` dicts)."""
    value = state.get(key) if isinstance(state, dict) else getattr(state, key, None)
    if isinstance(value, str):
        return value
    for m in reversed(list(value or [])):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
        if role in ("user", "human"):
            return _content(m.get("content") if isinstance(m, dict) else m.content)
    return ""


def route_on(x: Any, field: str | None = None, key: str = "messages") -> Any:
    """A path for `add_conditional_edges`: the decision's `field` (or the label of a `ds.model(labels)`) for the
    text at `state[key]`. Works with `invoke` and `ainvoke` (a Runnable with sync and async functions)."""
    from langchain_core.runnables import RunnableLambda

    d = resolve(x)

    def text(state: Any) -> str:
        t = text_of(state, key)
        if not t.strip():
            raise ValueError("route_on: no text at state[%r]" % key)
        return t

    def route(state: Any) -> Any:
        return d.field(d.call(text(state)), field)

    async def aroute(state: Any) -> Any:
        return d.field(await d.acall(text(state)), field)

    return RunnableLambda(route, afunc=aroute, name="route_on_%s" % (field or "label"))


def guard_node(
    x: Any, field: str | None = None, block: Iterable[Any] = (True,), key: str = "messages", flag: str = "blocked"
) -> Any:
    """A node that sets `state[flag]` to True when the decision's `field` for the text at `state[key]` is one of
    `block` (False otherwise, and for empty text); route on the flag. Works with `invoke` and `ainvoke`."""
    from langchain_core.runnables import RunnableLambda

    d, blocked = resolve(x), list(block)

    def node(state: Any) -> dict[str, Any]:
        t = text_of(state, key)
        return {flag: bool(t.strip()) and d.field(d.call(t), field) in blocked}

    async def anode(state: Any) -> dict[str, Any]:
        t = text_of(state, key)
        return {flag: bool(t.strip()) and d.field(await d.acall(t), field) in blocked}

    return RunnableLambda(node, afunc=anode, name="guard_%s" % (field or "label"))
