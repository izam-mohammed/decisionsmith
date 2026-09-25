"""Portkey guardrail webhook (no extra needed): a model or harness decides whether a request or reply passes.

check = webhook(ds.harness(Injection, ...), field="is_attack", block=[True])
@app.post("/guard")                       # FastAPI, Flask, any web framework; Portkey posts JSON here
async def guard(body: dict): return await check.acall(body)   # or check(body) in sync code
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._base import resolve


def text_of(body: dict[str, Any]) -> str:
    """The text Portkey asks about: the request text before the call, the reply text after it."""
    part = "response" if body.get("eventType") == "afterRequestHook" else "request"
    return str((body.get(part) or {}).get("text") or "")


class Webhook:
    """Takes Portkey's webhook request JSON and returns its verdict JSON: `{"verdict": false}` when the decision's
    `field` is one of `block`. Call it (`check(body)`) or await `check.acall(body)`."""

    def __init__(self, x: Any, field: str | None = None, block: Iterable[Any] = (True,)) -> None:
        self.decider, self.field, self.block = resolve(x), field, list(block)

    def _verdict(self, value: Any) -> dict[str, Any]:
        return {"verdict": self.decider.field(value, self.field) not in self.block}

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        text = text_of(body)
        return self._verdict(self.decider.call(text)) if text.strip() else {"verdict": True}

    async def acall(self, body: dict[str, Any]) -> dict[str, Any]:
        text = text_of(body)
        return self._verdict(await self.decider.acall(text)) if text.strip() else {"verdict": True}


def webhook(x: Any, field: str | None = None, block: Iterable[Any] = (True,)) -> Webhook:
    """A Portkey webhook guardrail handler backed by a `ds.model` or `ds.harness`."""
    return Webhook(x, field, block)
