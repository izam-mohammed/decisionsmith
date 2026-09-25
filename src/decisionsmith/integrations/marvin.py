"""Marvin `classify()` drop-in (no extra needed), answered by a `ds.model` (Laya by default) instead of an LLM.

from decisionsmith.integrations.marvin import classify     # instead of: from marvin import classify
classify("my card was charged twice", ["billing", "technical", "sales"])            # -> "billing"
classify(text, Team, model=ds.model(["billing", "technical", "sales"], "laya:./runs/v1"))
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Any

from ._base import resolve

_MODELS: dict[tuple[Any, ...], Any] = {}


def _key(label: Any) -> str:
    return str(label.value) if isinstance(label, enum.Enum) else str(label)


def _decider(options: list[Any], instructions: str | None, model: Any, unsupported: dict[str, Any]) -> Any:
    import decisionsmith as ds

    used = [k for k, v in unsupported.items() if v is not None]
    if used:
        raise TypeError("decisionsmith's classify does not support %s" % ", ".join(used))
    if model is None:
        key = (tuple(_key(o) for o in options), instructions)
        if key not in _MODELS:
            _MODELS[key] = ds.model(list(key[0]), question=instructions)
        model = _MODELS[key]
    return resolve(model)


def _pick(options: list[Any], label: Any) -> Any:
    for o in options:
        if _key(o) == str(label):
            return o
    raise ValueError("the model answered %r, which is not one of the labels %s" % (label, [_key(o) for o in options]))


def classify(
    data: Any,
    labels: Sequence[Any] | type[enum.Enum],
    multi_label: bool = False,
    *,
    instructions: str | None = None,
    agent: Any = None,
    thread: Any = None,
    context: dict[str, Any] | None = None,
    handlers: list[Any] | None = None,
    prompt: str | None = None,
    model: Any = None,
) -> Any:
    """Marvin's `classify(data, labels)`: returns one of `labels` (an item, or a member of an Enum class).
    `model=` is a `ds.model(labels)` or a harness of one; the default is `ds.model(labels)` (base Laya).
    `multi_label`, `agent`, `thread`, `context`, `handlers` and `prompt` are not supported."""
    options = list(labels)
    extra = {"multi_label": multi_label or None, "agent": agent, "thread": thread, "context": context}
    d = _decider(options, instructions, model, {**extra, "handlers": handlers, "prompt": prompt})
    return _pick(options, d.call(str(data)))


async def classify_async(
    data: Any,
    labels: Sequence[Any] | type[enum.Enum],
    multi_label: bool = False,
    *,
    instructions: str | None = None,
    agent: Any = None,
    thread: Any = None,
    context: dict[str, Any] | None = None,
    handlers: list[Any] | None = None,
    prompt: str | None = None,
    model: Any = None,
) -> Any:
    """`classify` for async code (Marvin's `classify_async`)."""
    options = list(labels)
    extra = {"multi_label": multi_label or None, "agent": agent, "thread": thread, "context": context}
    d = _decider(options, instructions, model, {**extra, "handlers": handlers, "prompt": prompt})
    return _pick(options, await d.acall(str(data)))
