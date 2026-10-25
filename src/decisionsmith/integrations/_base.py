"""Shared by every integration: accept a `ds.model(...)` or a `ds.harness(...)`, and wrap LLM calls as teachers."""

from __future__ import annotations

import asyncio
import enum
import threading
import uuid
import weakref
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


def plain(value: Any) -> Any:
    """A decision value as a plain Python value (an enum member becomes its value)."""
    return value.value if isinstance(value, enum.Enum) else value


def value_type(x: Any, field: str) -> type:
    """The Python type of one field's values: `bool`, `int`, `float` or `str` (mixed option types become `str`)."""
    f = x.schema.fields[field]
    if f.kind == "bool":
        return bool
    kinds = {type(plain(v)) for v in f.values}
    return kinds.pop() if len(kinds) == 1 and kinds <= {int, float, str} else str


def pick_fields(x: Any, fields: Any = None) -> list[str]:
    """The decision fields to write: all of them by default, or the ones named (a name or a list)."""
    names = resolve(x).fields
    if fields is None:
        return names
    chosen = [fields] if isinstance(fields, str) else list(fields)
    unknown = [f for f in chosen if f not in names]
    if unknown or not chosen:
        raise ValueError("unknown fields %s; this model decides %s" % (unknown, names))
    return chosen


def columns(fields: list[str], prefix: str = "") -> list[str]:
    """The column names for `fields`: `<field>`, `<field>_confidence` and `<field>_source` for each."""
    return [prefix + f + end for f in fields for end in ("", "_confidence", "_source")]


_LOCKS: weakref.WeakKeyDictionary[Any, threading.Lock] = weakref.WeakKeyDictionary()
_LOCKS_LOCK = threading.Lock()


def _lock(x: Any) -> threading.Lock:
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(x, threading.Lock())


def _decide(x: Any, texts: list[str]) -> list[tuple[Any, dict[str, str], dict[str, float]]]:
    from ..core import Harness
    from ..engines import response
    from ..schema import confidence
    from ..training.adapt import calibrated

    if isinstance(x, Harness):
        return [(r.value, r.source, r.confidence) for r in x._results(texts)]
    questions, out = x.schema.questions(), []
    for raw in x.ask_many(texts, questions):
        answers, _ = response(x, raw, questions)
        dists = {n: calibrated(x.calibration, n, d) for n, d in x.schema.distributions(answers).items()}
        out.append(
            (x.schema.build(dists), dict.fromkeys(dists, "student"), {n: confidence(d) for n, d in dists.items()})
        )
    return out


def decide_columns(
    x: Any, texts: list[Any], fields: Any = None, *, prefix: str = "", batch_size: int = 256
) -> dict[str, list[Any]]:
    """Decide `texts` in batches into columns (see `columns`); a missing or empty text gets `None` in every column.
    `source` is `"student"` or `"teacher"` (a `ds.model` is the student). Threads (Dask, Polars) take turns on one
    model or harness, since an in-process model is not safe to run from several threads at once."""
    names = pick_fields(x, fields)
    out: dict[str, list[Any]] = {c: [None] * len(texts) for c in columns(names, prefix)}
    todo = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    for start in range(0, len(todo), max(1, batch_size)):
        chunk = todo[start : start + max(1, batch_size)]
        with _lock(x):
            got = _decide(x, [texts[i] for i in chunk])
        for i, (value, source, conf) in zip(chunk, got):
            for f in names:
                out[prefix + f][i] = plain(getattr(value, f))
                out[prefix + f + "_confidence"][i] = conf[f]
                out[prefix + f + "_source"][i] = source[f]
    return out


_BUILT: dict[str, Any] = {}


class Lazy:
    """A model or harness, or a function with no arguments that builds one the first time it is needed in each
    process (Spark and Dask workers, Ray workers and actors). What it built is never pickled; copies of the same
    `Lazy` in one process share one build."""

    def __init__(self, x: Any) -> None:
        from ..core import Harness
        from ..predictor import Model

        built = isinstance(x, (Model, Harness))
        if not built and not callable(x):
            raise TypeError("expected ds.model(...), ds.harness(...) or a function that returns one, got %r" % (x,))
        self.make: Callable[[], Any] | None = None if built else x
        self.x: Any = x if built else None
        self.token = uuid.uuid4().hex

    def __dask_tokenize__(self) -> str:
        return self.token

    def get(self) -> Any:
        if self.x is None:
            assert self.make is not None
            if self.token not in _BUILT:
                built = self.make()
                resolve(built)
                _BUILT[self.token] = built
            self.x = _BUILT[self.token]
        return self.x

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "x": None if self.make is not None else self.x}


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
