"""Test helpers: a deterministic engine that needs no network and no weights."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from typing import Any

from .engines.base import EngineError
from .schema import option_keys as _options

Truth = Callable[[str], Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]


def _index(q: dict[str, Any], value: Any) -> int:
    options = _options(q)
    if q["type"] == "noul":
        return 1 if str(value).strip().lower() in ("true", "yes", "1") else 0
    if q["type"] == "score":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        for i, c in enumerate(q["criteria"]):
            if str(c) == str(value) or str(c).startswith("%s: " % value):
                return i
        raise ValueError("FakeEngine truth %r is not a level of %r" % (value, q["criteria"]))
    value = getattr(value, "value", value)
    return options.index(str(value))


class FakeEngine:
    """Deterministic engine for tests, docs and demos.

    `truth` maps a text to the right answer per question (a dict of texts, or a function). With it,
    the engine is right `accuracy` of the time; without it, answers are a deterministic chance baseline.
    `confidence` is the probability put on the chosen option (a float or `f(text, question_id)`).
    """

    def __init__(
        self,
        truth: Truth | None = None,
        *,
        accuracy: float = 1.0,
        confidence: float | Callable[[str, str], float] = 0.9,
        error: str | BaseException | None = None,
        latency: float = 0.0,
        name: str = "fake",
        seed: int = 0,
    ) -> None:
        self.truth, self.accuracy, self.confidence = truth, accuracy, confidence
        self.error, self.latency, self.name, self.seed = error, latency, name, seed
        self.calls: list[tuple[str, list[str]]] = []

    def _hash(self, *parts: object) -> int:
        blob = "|".join(str(p) for p in (self.seed, *parts)).encode()
        return int(hashlib.sha256(blob).hexdigest()[:12], 16)

    def ask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.calls.append((text, list(questions)))
        if self.error is not None:
            if isinstance(self.error, BaseException):
                raise self.error
            raise EngineError(self.name, self.error)
        if self.latency:
            time.sleep(self.latency)
        truth = self.truth(text) if callable(self.truth) else (self.truth or {}).get(text, {})
        answers = {}
        for qid, q in questions.items():
            options = _options(q)
            k = len(options)
            h = self._hash(text, qid)
            if qid in truth:
                right = _index(q, truth[qid])
                correct = (h % 10_000) / 10_000 < self.accuracy or k == 1
                idx = right if correct else (right + 1 + h // 10_000 % (k - 1)) % k
            else:
                idx = h % k
            conf = self.confidence(text, qid) if callable(self.confidence) else self.confidence
            conf = 1.0 if k == 1 else min(1.0, max(1.0 / k, conf))
            probs = [conf if i == idx else (1.0 - conf) / (k - 1) for i in range(k)]
            answers[qid] = _answer(q, options, probs, idx)
        return {
            "model": self.name,
            "answers": answers,
            "usage": {"input_tokens": len(text.split()), "output_tokens": 0},
        }


def _answer(q: dict[str, Any], options: list[str], probs: list[float], idx: int) -> dict[str, Any]:
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs[1], "confidence": max(probs)}
    out: dict[str, Any] = {
        "type": q["type"],
        "probabilities": dict(zip(options, probs)),
        "confidence": max(probs),
    }
    if q["type"] == "choice":
        out["choice"] = options[idx]
    else:
        out["score"] = float(sum(i * p for i, p in enumerate(probs)))
    return out
