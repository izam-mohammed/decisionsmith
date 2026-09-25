"""DSPy (`pip install "decisionsmith[dspy]"`): a `dspy.LM` as the teacher, a `dspy.Module` backed by a decision,
and a metric for `dspy.Evaluate`.

ds.harness(Ticket, teacher=dspy.LM("openai/gpt-5-mini"))            # detected automatically
program = DecisionModule(h)                                         # program(text=...) -> Prediction(team=..., ...)
dspy.Evaluate(devset=devset, metric=metric("team"))(program)        # or any DSPy program with a `team` output
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..engines.structured import TextEngine
from ._base import resolve

_CLASSES: dict[str, type] = {}


def _text(outputs: Any) -> str:
    first = outputs[0] if outputs else ""
    return str(first.get("text", "") if isinstance(first, dict) else first)


def teacher(lm: Any) -> TextEngine:
    """A `dspy.LM` (any provider DSPy supports) as a decisionsmith teacher. System and user text go in one prompt."""

    def complete(system: str, user: str) -> str:
        return _text(lm("%s\n\n%s" % (system, user)))

    async def acomplete(system: str, user: str) -> str:
        return _text(await lm.acall("%s\n\n%s" % (system, user)))

    return TextEngine("dspy:%s" % getattr(lm, "model", type(lm).__name__), complete, acomplete)


def _module() -> type:
    import dspy

    class DecisionModule(dspy.Module):
        """A `dspy.Module`: `program(text=...)` returns a `dspy.Prediction` with one output per field of the
        decision (`label` for a `ds.model(labels)`). `input` names the input field."""

        def __init__(self, x: Any, input: str = "text") -> None:
            super().__init__()
            self.decider, self.input = resolve(x), input

        def _prediction(self, value: Any) -> Any:
            if isinstance(value, BaseModel):
                return dspy.Prediction(**value.model_dump(mode="json"))
            return dspy.Prediction(label=value)

        def forward(self, **inputs: Any) -> Any:
            return self._prediction(self.decider.call(str(inputs[self.input])))

        async def aforward(self, **inputs: Any) -> Any:
            return self._prediction(await self.decider.acall(str(inputs[self.input])))

    return DecisionModule


def __getattr__(name: str) -> Any:
    if name == "DecisionModule":
        return _CLASSES.setdefault(name, _module())
    raise AttributeError(name)


def _same(a: Any, b: Any) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def metric(*fields: str) -> Any:
    """A DSPy metric `(example, prediction, trace=None)`: the share of `fields` (default: the example's labels) where
    the prediction matches the example, case-insensitively. While optimising (`trace` set) it is True only when all
    of them match."""

    def score(example: Any, prediction: Any, trace: Any = None) -> float | bool:
        names = list(fields) or list(example.labels().keys())
        hits = sum(_same(example[n], prediction.get(n)) for n in names)
        value = hits / len(names) if names else 0.0
        return value == 1.0 if trace is not None else value

    return score
