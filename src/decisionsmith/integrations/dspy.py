"""DSPy (`pip install "decisionsmith[dspy]"`): a `dspy.LM` as the teacher.

ds.harness(Ticket, teacher=dspy.LM("openai/gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine


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
