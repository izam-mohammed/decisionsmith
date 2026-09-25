"""The one place decisionsmith touches laya internals: turning a question into laya's internal form.

laya's `Agent` validates and normalises questions in private static methods; training builds sequences with the
same code so a fine-tuned checkpoint sees exactly what `laya.load(...).predict` sends. If a laya release renames
them, this fails with one clear message instead of an AttributeError deep in training.
"""

from __future__ import annotations

from typing import Any

SUPPORTED = "laya>=0.3.20,<0.4"


def internal_question(qid: str, question: dict[str, Any]) -> dict[str, Any]:
    from laya.agent import Agent

    check = getattr(Agent, "_check_question", None)
    to_internal = getattr(Agent, "_to_internal", None)
    if not callable(check) or not callable(to_internal):
        import laya

        raise RuntimeError(
            "this laya (%s) changed the question API decisionsmith trains with; install %s: "
            'uv add "decisionsmith[laya]"' % (getattr(laya, "__version__", "?"), SUPPORTED)
        )
    check(qid, question)
    return dict(to_internal(question))
