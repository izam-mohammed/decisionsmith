"""CrewAI (`pip install "decisionsmith[crewai]"`): a CrewAI `LLM` as the teacher.

ds.harness(Ticket, teacher=crewai.LLM(model="gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def teacher(llm: Any) -> TextEngine:
    """A CrewAI `LLM` (any provider CrewAI supports) as a decisionsmith teacher."""

    def complete(system: str, user: str) -> str:
        return str(llm.call(_messages(system, user)))

    async def acomplete(system: str, user: str) -> str:
        return str(await llm.acall(_messages(system, user)))

    return TextEngine("crewai:%s" % model_name(llm), complete, acomplete)
