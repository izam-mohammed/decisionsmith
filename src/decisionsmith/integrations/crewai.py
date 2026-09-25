"""CrewAI (`uv add "decisionsmith[crewai]"`): a CrewAI `LLM` as the teacher, a decision as a `BaseTool`, and a
task guardrail.

ds.harness(Ticket, teacher=crewai.LLM(model="gpt-5-mini"))            # detected automatically
Agent(role=..., tools=[tool(ds.harness(Ticket, ...), name="route_ticket")])
Task(..., guardrail=guardrail(ds.harness(Unsafe, ...), "is_unsafe", block=[True]))  # the agent retries
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..engines.structured import TextEngine
from ._base import model_name, resolve


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def teacher(llm: Any) -> TextEngine:
    """A CrewAI `LLM` (any provider CrewAI supports) as a decisionsmith teacher."""

    def complete(system: str, user: str) -> str:
        return str(llm.call(_messages(system, user)))

    async def acomplete(system: str, user: str) -> str:
        return str(await llm.acall(_messages(system, user)))

    return TextEngine("crewai:%s" % model_name(llm), complete, acomplete)


class _Text(BaseModel):
    text: str


def _show(value: Any) -> str:
    return value.model_dump_json() if isinstance(value, BaseModel) else str(value)


def tool(x: Any, name: str = "decide", description: str | None = None, field: str | None = None) -> Any:
    """A CrewAI `BaseTool` taking `text` and returning the decision as JSON (or the label, or one `field`)."""
    from crewai.tools import BaseTool

    d = resolve(x)

    class DecisionTool(BaseTool):
        def _run(self, text: str) -> str:
            return _show(d.field(d.call(text), field))

        async def _arun(self, text: str) -> str:
            return _show(d.field(await d.acall(text), field))

    about = description or "Decide %s for a text (%s)." % (", ".join(d.fields), d.name)
    return DecisionTool(name=name, description=about, args_schema=_Text)


def guardrail(x: Any, field: str | None = None, block: Iterable[Any] = (True,), message: str = "") -> Any:
    """For `Task(guardrail=...)` (or `Agent(guardrail=...)`): `(False, feedback)` when the decision about the
    output's raw text has `field` in `block`, so CrewAI asks the agent again; otherwise `(True, output)`."""
    d, blocked = resolve(x), list(block)

    def check(output):  # unannotated: CrewAI checks the return annotation at runtime and rejects a string one
        value = d.field(d.call(str(output.raw)), field)
        if value in blocked:
            return False, message or "rejected by decisionsmith: %s=%r; answer again" % (field or "label", value)
        return True, output

    return check
