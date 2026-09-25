"""OpenAI Agents SDK (`uv add "decisionsmith[openai-agents]"`): guardrails, a function tool and a router backed
by a model or harness, and any Agents SDK model (OpenAI, `LitellmModel`, ...) as the teacher.

Agent(..., input_guardrails=[input_guardrail(ds.harness(Injection, ...), "is_attack")])
Agent(..., tools=[function_tool(ds.model(Ticket))])
await router(x, "team", {"billing": billing_agent, "technical": tech_agent}).run(text)
ds.harness(Ticket, teacher=teacher(LitellmModel("anthropic/claude-haiku-4-5")))
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..engines.structured import TextEngine
from ._base import model_name, resolve, run_sync
from .litellm import last_text


def _plain(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _text(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return last_text(value) if isinstance(value, list) else str(value)


def _guard(x: Any, field: str | None, block: Iterable[Any]) -> Any:
    from agents import GuardrailFunctionOutput

    d, blocked = resolve(x), list(block)

    async def guard(ctx: Any, agent: Any, value: Any) -> Any:
        text = _text(value)
        if not text.strip():
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)
        v = await d.acall(text)
        return GuardrailFunctionOutput(output_info=_plain(v), tripwire_triggered=d.field(v, field) in blocked)

    return guard


def input_guardrail(x: Any, field: str | None = None, block: Iterable[Any] = (True,), name: str = "ds") -> Any:
    """An `InputGuardrail`: trips (the run raises `InputGuardrailTripwireTriggered`) when the decision about the
    last user message has `field` in `block`. `output_info` is the decision."""
    from agents import InputGuardrail

    return InputGuardrail(guardrail_function=_guard(x, field, block), name=name)


def output_guardrail(x: Any, field: str | None = None, block: Iterable[Any] = (True,), name: str = "ds") -> Any:
    """An `OutputGuardrail`: trips when the decision about the agent's final output (text, or JSON for structured
    outputs) has `field` in `block`."""
    from agents import OutputGuardrail

    return OutputGuardrail(guardrail_function=_guard(x, field, block), name=name)


def function_tool(x: Any, name: str = "decide", description: str | None = None) -> Any:
    """A `FunctionTool(text) -> decision` (JSON of the fields, or the label for `ds.model(labels)`)."""
    from agents import FunctionTool

    d = resolve(x)
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def invoke(ctx: Any, args: str) -> str:
        return json.dumps(_plain(await d.acall(json.loads(args)["text"])))

    about = description or "Decide %s for a text (%s)." % (", ".join(d.fields), d.name)
    return FunctionTool(name, about, {**schema, "additionalProperties": False}, invoke)


class Router:
    """Hands each input to the Agent picked by a decision's `field` (no triage LLM call); `run` uses `Runner.run`."""

    def __init__(self, x: Any, field: str | None, agents: dict[Any, Any], default: Any = None) -> None:
        self.decider, self.field, self.agents, self.default = resolve(x), field, dict(agents), default

    async def pick(self, input: Any) -> Any:
        """The Agent for this input (a string or a list of input items)."""
        value = await self.decider.acall(_text(input))
        value = self.decider.field(value, self.field)
        if value in self.agents:
            return self.agents[value]
        if self.default is None:
            raise KeyError("no agent for %s=%r; add it to agents or pass default=" % (self.field or "label", value))
        return self.default

    async def run(self, input: Any, **options: Any) -> Any:
        from agents import Runner

        return await Runner.run(await self.pick(input), input, **options)


def router(x: Any, field: str | None, agents: dict[Any, Any], default: Any = None) -> Router:
    """A `Router` over Agents SDK agents keyed by the values of the decision's `field`."""
    return Router(x, field, agents, default)


def teacher(model: Any) -> TextEngine:
    """An Agents SDK `Model` (or model name) as a teacher: each call is one `Runner.run` of a tool-less agent."""
    from agents import Agent, Runner

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        result = await Runner.run(Agent(name="decisionsmith-teacher", instructions=system, model=model), user)
        u = result.context_wrapper.usage
        return str(result.final_output), {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}

    name = model if isinstance(model, str) else model_name(model)
    return TextEngine("openai-agents:%s" % name, lambda s, u: run_sync(lambda: acomplete(s, u)), acomplete)
