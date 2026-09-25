"""Pydantic AI (`uv add "decisionsmith[pydantic-ai]"`): any Pydantic AI model as the teacher, a tool, an output
validator and an agent router.

ds.harness(Ticket, teacher=OpenAIChatModel("gpt-5-mini"))            # detected automatically
Agent(model, tools=[tool(ds.model(Ticket))])                         # the agent asks your model
agent.output_validator(output_validator(x, "is_unsafe", block=[True]))  # retry replies the model flags
await router(x, "team", {"billing": billing_agent, ...}).run(prompt)  # the decision picks the agent
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..engines.structured import TextEngine
from ._base import model_name, resolve


def _reply(result: Any) -> tuple[str, dict[str, Any]]:
    u = result.usage
    u = u() if callable(u) else u
    return str(result.output), {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}


def teacher(model: Any) -> TextEngine:
    """A Pydantic AI model (OpenAI, Anthropic, Gemini, Groq, Mistral, Bedrock, ...) as a decisionsmith teacher."""
    from pydantic_ai import Agent

    agents: dict[str, Any] = {}

    def agent(system: str) -> Any:
        if system not in agents:
            agents[system] = Agent(model, instructions=system)
        return agents[system]

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(agent(system).run_sync(user))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return _reply(await agent(system).run(user))

    return TextEngine("pydantic-ai:%s" % model_name(model), complete, acomplete)


def _plain(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def tool(x: Any, name: str = "decide", description: str | None = None) -> Any:
    """A Pydantic AI `Tool(text) -> decision` (a dict of fields, or the label for `ds.model(labels)`)."""
    from pydantic_ai import Tool

    d = resolve(x)

    async def decide(text: str) -> Any:
        return _plain(await d.acall(text))

    about = description or "Decide %s for a text (%s)." % (", ".join(d.fields), d.name)
    return Tool(decide, name=name, description=about, takes_ctx=False)


def output_validator(x: Any, field: str | None = None, block: Iterable[Any] = (True,), message: str = "") -> Any:
    """For `agent.output_validator(...)`: raises `ModelRetry` (the agent tries again) when the decision about the
    output's text has `field` in `block`. Structured outputs are judged by their JSON."""
    from pydantic_ai import ModelRetry

    d, blocked = resolve(x), list(block)

    async def validate(output: Any) -> Any:
        text = output.model_dump_json() if isinstance(output, BaseModel) else str(output)
        value = d.field(await d.acall(text), field)
        if value in blocked:
            raise ModelRetry(message or "rejected by decisionsmith: %s=%r; answer again" % (field or "label", value))
        return output

    return validate


class Router:
    """Picks the Agent for a prompt from a decision's `field`; `run` / `run_sync` run the picked agent."""

    def __init__(self, x: Any, field: str | None, agents: dict[Any, Any], default: Any = None) -> None:
        self.decider, self.field, self.agents, self.default = resolve(x), field, dict(agents), default

    def _agent(self, value: Any) -> Any:
        value = self.decider.field(value, self.field)
        if value in self.agents:
            return self.agents[value]
        if self.default is None:
            raise KeyError("no agent for %s=%r; add it to agents or pass default=" % (self.field or "label", value))
        return self.default

    def pick(self, prompt: str) -> Any:
        return self._agent(self.decider.call(prompt))

    async def apick(self, prompt: str) -> Any:
        return self._agent(await self.decider.acall(prompt))

    async def run(self, prompt: str, **options: Any) -> Any:
        return await (await self.apick(prompt)).run(prompt, **options)

    def run_sync(self, prompt: str, **options: Any) -> Any:
        return self.pick(prompt).run_sync(prompt, **options)


def router(x: Any, field: str | None, agents: dict[Any, Any], default: Any = None) -> Router:
    """A `Router` over Pydantic AI agents keyed by the values of the decision's `field`."""
    return Router(x, field, agents, default)
