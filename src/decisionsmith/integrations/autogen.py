"""AutoGen (`pip install "decisionsmith[autogen]"`): any AutoGen model client as the teacher, a `FunctionTool` and a
termination condition that stops a team when a message is blocked.

ds.harness(Ticket, teacher=OpenAIChatCompletionClient(model="gpt-5-mini"))  # detected automatically
AssistantAgent("support", model_client, tools=[tool(ds.harness(Ticket, ...), name="route_ticket")])
RoundRobinGroupChat(agents, termination_condition=stop_on(ds.harness(Safety, ...), "is_unsafe"))
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name, resolve, run_sync


def teacher(client: Any) -> TextEngine:
    """An AutoGen (0.4+) model client (OpenAI, Azure OpenAI, Anthropic, Ollama, ...) as a decisionsmith teacher."""

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        from autogen_core.models import SystemMessage, UserMessage

        result = await client.create([SystemMessage(content=system), UserMessage(content=user, source="user")])
        u = result.usage
        return str(result.content), {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return run_sync(lambda: acomplete(system, user))

    info = getattr(client, "_raw_config", None) or {}
    return TextEngine("autogen:%s" % info.get("model", model_name(client)), complete, acomplete)


def tool(x: Any, name: str = "decide", description: str | None = None) -> Any:
    """An `autogen_core.tools.FunctionTool` taking `text` and returning the decision (the model sees it as JSON,
    or the label for `ds.model(labels)`)."""
    from autogen_core.tools import FunctionTool

    d = resolve(x)

    async def decide(text: str) -> Any:
        return await d.acall(text)

    about = description or "Decide %s for a text; returns the answer." % ", ".join(d.fields)
    return FunctionTool(decide, description=about, name=name)


def stop_on(
    x: Any, field: str | None = None, block: Iterable[Any] = (True,), sources: Sequence[str] | None = None
) -> Any:
    """A team `TerminationCondition`: stops the run when a chat message (from `sources`, or any) has the decision's
    `field` in `block`. The `StopMessage` says which message and value; combine with `|` / `&` as usual."""
    from autogen_agentchat.base import TerminatedException, TerminationCondition
    from autogen_agentchat.messages import BaseChatMessage, StopMessage

    d, blocked = resolve(x), list(block)

    class DecisionTermination(TerminationCondition):
        def __init__(self) -> None:
            self._terminated = False

        @property
        def terminated(self) -> bool:
            return self._terminated

        async def __call__(self, messages: Sequence[Any]) -> Any:
            if self._terminated:
                raise TerminatedException("Termination condition has already been reached")
            for m in messages:
                if not isinstance(m, BaseChatMessage) or (sources is not None and m.source not in sources):
                    continue
                value = d.field(await d.acall(m.to_text()), field)
                if value in blocked:
                    self._terminated = True
                    about = "blocked by decisionsmith: %s=%r in a message from %s" % (field or "label", value, m.source)
                    return StopMessage(content=about, source="DecisionTermination")
            return None

        async def reset(self) -> None:
            self._terminated = False

    return DecisionTermination()
