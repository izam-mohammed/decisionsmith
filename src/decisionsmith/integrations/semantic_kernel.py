"""Semantic Kernel (`uv add "decisionsmith[semantic-kernel]"`): any chat completion service as the teacher,
a plugin with a `@kernel_function`, and a function invocation filter that blocks by decision.

ds.harness(Ticket, teacher=OpenAIChatCompletion(ai_model_id="gpt-5-mini"))   # detected automatically
kernel.add_plugin(plugin(ds.harness(Ticket, ...), name="route_ticket"), "tickets")
kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, invocation_filter(ds.harness(Injection, ...), "is_attack"))
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from pydantic import BaseModel

from ..engines.structured import TextEngine
from ._base import model_name, resolve, run_sync


def teacher(service: Any) -> TextEngine:
    """A Semantic Kernel chat completion service (OpenAI, Azure OpenAI, Anthropic, Ollama, ...) as a teacher."""

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        from semantic_kernel.contents import ChatHistory

        history = ChatHistory()
        history.add_system_message(system)
        history.add_user_message(user)
        message = await service.get_chat_message_content(history, service.get_prompt_execution_settings_class()())
        u = (getattr(message, "metadata", None) or {}).get("usage")
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None),
        }
        return str(message.content if message is not None else ""), usage

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        return run_sync(lambda: acomplete(system, user))

    return TextEngine("semantic-kernel:%s" % model_name(service), complete, acomplete)


def _show(value: Any) -> str:
    return value.model_dump_json() if isinstance(value, BaseModel) else str(value)


def plugin(x: Any, name: str = "decide", description: str | None = None) -> Any:
    """A plugin object for `kernel.add_plugin(...)` with one `@kernel_function` named `name`: text in, the
    decision out (JSON, or the label for `ds.model(labels)`)."""
    from semantic_kernel.functions import kernel_function

    d = resolve(x)
    about = description or "Decide %s for a text (%s)." % (", ".join(d.fields), d.name)

    class DecisionPlugin:
        @kernel_function(name=name, description=about)
        async def decide(self, text: str) -> str:
            return _show(await d.acall(text))

    return DecisionPlugin()


def invocation_filter(
    x: Any,
    field: str | None = None,
    block: Iterable[Any] = (True,),
    functions: Iterable[str] | None = None,
    message: str = "Sorry, I can't help with that.",
) -> Callable[[Any, Callable[[Any], Awaitable[None]]], Awaitable[None]]:
    """A `FUNCTION_INVOCATION` filter: when the decision's `field` for the call's text arguments is one of `block`,
    the function is not run and `message` is its result. `functions` limits it to those names (or `plugin-name`)."""
    from semantic_kernel.functions import FunctionResult

    d, blocked, only = resolve(x), list(block), set(functions) if functions is not None else None

    async def check(context: Any, next: Callable[[Any], Awaitable[None]]) -> None:
        fn = context.function
        text = "\n".join(v for v in context.arguments.values() if isinstance(v, str))
        chosen = only is None or fn.name in only or fn.fully_qualified_name in only
        if chosen and text.strip() and d.field(await d.acall(text), field) in blocked:
            context.result = FunctionResult(function=fn.metadata, value=message, metadata={"blocked": True})
            return
        await next(context)

    return check
