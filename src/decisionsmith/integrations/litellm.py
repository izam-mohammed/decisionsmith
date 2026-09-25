"""LiteLLM (`uv add "decisionsmith[litellm]"`): any LiteLLM model as the teacher, a Proxy guardrail, Router tiers.

ds.harness(Ticket, teacher="litellm/bedrock/anthropic.claude-haiku-4-5", student="laya")
Guard = guardrail(ds.harness(Injection, ...), field="is_attack", block=[True])   # proxy: guardrail: guard.Guard
router.completion(model=tier(x, "hard", {True: "strong", False: "cheap"})(messages), messages=messages)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..engines.base import need
from ..engines.structured import TextEngine
from ._base import resolve


def _litellm() -> Any:
    return need("litellm", "LiteLLM is not installed", "litellm", "litellm")[0]


def _reply(litellm: Any, response: Any) -> tuple[str, dict[str, Any]]:
    u = getattr(response, "usage", None)
    usage: dict[str, Any] = {
        "input_tokens": getattr(u, "prompt_tokens", None),
        "output_tokens": getattr(u, "completion_tokens", None),
    }
    try:
        usage["cost_usd"] = float(litellm.completion_cost(completion_response=response))
    except Exception:
        usage["cost_usd"] = None
    return response.choices[0].message.content or "", usage


def teacher(model: str, **options: Any) -> TextEngine:
    """Any LiteLLM model id as a teacher; `options` go to `litellm.completion` (api_base, api_key, ...)."""

    def messages(system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        litellm = _litellm()
        return _reply(litellm, litellm.completion(model=model, messages=messages(system, user), **options))

    async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        litellm = _litellm()
        return _reply(litellm, await litellm.acompletion(model=model, messages=messages(system, user), **options))

    return TextEngine("litellm/" + model, complete, acomplete)


def last_text(messages: Any) -> str:
    """The text of the last user message (plain or content-part lists)."""
    for m in reversed(list(messages or [])):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content or "")
    return ""


def guardrail(x: Any, field: str | None = None, block: Iterable[Any] = (True,)) -> type:
    """A LiteLLM Proxy `CustomGuardrail` class: blocks a request (pre_call) or a reply (post_call) when the
    decision's `field` is one of `block`. Put `Guard = guardrail(...)` in a file next to the proxy config."""
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import GuardrailEventHooks

    d, blocked = resolve(x), list(block)

    async def check(text: str) -> None:
        if text.strip():
            value = d.field(await d.acall(text), field)
            if value in blocked:
                raise ValueError("blocked by decisionsmith: %s=%r" % (field or "label", value))

    class Guard(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> Any:
            if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call):
                await check(last_text(data.get("messages")))
            return data

        async def async_post_call_success_hook(self, data: dict, user_api_key_dict: Any, response: Any) -> Any:
            if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call):
                for choice in getattr(response, "choices", None) or []:
                    await check(getattr(getattr(choice, "message", None), "content", None) or "")
            return response

    return Guard


def tier(x: Any, field: str | None, models: dict[Any, str], default: str | None = None) -> Callable[[Any], str]:
    """`pick(messages_or_text) -> model group name` from a decision, for a LiteLLM `Router` (or any client)."""
    d = resolve(x)

    def pick(messages: Any) -> str:
        text = messages if isinstance(messages, str) else last_text(messages)
        value = d.field(d.call(text), field)
        if value in models:
            return models[value]
        if default is None:
            raise KeyError("no model for %s=%r; add it to models or pass default=" % (field or "label", value))
        return default

    return pick
