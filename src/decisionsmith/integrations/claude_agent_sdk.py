"""Claude Agent SDK (`uv add "decisionsmith[claude-agent-sdk]"`): `PreToolUse` / `PostToolUse` hooks that let a
decision deny a tool call or flag its output.

bash = hook(ds.harness(Risky, ...), field="is_destructive", block=[True], matcher="Bash")
options = ClaudeAgentOptions(hooks={"PreToolUse": [bash]})
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ._base import resolve

EVENTS = ("PreToolUse", "PostToolUse")


def flatten(value: Any) -> str:
    """The text in a tool input or output: strings, and the strings nested in dicts and lists, joined."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return " ".join(t for t in (flatten(v) for v in value) if t)
    return ""


def hook(
    x: Any,
    field: str | None = None,
    block: Iterable[Any] = (True,),
    event: str = "PreToolUse",
    matcher: str | None = None,
    text: Callable[[dict[str, Any]], str] | None = None,
) -> Any:
    """A `HookMatcher` for `ClaudeAgentOptions(hooks={event: [...]})`. `PreToolUse`: the tool call is denied
    (`permissionDecision: "deny"`) when the decision's `field` for the tool input is one of `block`.
    `PostToolUse`: the output is blocked (`decision: "block"`, the reason goes to Claude). Otherwise the hook
    returns `{}` and the normal permission flow applies. `text(hook_input)` picks the text to decide (default:
    the strings in `tool_input`, or in `tool_response` after the call); `matcher` is a tool name pattern."""
    from claude_agent_sdk import HookMatcher

    if event not in EVENTS:
        raise ValueError("event must be one of %s, got %r" % (EVENTS, event))
    d, blocked = resolve(x), list(block)
    part = "tool_input" if event == "PreToolUse" else "tool_response"
    pick = text or (lambda data: flatten(data.get(part)))

    async def callback(data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        t = pick(data)
        value = d.field(await d.acall(t), field) if t.strip() else None
        if not t.strip() or value not in blocked:
            return {}
        reason = "blocked by decisionsmith: %s=%r" % (field or "label", value)
        if event == "PreToolUse":
            decision = {"permissionDecision": "deny", "permissionDecisionReason": reason}
            return {"hookSpecificOutput": {"hookEventName": event, **decision}}
        return {"decision": "block", "reason": reason, "hookSpecificOutput": {"hookEventName": event}}

    return HookMatcher(matcher=matcher, hooks=[callback])
