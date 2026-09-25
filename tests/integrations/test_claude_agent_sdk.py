import asyncio

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from claude_agent_sdk.types import HookContext, PostToolUseHookInput, PreToolUseHookInput

import decisionsmith as ds
from decisionsmith.integrations.claude_agent_sdk import flatten, hook
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth

CTX = HookContext(signal=None)


def harness():
    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def pre(tool_input, name="Bash"):
    base = {"session_id": "s", "transcript_path": "/tmp/t.jsonl", "cwd": "/tmp", "hook_event_name": "PreToolUse"}
    return PreToolUseHookInput(**base, tool_name=name, tool_input=tool_input, tool_use_id="tu1")


def post(response):
    base = {"session_id": "s", "transcript_path": "/tmp/t.jsonl", "cwd": "/tmp", "hook_event_name": "PostToolUse"}
    return PostToolUseHookInput(**base, tool_name="Bash", tool_input={}, tool_response=response, tool_use_id="tu1")


def run(matcher, data):
    return asyncio.run(matcher.hooks[0](data, data.get("tool_use_id"), CTX))


def test_pre_tool_use_denies_blocked_calls():
    m = hook(harness(), "team", block=["billing"], matcher="Bash")
    assert isinstance(m, HookMatcher) and m.matcher == "Bash"
    options = ClaudeAgentOptions(hooks={"PreToolUse": [m]})
    assert options.hooks["PreToolUse"][0] is m
    out = run(m, pre({"command": "echo 'refund my card charge'", "timeout": 5}))
    specific = out["hookSpecificOutput"]
    assert specific["hookEventName"] == "PreToolUse" and specific["permissionDecision"] == "deny"
    assert "team='billing'" in specific["permissionDecisionReason"]
    assert run(m, pre({"command": "cat sync is broken.log"})) == {}
    assert run(m, pre({"timeout": 5})) == {}


def test_post_tool_use_blocks_outputs_and_custom_text():
    m = hook(harness(), "team", block=["technical"], event="PostToolUse")
    out = run(m, post({"stdout": "error: the page shows an error", "stderr": "", "interrupted": False}))
    assert out["decision"] == "block" and "team='technical'" in out["reason"]
    assert out["hookSpecificOutput"] == {"hookEventName": "PostToolUse"}
    assert run(m, post([{"type": "text", "text": "how much is the plan"}])) == {}
    labels = ds.model(["safe", "risky"], FakeEngine(lambda t: {"label": "risky" if "rm" in t else "safe"}))
    only_cmd = hook(labels, block=["risky"], text=lambda data: data["tool_input"].get("command", ""))
    assert run(only_cmd, pre({"command": "rm -rf build", "description": "clean"}))["hookSpecificOutput"]
    assert run(only_cmd, pre({"command": "ls", "description": "rm nothing"})) == {}


def test_bad_event_and_flatten():
    with pytest.raises(ValueError, match="event must be"):
        hook(harness(), "team", event="Stop")
    assert flatten({"a": ["x", {"b": "y"}, 3], "c": None}) == "x y" and flatten(7) == ""
