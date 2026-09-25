"""Offline (DS_OFFLINE=1) the agents run on the SDK's ScriptedModel instead of calling OpenAI."""

import os

from agents import set_tracing_disabled
from agents.testing import ScriptedModel, assistant_message

OFFLINE = bool(os.environ.get("DS_OFFLINE"))
if OFFLINE:
    set_tracing_disabled(True)


def llm(reply: str = "(offline reply)", tool: str | None = None):
    """The agent's model: gpt-5-mini, or offline a scripted one (that calls `tool` first, if given)."""
    if not OFFLINE:
        return "gpt-5-mini"
    from agents.testing import function_call

    first = [[function_call(tool, {"text": "I was charged twice"}, call_id="c1")]] if tool else []
    return ScriptedModel([*first, [assistant_message(reply)]])
