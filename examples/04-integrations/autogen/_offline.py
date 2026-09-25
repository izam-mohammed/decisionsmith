"""Offline (DS_OFFLINE=1) the agents run on AutoGen's ReplayChatCompletionClient instead of calling OpenAI."""

import os

from autogen_core import FunctionCall
from autogen_core.models import CreateResult, ModelFamily, ModelInfo, RequestUsage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.replay import ReplayChatCompletionClient


def client(replies: list[str], tool: str | None = None, text: str = ""):
    """gpt-5-mini, or offline a replay client that says `replies` (after calling `tool` with `text`, if given)."""
    if not os.environ.get("DS_OFFLINE"):
        return OpenAIChatCompletionClient(model="gpt-5-mini")
    calls: list = []
    if tool:
        call = FunctionCall(id="c1", name=tool, arguments='{"text": "%s"}' % text)
        calls.append(
            CreateResult(finish_reason="function_calls", content=[call], usage=RequestUsage(0, 0), cached=False)
        )
    info = ModelInfo(
        vision=False, function_calling=True, json_output=False, family=ModelFamily.UNKNOWN, structured_output=False
    )
    return ReplayChatCompletionClient([*calls, *replies], model_info=info)
