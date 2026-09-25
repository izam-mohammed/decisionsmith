import asyncio
import json

import pytest

pytest.importorskip("semantic_kernel")

import decisionsmith as ds
from decisionsmith.engines import from_string
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth
from tests.integrations.openai_mock import http_lib, openai_client

Q = compile_schema(Ticket).questions()


def test_openai_chat_completion_is_a_teacher():
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    client, seen = openai_client()
    e = from_string(OpenAIChatCompletion(ai_model_id="gpt-4o-mini", async_client=client))
    assert e.name == "semantic-kernel:gpt-4o-mini"
    out = e.ask("you charged me twice", Q)
    assert out["answers"]["team"]["choice"] == "billing" and out["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert asyncio.run(e.aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    assert '"role":"system"' in seen[0].content.decode().replace(" ", "")


def harness():
    return ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), log=None)


def spam():
    return ds.model(["spam", "ok"], FakeEngine(lambda t: {"label": "spam" if "win" in t else "ok"}))


def kernel(*plugins):
    from semantic_kernel import Kernel

    k = Kernel()
    for name, p in plugins:
        k.add_plugin(p, name)
    return k


def test_plugin_is_a_kernel_function():
    from decisionsmith.integrations.semantic_kernel import plugin

    k = kernel(("tickets", plugin(harness(), name="route_ticket")), ("spam", plugin(spam(), description="Spam?")))
    fn = k.get_function("tickets", "route_ticket")
    assert "team, wants_refund" in fn.description and [p.name for p in fn.parameters] == ["text"]
    out = asyncio.run(k.invoke(plugin_name="tickets", function_name="route_ticket", text="refund my card charge"))
    assert json.loads(str(out)) == {"team": "billing", "wants_refund": True}
    assert k.get_function("spam", "decide").description == "Spam?"
    assert str(asyncio.run(k.invoke(plugin_name="spam", function_name="decide", text="you win"))) == "spam"


def echo():
    from semantic_kernel.functions import kernel_function

    class Echo:
        def __init__(self) -> None:
            self.calls: list[str] = []

        @kernel_function(name="reply")
        def reply(self, text: str) -> str:
            self.calls.append(text)
            return "sent: " + text

    return Echo()


def test_filter_blocks_by_decision():
    from semantic_kernel.filters import FilterTypes

    from decisionsmith.integrations.semantic_kernel import invocation_filter

    e = echo()
    k = kernel(("mail", e))
    k.add_filter(FilterTypes.FUNCTION_INVOCATION, invocation_filter(spam(), block=["spam"], message="nope"))

    def run(text):
        return asyncio.run(k.invoke(plugin_name="mail", function_name="reply", text=text))

    blocked = run("you win a prize")
    assert str(blocked) == "nope" and blocked.metadata == {"blocked": True}
    assert str(run("see you monday")) == "sent: see you monday" and e.calls == ["see you monday"]
    assert str(asyncio.run(k.invoke(plugin_name="mail", function_name="reply", text=" "))) == "sent:  "


def test_filter_limited_to_functions():
    from semantic_kernel.filters import FilterTypes

    from decisionsmith.integrations.semantic_kernel import invocation_filter

    k = kernel(("mail", echo()))
    k.add_filter(FilterTypes.FUNCTION_INVOCATION, invocation_filter(harness(), "team", ["billing"], ["other"]))
    assert str(asyncio.run(k.invoke(plugin_name="mail", function_name="reply", text="my invoice is wrong"))).startswith(
        "sent"
    )
    k2 = kernel(("mail", echo()))
    k2.add_filter(FilterTypes.FUNCTION_INVOCATION, invocation_filter(harness(), "team", ["billing"], ["mail-reply"]))
    assert str(
        asyncio.run(k2.invoke(plugin_name="mail", function_name="reply", text="my invoice is wrong"))
    ).startswith("Sorry")


def test_auto_function_calling_uses_the_plugin():
    """A real chat completion with auto function calling: the (mocked) LLM calls the plugin, SK runs it."""
    import openai
    from semantic_kernel.connectors.ai import FunctionChoiceBehavior
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
    from semantic_kernel.contents import ChatHistory

    from decisionsmith.integrations.semantic_kernel import plugin

    lib, bodies = http_lib(), []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        tool = [m for m in body["messages"] if m["role"] == "tool"]
        if tool:
            message = {"role": "assistant", "content": "routed: " + tool[0]["content"]}
        else:
            call = {"name": "tickets-route", "arguments": json.dumps({"text": "sync is broken"})}
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": call}],
            }
        choice = {"index": 0, "finish_reason": "stop", "message": message}
        return lib.Response(
            200, json={"id": "1", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice]}
        )

    client = openai.AsyncOpenAI(api_key="sk-test", http_client=lib.AsyncClient(transport=lib.MockTransport(handler)))
    service = OpenAIChatCompletion(ai_model_id="gpt-4o-mini", async_client=client)
    k = kernel(("tickets", plugin(harness(), name="route")))
    k.add_service(service)
    history = ChatHistory()
    history.add_user_message("route my ticket")
    settings = OpenAIChatPromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto())
    reply = asyncio.run(service.get_chat_message_content(history, settings, kernel=k))
    assert str(reply) == 'routed: {"team":"technical","wants_refund":false}'
    assert bodies[0]["tools"][0]["function"]["name"] == "tickets-route"
