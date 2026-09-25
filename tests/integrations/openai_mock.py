"""An OpenAI-compatible chat endpoint for framework tests: frameworks use their real OpenAI clients, respx answers."""

import json

import httpx

GOOD = '{"q0": "billing", "q1": true}'


def reply(content=GOOD, usage=(11, 3), http=httpx, seen=None):
    """A handler for respx (httpx) or for `http.MockTransport` (httpx2, used by openai 3.x)."""

    def handler(request):
        if seen is not None:
            seen.append(request)
        body = json.loads(request.content or b"{}")
        return http.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "m"),
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
                ],
                "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
            },
        )

    return handler


def http_lib():
    """The HTTP library the installed `openai` SDK uses: httpx2 from openai 3, httpx before."""
    import openai

    if int(openai.__version__.split(".")[0]) >= 3:
        import httpx2

        return httpx2
    return httpx


def http_client(is_async=True, **reply_kwargs):
    """An HTTP client for the installed `openai` SDK whose requests `reply()` answers. Returns (client, seen)."""
    lib, seen = http_lib(), []
    transport = lib.MockTransport(reply(http=lib, seen=seen, **reply_kwargs))
    return (lib.AsyncClient if is_async else lib.Client)(transport=transport), seen


def openai_client(is_async=True, **reply_kwargs):
    """An `openai` client whose requests `reply()` answers. Returns (client, seen)."""
    import openai

    http, seen = http_client(is_async, **reply_kwargs)
    return (openai.AsyncOpenAI if is_async else openai.OpenAI)(api_key="sk-test", http_client=http), seen
