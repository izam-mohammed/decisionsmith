import asyncio
import json
import types
from typing import Literal

import httpx
import pytest
import respx
from pydantic import BaseModel

import decisionsmith as ds
from decisionsmith import offline
from decisionsmith.engines import EngineError, LLMEngine, TextEngine, framework, from_string, http, write
from decisionsmith.engines import structured as st
from decisionsmith.engines.llm import resolve
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket

Q = compile_schema(Ticket).questions()
URL = "https://api.openai.com/v1/chat/completions"
GOOD = '{"q0": "sales", "q1": false}'
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"


def chat(content, usage=None):
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage is not False:
        body["usage"] = usage or {"prompt_tokens": 10, "completion_tokens": 2}
    return httpx.Response(200, json=body)


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DS_OFFLINE", raising=False)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)


@pytest.mark.parametrize(
    "model,url,expected",
    [
        ("gpt-5-mini", None, ("openai", "gpt-5-mini", "https://api.openai.com/v1", "OPENAI_API_KEY")),
        ("o3", None, ("openai", "o3", "https://api.openai.com/v1", "OPENAI_API_KEY")),
        ("openai/gpt-5", None, ("openai", "gpt-5", "https://api.openai.com/v1", "OPENAI_API_KEY")),
        ("gemini-2.5-flash", None, ("gemini", "gemini-2.5-flash", GEMINI, "GEMINI_API_KEY")),
        ("gemini/gemini-2.5-pro", None, ("gemini", "gemini-2.5-pro", GEMINI, "GEMINI_API_KEY")),
        ("groq/llama-3.3-70b", None, ("groq", "llama-3.3-70b", "https://api.groq.com/openai/v1", "GROQ_API_KEY")),
        ("openrouter/a/b", None, ("openrouter", "a/b", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")),
        ("together/m/x", None, ("together", "m/x", "https://api.together.ai/v1", "TOGETHER_API_KEY")),
        ("fireworks/m", None, ("fireworks", "m", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY")),
        ("deepseek/deepseek-chat", None, ("deepseek", "deepseek-chat", "https://api.deepseek.com", "DEEPSEEK_API_KEY")),
        ("xai/grok-4", None, ("xai", "grok-4", "https://api.x.ai/v1", "XAI_API_KEY")),
        ("mistral/mistral-small", None, ("mistral", "mistral-small", "https://api.mistral.ai/v1", "MISTRAL_API_KEY")),
        ("ollama/qwen3", None, ("ollama", "qwen3", "http://localhost:11434/v1", None)),
        ("claude-haiku-4-5", None, ("anthropic", "claude-haiku-4-5", None, "ANTHROPIC_API_KEY")),
        ("anthropic/claude-sonnet-5", None, ("anthropic", "claude-sonnet-5", None, "ANTHROPIC_API_KEY")),
        ("my-model", "http://h/v1/", ("custom", "my-model", "http://h/v1", None)),
        ("openai/my-model", "http://h/v1", ("custom", "my-model", "http://h/v1", None)),
    ],
)
def test_resolve(model, url, expected):
    assert resolve(model, url) == expected


def test_resolve_unknown():
    with pytest.raises(ValueError, match="provider prefix"):
        resolve("mystery", None)
    with pytest.raises(ValueError, match="model name"):
        ds.LLM("")


@respx.mock
def test_ask_json_schema():
    route = respx.post(URL).mock(return_value=chat(GOOD))
    e = ds.LLM("gpt-5-mini", max_tokens=50)
    out = e.ask("How much is the plan?", Q)
    assert out["answers"]["team"] == {
        "type": "choice",
        "probabilities": {"billing": 0.0, "technical": 0.0, "sales": 1.0},
        "confidence": 1.0,
        "choice": "sales",
    }
    assert out["answers"]["wants_refund"]["noul"] == 0.0
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 2, "cost_usd": None}
    body = json.loads(route.calls[0].request.content)
    assert body["temperature"] == 0.0 and body["max_tokens"] == 50 and body["model"] == "gpt-5-mini"
    assert body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False and schema["required"] == ["q0", "q1"]
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"
    user = body["messages"][1]["content"]
    assert "<text>\nHow much is the plan?\n</text>" in user and "- billing: payments and refunds" in user
    assert repr(e) == "ds.LLM('gpt-5-mini')"


@respx.mock
def test_degrades_on_unsupported_options():
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(400, text="Unsupported value: 'temperature' does not support 0"),
            httpx.Response(400, text="response_format json_schema is not supported"),
            httpx.Response(400, text="Invalid response_format type json_object"),
            chat("```json\n" + GOOD + "\n```", usage=False),
        ]
    )
    out = ds.LLM("o3").ask("x", Q)
    assert out["answers"]["team"]["choice"] == "sales" and out["usage"]["input_tokens"] == 0
    bodies = [json.loads(c.request.content) for c in route.calls]
    assert "temperature" in bodies[0] and "temperature" not in bodies[1]
    assert bodies[2]["response_format"] == {"type": "json_object"} and "response_format" not in bodies[3]


@respx.mock
def test_user_options_win():
    route = respx.post(URL).mock(side_effect=[httpx.Response(400, text="temperature bad"), chat(GOOD)])
    e = ds.LLM("gpt-5", temperature=0.3, response_format={"type": "text"})
    with pytest.raises(EngineError, match="HTTP 400"):
        e.ask("x", Q)
    assert json.loads(route.calls[0].request.content)["response_format"] == {"type": "text"}


@respx.mock
def test_repairs_an_invalid_reply_once():
    route = respx.post(URL).mock(side_effect=[chat('Sure! {"q0": "nope", "q1": true}'), chat(GOOD)])
    assert ds.LLM("gpt-5").ask("x", Q)["answers"]["team"]["choice"] == "sales"
    assert "not valid" in json.loads(route.calls[1].request.content)["messages"][1]["content"]
    respx.post(URL).mock(side_effect=[chat("no json here"), chat("still none")])
    with pytest.raises(EngineError, match="not valid JSON"):
        ds.LLM("gpt-5").ask("x", Q)


@respx.mock
@pytest.mark.parametrize(
    "resp,match",
    [
        (httpx.Response(401), "authentication failed"),
        (httpx.Response(404, text="no model"), "HTTP 404"),
        (httpx.Response(200, text="<html>"), "unexpected response"),
        (httpx.Response(200, json={"choices": []}), "unexpected response"),
    ],
)
def test_errors(resp, match):
    respx.post(URL).mock(return_value=resp)
    with pytest.raises(EngineError, match=match):
        ds.LLM("gpt-5").ask("x", Q)


@respx.mock
def test_retries_then_gives_up():
    respx.post(URL).mock(side_effect=[httpx.Response(429), httpx.ReadTimeout("slow"), chat(GOOD)])
    assert ds.LLM("gpt-5").ask("x", Q)["answers"]["team"]["choice"] == "sales"
    respx.post(URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(EngineError, match="no response after 3"):
        ds.LLM("gpt-5").ask("x", Q)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(EngineError, match="export GROQ_API_KEY"):
        ds.LLM("groq/llama").ask("x", Q)


@respx.mock
def test_custom_url_and_ollama_send_no_key(monkeypatch):
    route = respx.post("http://h/v1/chat/completions").mock(return_value=chat(GOOD))
    e = ds.LLM("my-model", url="http://h/v1")
    assert e.name == "my-model@http://h/v1" and e.ask("x", Q)["answers"]["team"]["choice"] == "sales"
    assert "authorization" not in route.calls[0].request.headers
    ds.LLM("m", url="http://h/v1", api_key="k").ask("x", Q)
    assert route.calls[1].request.headers["authorization"] == "Bearer k"
    ollama = respx.post("http://localhost:11434/v1/chat/completions").mock(return_value=chat(GOOD))
    ds.LLM("ollama/qwen3").ask("x", Q)
    assert json.loads(ollama.calls[0].request.content)["model"] == "qwen3"


@respx.mock
def test_async_and_write():
    route = respx.post(URL).mock(side_effect=[chat("junk"), chat(GOOD), chat('{"texts": ["one", " ", "two"]}')])
    e = ds.LLM("gpt-5")

    async def go():
        return await e.aask("x", Q)

    assert asyncio.run(go())["answers"]["team"]["choice"] == "sales"
    assert write(e, "write 2", 2) == ["one", "two"]
    body = json.loads(route.calls[2].request.content)
    assert body["temperature"] == 1.0 and "exactly 2 texts" in body["messages"][1]["content"]
    with pytest.raises(EngineError, match="can't write"):
        write(ds.testing.FakeEngine(), "w", 1)


def claude(handler, is_async=False):
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    e = ds.LLM("claude-haiku-4-5", stop_sequences=["END"])
    make = anthropic.DefaultAsyncHttpxClient if is_async else anthropic.DefaultHttpxClient
    cls = anthropic.AsyncAnthropic if is_async else anthropic.Anthropic
    e._claude_clients[is_async] = cls(
        api_key="k", max_retries=0, http_client=make(transport=httpx2.MockTransport(handler))
    )
    return e


def message(text, stop="end_turn"):
    import httpx2

    content = [{"type": "text", "text": text}] if text else []
    return httpx2.Response(
        200,
        json={
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    )


def test_claude_uses_the_sdk():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return message(GOOD if len(seen) == 1 else '{"texts": ["a", "b"]}')

    e = claude(handler)
    out = e.ask("x", Q)
    assert out["answers"]["team"]["choice"] == "sales" and out["usage"]["input_tokens"] == 7
    body = seen[0]
    assert body["model"] == "claude-haiku-4-5" and body["max_tokens"] == 1024 and body["stop_sequences"] == ["END"]
    assert body["output_config"]["format"]["type"] == "json_schema" and "system" in body
    assert e.write("w", 2) == ["a", "b"] and seen[1]["max_tokens"] == 4096


def test_claude_async_refusal_and_errors():
    e = claude(lambda r: message(GOOD), is_async=True)
    assert asyncio.run(e.aask("x", Q))["answers"]["team"]["choice"] == "sales"
    with pytest.raises(EngineError, match="declined"):
        claude(lambda r: message("", stop="refusal")).ask("x", Q)
    with pytest.raises(EngineError, match="declined"):
        asyncio.run(claude(lambda r: message("", stop="refusal"), is_async=True).aask("x", Q))
    import httpx2

    with pytest.raises(EngineError, match="Error"):
        claude(
            lambda r: httpx2.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
        ).ask("x", Q)
    with pytest.raises(EngineError, match="Error"):
        asyncio.run(claude(lambda r: httpx2.Response(400, json={}), is_async=True).aask("x", Q))
    with pytest.raises(EngineError, match="did not match"):
        claude(lambda r: message("I can't help", stop="refusal")).ask("x", Q)
    e = LLMEngine("claude-haiku-4-5")
    with pytest.raises(EngineError, match="no structured answer"):
        e._read_claude(types.SimpleNamespace(stop_reason="max_tokens", parsed_output=None))


def test_claude_client_setup(monkeypatch):
    pytest.importorskip("anthropic")
    e = LLMEngine("claude-haiku-4-5", api_key="k", retries=1, timeout=5)
    sync, async_ = e._claude(False), e._claude(True)
    assert sync.api_key == "k" and sync.max_retries == 1 and type(async_).__name__ == "AsyncAnthropic"
    assert e._claude(False) is sync
    import builtins

    real = builtins.__import__

    def no_anthropic(name, *a, **k):
        if name == "anthropic":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    with pytest.raises(EngineError, match=r"decisionsmith\[anthropic\]"):
        LLMEngine("claude-haiku-4-5").ask("x", Q)


def test_offline_mode(monkeypatch):
    monkeypatch.setenv("DS_OFFLINE", "1")
    assert offline.enabled()
    for e in (ds.LLM("gpt-5"), ds.LLM("claude-haiku-4-5"), from_string("jev"), from_string("systemone:http://x")):
        a = e.ask("please refund my billing charge, yes", Q)["answers"]
        assert a["team"]["choice"] == "billing" and a["wants_refund"]["noul"] == 1.0
        assert asyncio.run(e.aask("a sales question, no", Q))["answers"]["team"]["choice"] == "sales"
    texts = ds.LLM("gpt-5").write(st.write_prompt("Every text must have these answers:\n- q -> billing", 3), 3)
    assert len(texts) == 3 and all("billing" in t for t in texts)
    monkeypatch.setenv("DS_OFFLINE", "0")
    assert not offline.enabled()


def test_offline_fill_and_laya(monkeypatch, tmp_path):
    class M(BaseModel):
        ok: bool
        pick: Literal["x", "y"]

    assert offline.fill("<text>\nfalse\n</text>", M)["ok"] is False
    assert offline.fill("nothing", M)["pick"] in ("x", "y") and isinstance(offline.fill("q", M)["ok"], bool)
    assert len(offline.fill("write some", st.texts_model(2))["texts"]) == 3
    monkeypatch.setenv("DS_OFFLINE", "1")
    monkeypatch.setenv("DS_LAYA", str(tmp_path))
    assert offline.laya("laya") == str(tmp_path) and from_string("laya:multilingual").model_id == str(tmp_path)
    monkeypatch.delenv("DS_LAYA")
    assert offline.laya("laya") is None


def test_text_engine():
    replies = iter(["junk", GOOD])
    e = TextEngine("mine", lambda s, u: next(replies))
    assert e.ask("x", Q)["answers"]["team"]["choice"] == "sales"

    async def acomplete(s, u):
        return (GOOD, {"input_tokens": 3})

    a = TextEngine("a", lambda s, u: (GOOD, None), acomplete)
    assert a.ask("x", Q)["usage"] == {}
    assert asyncio.run(a.aask("x", Q))["usage"] == {"input_tokens": 3}
    bad = iter(["x", '{"q0": "sales", "q1": true}'])

    async def abad(s, u):
        return next(bad)

    assert asyncio.run(TextEngine("b", lambda s, u: "", abad).aask("x", Q))["answers"]["wants_refund"]["noul"] == 1.0
    sync_only = TextEngine("s", lambda s, u: GOOD)
    assert asyncio.run(sync_only.aask("x", Q))["answers"]["team"]["choice"] == "sales"
    assert TextEngine("w", lambda s, u: '{"texts": ["a"]}').write("w", 1) == ["a"]

    def boom(s, u):
        raise RuntimeError("quota")

    async def aboom(s, u):
        raise RuntimeError("quota")

    async def aengine(s, u):
        raise EngineError("x", "already")

    with pytest.raises(EngineError, match="quota"):
        TextEngine("x", boom).ask("x", Q)
    with pytest.raises(EngineError, match="quota"):
        asyncio.run(TextEngine("x", boom, aboom).aask("x", Q))
    with pytest.raises(EngineError, match="already"):
        asyncio.run(TextEngine("x", boom, aengine).aask("x", Q))
    with pytest.raises(EngineError, match="already"):
        TextEngine("x", lambda s, u: (_ for _ in ()).throw(EngineError("x", "already"))).ask("x", Q)


def test_offline_text_engine(monkeypatch):
    monkeypatch.setenv("DS_OFFLINE", "1")

    async def never(s, u):
        raise AssertionError

    e = TextEngine("x", lambda s, u: 1 / 0, never)
    assert e.ask("a technical bug", Q)["answers"]["team"]["choice"] == "technical"
    assert asyncio.run(e.aask("sales", Q))["answers"]["team"]["choice"] == "sales"


def test_prompt_and_parse():
    score = {"lvl": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high: very"]}}
    model = st.answers_model(score)
    assert st.answers_model(score) is model
    assert st.one_hot(model(q0="high: very"), score, {}, "m")["answers"]["lvl"] == {
        "type": "score",
        "probabilities": {"0": 0.0, "1": 1.0},
        "confidence": 1.0,
        "score": 1.0,
    }
    noul = {"s": {"type": "noul", "instructions": "Spam?", "criteria": {"true": "junk"}}}
    assert "- true: junk" in st.prompt("x", noul) and '{"q0": true}' in st.prompt("x", noul)
    assert st.options({"type": "choice", "criteria": ["a", "b"]}) == (["a", "b"], {})
    assert st.options({"type": "choice"}) == ([], {})
    assert st.parse('text {"q0": "low"} more', model).q0 == "low"  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        st.parse("nothing", model)


def test_litellm_prefix_routes_to_the_integration():
    e = from_string("litellm/bedrock/x")
    assert isinstance(e, TextEngine) and e.name == "litellm/bedrock/x"
    assert ds.LLM("litellm/groq/y", api_base="http://h").name == "litellm/groq/y"


def test_framework_detection():
    def obj(module):
        return type("Thing", (), {"__module__": module})()

    assert framework(obj("langchain_openai.chat_models.base")) == "langchain"
    assert framework(obj("langchain_core.x")) == "langchain"
    assert framework(obj("llama_index.llms.openai")) == "llamaindex"
    assert framework(obj("dspy.clients.lm")) == "dspy"
    assert framework(obj("crewai.llm")) == "crewai"
    assert framework(obj("pydantic_ai.models.openai")) == "pydantic_ai"
    assert framework(obj("haystack_integrations.components")) == "haystack"
    assert framework(obj("smolagents.models")) == "smolagents"
    assert framework(obj("autogen_ext.models.openai")) == "autogen"
    assert framework(obj("semantic_kernel.connectors")) == "semantic_kernel"
    assert framework(obj("somewhere.else")) is None
    sub = type("Mine", (type(obj("dspy.x")),), {"__module__": "myapp"})()
    assert framework(sub) == "dspy"
