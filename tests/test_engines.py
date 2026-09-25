import asyncio
import sys
import types

import httpx
import pytest
import respx

import decisionsmith as ds
from decisionsmith import engines as eng
from decisionsmith.engines import (
    EngineError,
    JevEngine,
    LayaEngine,
    LLMEngine,
    SystemOneEngine,
    ask_many,
    from_string,
    http,
    resolve_laya,
    response,
)
from decisionsmith.schema import compile_schema
from tests.conftest import Ticket

Q = compile_schema(Ticket).questions()
JEV_REPLY = {
    "model": "jev-1.13.0",
    "answers": {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 1.0,
            "probabilities": {"billing": 1.0, "technical": 0.0, "sales": 0.0},
        },
        "wants_refund": {"type": "noul", "noul": 0.9, "confidence": 0.9},
    },
    "usage": {"input_tokens": 786, "output_tokens": 47},
}


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)


@respx.mock
def test_systemone_ok_and_body():
    route = respx.post("http://h:8000/v1/systemone").mock(return_value=httpx.Response(200, json=JEV_REPLY))
    e = SystemOneEngine("http://h:8000/", api_key="k", model="m")
    assert e.ask("text", Q) == JEV_REPLY
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer k"
    assert httpx.Request("POST", "x", content=sent.content).read() and b'"model":"m"' in sent.content.replace(b" ", b"")
    assert e.name == "systemone:http://h:8000/"
    assert SystemOneEngine("http://h/v1/systemone").url == "http://h/v1/systemone"


@respx.mock
def test_systemone_retries_then_succeeds(no_sleep):
    respx.post("http://h/v1/systemone").mock(
        side_effect=[httpx.Response(503), httpx.ConnectTimeout("slow"), httpx.Response(200, json=JEV_REPLY)]
    )
    assert SystemOneEngine("http://h").ask("t", Q)["answers"]["team"]["choice"] == "billing"


@respx.mock
def test_systemone_gives_up(no_sleep):
    respx.post("http://h/v1/systemone").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(EngineError, match="no response after 3 attempts") as e:
        SystemOneEngine("http://h").ask("t", Q)
    assert e.value.fix


@respx.mock
@pytest.mark.parametrize(
    "resp,match",
    [
        (httpx.Response(401), "authentication failed"),
        (httpx.Response(422, text="bad question"), "HTTP 422: bad question"),
        (httpx.Response(200, text="<html>"), "not JSON"),
        (httpx.Response(200, json={"answers": {"team": {}}}), "no answer for"),
        (httpx.Response(503), "HTTP 503"),
    ],
)
def test_systemone_errors(resp, match):
    respx.post("http://h/v1/systemone").mock(return_value=resp)
    with pytest.raises(EngineError, match=match):
        SystemOneEngine("http://h", retries=0).ask("t", Q)


def test_backoff_delay():
    assert http.delay(1) == 1.0
    assert http.delay(0, httpx.Response(429, headers={"retry-after": "3"})) == 3.0
    assert http.delay(0, httpx.Response(429, headers={"retry-after": "999"})) == 20.0
    assert http.delay(2, httpx.Response(429, headers={"retry-after": "soon"})) == 2.0


@respx.mock
def test_jev(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    route = respx.post(eng.JEV_URL).mock(return_value=httpx.Response(200, json=JEV_REPLY))
    e = JevEngine()
    assert e.name == "jev" and JevEngine("jev-2").name == "jev:jev-2"
    e.ask("t", Q)
    assert route.calls[0].request.headers["authorization"] == "Bearer secret"
    respx.post(eng.JEV_URL).mock(return_value=httpx.Response(403))
    with pytest.raises(EngineError, match="TYPESAFE_API_KEY"):
        e.ask("t", Q)
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(EngineError, match="no API key"):
        JevEngine()


def test_response_shapes():
    fake = ds.testing.FakeEngine()
    assert response(fake, JEV_REPLY, Q)[1]["input_tokens"] == 786
    bare = JEV_REPLY["answers"]
    assert response(fake, bare, Q) == (bare, {})
    assert response(fake, {"answers": bare, "usage": None}, Q)[1] == {}
    with pytest.raises(EngineError, match="expected a dict"):
        response(fake, [], Q)


def test_from_string(monkeypatch, tiny):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert from_string(None) is None
    assert isinstance(from_string("fake"), ds.testing.FakeEngine)
    assert isinstance(from_string("jev"), JevEngine) and from_string("jev:jev-2").name == "jev:jev-2"
    assert isinstance(from_string("systemone:http://x"), SystemOneEngine)
    assert isinstance(from_string("laya"), LayaEngine)
    assert from_string("laya:multilingual").subfolder == "multilingual"
    assert from_string("laya:%s" % tiny).model_id == str(tiny)
    assert isinstance(from_string("gpt-5"), LLMEngine)
    custom = ds.testing.FakeEngine(name="mine")
    assert from_string(custom) is custom
    for bad in ("", "systemone:"):
        with pytest.raises(ValueError):
            from_string(bad)
    with pytest.raises(TypeError, match="supported framework"):
        from_string(object())


def test_resolve_laya(tiny):
    assert resolve_laya("laya") == (eng.LAYA_REPO, None)
    assert resolve_laya("typed-decisions") == (eng.LAYA_REPO, "typed-decisions")
    assert resolve_laya("org/model") == ("org/model", None)
    with pytest.raises(EngineError, match="no such checkpoint"):
        resolve_laya("./missing/dir")


def test_laya_engine(tiny):
    e = LayaEngine(str(tiny), device="cpu")
    out = e.ask("you charged me twice", Q)
    assert set(out["answers"]) == set(Q)
    assert len(ask_many(e, ["a", "b", "c"], Q)) == 3
    assert e.agent is e.agent
    bad = {"x": {"type": "choice", "instructions": "?", "criteria": {}}}
    with pytest.raises(EngineError, match="at least one"):
        e.ask("t", bad)


def test_laya_engine_load_errors(monkeypatch, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(EngineError, match="could not load"):
        LayaEngine(str(tmp_path / "empty")).agent
    import builtins

    real = builtins.__import__

    def no_laya(name, *a, **k):
        if name == "laya":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_laya)
    with pytest.raises(EngineError, match="not installed"):
        LayaEngine("org/other-model").agent


def test_ask_many_falls_back_to_loop():
    fake = ds.testing.FakeEngine()
    assert len(ask_many(fake, ["a", "b"], Q)) == 2


def test_engine_error_text():
    assert str(EngineError("jev", "down", "retry")) == "engine 'jev': down\n  fix: retry"
    assert str(EngineError("jev", "down")) == "engine 'jev': down"
    assert isinstance(ds.testing.FakeEngine(), ds.Engine)


@respx.mock
def test_systemone_async(monkeypatch):
    async def no_sleep(s):
        return None

    monkeypatch.setattr(http.asyncio, "sleep", no_sleep)
    e = SystemOneEngine("http://h")

    async def go():
        first = await e.aask("t", Q)
        again = await e.aask("t", Q)
        return first, again, e._clients.async_()

    respx.post("http://h/v1/systemone").mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectTimeout("slow"),
            httpx.Response(200, json=JEV_REPLY),
            httpx.Response(200, json=JEV_REPLY),
        ]
    )
    first, again, client = asyncio.run(go())
    assert first["answers"]["team"]["choice"] == "billing" == again["answers"]["team"]["choice"]
    respx.post("http://h/v1/systemone").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(EngineError, match="no response after 3 attempts"):
        asyncio.run(e.aask("t", Q))


def test_framework_objects_route_to_their_integration(monkeypatch):
    made = []
    fake = types.ModuleType("decisionsmith.integrations.dspy")
    fake.teacher = lambda obj: made.append(obj) or ds.testing.FakeEngine(name="wrapped")
    monkeypatch.setitem(sys.modules, "decisionsmith.integrations.dspy", fake)
    lm = type("LM", (), {"__module__": "dspy.clients.lm"})()
    assert from_string(lm).name == "wrapped" and made == [lm]
