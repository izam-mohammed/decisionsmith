import asyncio
import sys
import time
from typing import Annotated, Literal

import httpx
import pytest
from pydantic import BaseModel, Field

import decisionsmith as ds
from decisionsmith import cli
from decisionsmith import serve as srv
from decisionsmith.engines import EngineError, LayaEngine, SystemOneEngine
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

LABELS = ["billing", "technical", "sales"]
TEXT = "you charged me twice refund"


class Rated(BaseModel):
    urgency: Annotated[Literal["low", "medium", "high"], ds.Scale, ds.Options(high="today")] = Field(
        description="How urgent is it?"
    )
    spam: bool


def teacher(**kw):
    return FakeEngine(truth, confidence=1.0, name="teacher", **kw)


@pytest.fixture
def h(db):
    with ds.harness(Ticket, teacher=teacher(), student=FakeEngine(confidence=0.6), log=db) as harness:
        yield harness


@pytest.fixture
def client(h):
    with TestClient(srv.app(h)) as c:
        yield c


def saved(tiny, tmp_path, spec=LABELS):
    m = ds.model(spec, str(tiny))
    m.trained = str(tiny)
    return m.save(str(tmp_path / "team"), verbose=False)


def test_health_decide_label_status_and_docs(client, h):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model"] == "Ticket" and health["fields"]["team"] == LABELS
    assert health["modes"] == {"team": "cascade", "wants_refund": "cascade"} and "student" not in health
    r = client.post("/v1/decide", json={"text": TEXT})
    assert r.status_code == 200
    body = r.json()
    assert body["value"] == {"team": "billing", "wants_refund": True}
    assert body["source"] == {"team": "teacher", "wants_refund": "teacher"} and body["sure"] is True
    assert set(body["confidence"]) == {"team", "wants_refund"} and body["latency_ms"] >= 0
    ok = client.post("/v1/label", json={"id": body["id"], "labels": {"team": "technical"}})
    assert ok.json() == {"id": body["id"], "labelled": ["team"]}
    assert h.log.rows("Ticket")[0]["labels"] == {"team": "technical"}
    status = client.get("/v1/status").json()
    assert status["model"] == "Ticket" and status["teacher"] == "teacher" and status["student"] == "fake"
    assert status["fields"]["team"]["human_labels"] == 1 and status["modes"]["team"] == "cascade"
    assert client.get("/openapi.json").json()["paths"].keys() >= {"/v1/decide", "/v1/systemone", "/v1/label"}
    assert client.get("/docs").status_code == 200


def test_label_errors(client):
    assert client.post("/v1/label", json={"id": "nope", "labels": {"team": "billing"}}).json()["error"] == {
        "code": "not_found",
        "message": "no decision with that id in the log",
        "fix": "use an id a decision returned",
    }
    rid = client.post("/v1/decide", json={"text": TEXT}).json()["id"]
    bad = client.post("/v1/label", json={"id": rid, "labels": {"colour": "red"}})
    assert bad.status_code == 422 and "unknown fields ['colour']" in bad.json()["error"]["message"]
    empty = client.post("/v1/label", json={"id": rid, "labels": {}})
    assert empty.status_code == 422 and empty.json()["error"]["message"] == "labels is empty"
    wrong = client.post("/v1/label", json={"id": rid, "labels": {"team": "marketing"}})
    assert wrong.status_code == 422 and "not an option" in wrong.json()["error"]["message"]


def test_without_a_log(tmp_path):
    h = ds.harness(Ticket, teacher=teacher(), log=None)
    with TestClient(srv.app(h)) as c:
        assert c.post("/v1/decide", json={"text": TEXT}).json()["id"] is None
        for r in (c.get("/v1/status"), c.post("/v1/label", json={"id": "x", "labels": {"team": "sales"}})):
            assert r.status_code == 409 and r.json()["error"]["code"] == "no_log"
            assert r.json()["error"]["fix"] == "start the server with a log (--log decisions.db)"


def test_request_errors_are_json_with_a_fix(client):
    missing = client.post("/v1/decide", json={"txt": "hi"})
    assert missing.status_code == 422
    assert missing.json()["error"] == {
        "code": "invalid",
        "message": "text: Field required; txt: Extra inputs are not permitted",
        "fix": "see the request examples at /docs",
    }
    assert (
        "JSON"
        in client.post("/v1/decide", content=b"{nope", headers={"content-type": "application/json"}).json()["error"][
            "message"
        ]
    )
    empty = client.post("/v1/decide", json={"text": "   "})
    assert empty.status_code == 422 and empty.json()["error"]["message"] == "text is empty"
    long = client.post("/v1/decide", json={"text": "a" * (srv.MAX_TEXT_CHARS + 1)})
    assert long.status_code == 413 and long.json()["error"]["code"] == "too_large"
    nf = client.get("/nope")
    assert nf.status_code == 404 and nf.json()["error"]["code"] == "not_found"
    assert client.get("/v1/decide").json()["error"]["code"] == "method_not_allowed"
    big = client.post("/v1/decide", content=b"x" * (srv.MAX_BODY_BYTES + 1))
    assert big.status_code == 413 and big.json()["error"]["message"] == "request body is over 1048576 bytes"


def test_body_limit_streams_and_passes_other_scopes():
    seen = []

    async def inner(scope, receive, send):
        seen.append(await receive())
        seen.append(await receive())

    limit = srv._BodyLimit(inner, limit=10)

    def feeder(*messages):
        queue = list(messages)

        async def receive():
            return queue.pop(0) if queue else {"type": "http.disconnect"}

        return receive

    sent = []

    async def send(message):
        sent.append(message)

    async def go():
        http = {"type": "http", "headers": [(b"content-length", b"abc")]}
        await limit(
            http,
            feeder(
                {"type": "http.request", "body": b"12345", "more_body": True}, {"type": "http.request", "body": b"678"}
            ),
            send,
        )
        assert seen[0] == {"type": "http.request", "body": b"12345678", "more_body": False}
        assert seen[1] == {"type": "http.disconnect"}
        await limit(
            {"type": "http"},
            feeder(
                {"type": "http.request", "body": b"123456", "more_body": True},
                {"type": "http.request", "body": b"123456"},
            ),
            send,
        )
        assert sent[0]["status"] == 413 and b"over 10 bytes" in sent[1]["body"]
        await limit({"type": "http", "headers": []}, feeder({"type": "http.disconnect"}), send)
        assert len(seen) == 2 and len(sent) == 2
        await limit({"type": "lifespan"}, feeder({"type": "lifespan.startup"}), send)
        assert seen[2] == {"type": "lifespan.startup"}

    asyncio.run(go())


def test_api_key(h, monkeypatch):
    monkeypatch.setenv(srv.API_KEY_ENV, "s3cret")
    with TestClient(srv.app(h)) as c:
        assert c.get("/health").status_code == 200
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer s\xe9cret".encode("latin-1")}):
            r = c.post("/v1/decide", json={"text": TEXT}, headers=headers)
            assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
            assert r.json()["error"]["code"] == "unauthorized" and srv.API_KEY_ENV in r.json()["error"]["fix"]
        assert c.get("/metrics").status_code == 401
        good = {"Authorization": "Bearer s3cret"}
        assert c.post("/v1/decide", json={"text": TEXT}, headers=good).status_code == 200
        assert c.get("/metrics", headers=good).status_code == 200
    monkeypatch.delenv(srv.API_KEY_ENV)
    with TestClient(srv.app(h, api_key="given")) as c:
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer given"}).status_code == 200


def test_systemone_is_jev_shaped(client, h):
    r = client.post(
        "/v1/systemone",
        json={"model": "jev-1.13.0", "state": TEXT, "questions": {"team": {"type": "choice"}, "wants_refund": {}}},
    ).json()
    assert set(r) == {"model", "answers", "usage", "routing"} and r["model"] == "Ticket"
    team, refund = r["answers"]["team"], r["answers"]["wants_refund"]
    assert set(team) == {"type", "choice", "probabilities", "confidence"} and team["choice"] == "billing"
    assert set(team["probabilities"]) == set(LABELS) and team["confidence"] == max(team["probabilities"].values())
    assert set(refund) == {"type", "noul", "confidence"} and refund["noul"] == 1.0
    assert r["usage"] == {"input_tokens": len(TEXT.split()), "output_tokens": 0}
    assert r["routing"]["source"] == {"team": "teacher", "wants_refund": "teacher"} and r["routing"]["sure"]
    assert h.log.rows("Ticket")[-1]["id"] == r["routing"]["id"]
    only = client.post("/v1/systemone", json={"state": TEXT, "questions": {"team": {"criteria": LABELS}}}).json()
    assert list(only["answers"]) == ["team"] and list(only["routing"]["source"]) == ["team"]
    assert set(client.post("/v1/systemone", json={"state": TEXT}).json()["answers"]) == {"team", "wants_refund"}


@pytest.mark.parametrize(
    "questions,message",
    [
        ({"colour": {"type": "choice"}}, "unknown: ['colour']"),
        ({"team": {"type": "noul"}}, "question 'team' is type 'choice' here, not 'noul'"),
        ({"team": {"criteria": ["a", "b"]}}, "question 'team' has options"),
    ],
)
def test_systemone_rejects_questions_it_cannot_answer(client, questions, message):
    r = client.post("/v1/systemone", json={"state": TEXT, "questions": questions})
    assert r.status_code == 422 and message in r.json()["error"]["message"] and r.json()["error"]["fix"]


def test_systemone_score_answers(db):
    student = FakeEngine(lambda t: {"urgency": "high", "spam": False}, confidence=0.9)
    with ds.harness(Rated, student=student, log=db) as h, TestClient(srv.app(h)) as c:
        r = c.post("/v1/systemone", json={"state": "server down", "questions": {"urgency": {"type": "score"}}})
        urgency = r.json()["answers"]["urgency"]
        assert set(urgency) == {"type", "score", "legend", "probabilities", "confidence"}
        assert urgency["legend"] == {"0": "low", "1": "medium", "2": "high: today"}
        assert list(urgency["probabilities"]) == ["0", "1", "2"] and urgency["score"] == pytest.approx(1.85)
        assert r.json()["usage"] == {"input_tokens": 0, "output_tokens": 0}
        assert c.post("/v1/decide", json={"text": "server down"}).json()["value"] == {"urgency": "high", "spam": False}


def test_contract_our_client_reads_our_server(client, h):
    """SystemOneEngine (the Jev client code) pointed at this server decodes every answer."""
    import respx

    def forward(request):
        r = client.post("/v1/systemone", content=request.content, headers={"content-type": "application/json"})
        return httpx.Response(r.status_code, content=r.content, headers=r.headers)

    with respx.mock(assert_all_called=True) as mock:
        mock.post("http://ds.test/v1/systemone").mock(side_effect=forward)
        engine = SystemOneEngine("http://ds.test")
        raw = engine.ask(TEXT, h.schema.questions())
        via = ds.harness(Ticket, student=engine, log=None)
        assert via.decide(TEXT).value == Ticket(team="billing", wants_refund=True)
        assert ds.harness(Ticket, student=engine, log=None, mode="student").decide(TEXT).sure
    answers, usage = ds.engines.response(engine, raw, h.schema.questions())
    assert h.schema.distributions(answers)["team"]["billing"] == 1.0 and usage["input_tokens"] > 0


def test_contract_same_answer_keys_as_laya(tiny, db):
    """Every answer carries the keys Laya's own /v1/systemone answer carries (minus Laya-only extras)."""
    student = LayaEngine(str(tiny))
    extras = {"answer_confidence", "action"}
    with ds.harness(Rated, student=student, log=db) as h, TestClient(srv.app(h)) as c:
        laya = student.ask("server down", h.schema.questions())
        ours = c.post("/v1/systemone", json={"state": "server down"}).json()
    assert set(ours) >= set(laya) - {"routing"} and set(ours["usage"]) == set(laya["usage"])
    for qid, answer in laya["answers"].items():
        assert set(ours["answers"][qid]) == set(answer) - extras, qid
        assert ours["answers"][qid]["type"] == answer["type"]
        if "legend" in answer:
            assert ours["answers"][qid]["legend"] == answer["legend"]


def test_engine_failures_never_crash(db):
    down = FakeEngine(error="rate limited", name="teacher")
    with (
        ds.harness(Ticket, teacher=down, student=FakeEngine(confidence=0.6), log=db) as h,
        TestClient(srv.app(h)) as c,
    ):
        r = c.post("/v1/decide", json={"text": TEXT}).json()
        assert r["source"] == {"team": "student", "wants_refund": "student"} and r["sure"] is False
        metrics = c.get("/metrics").text
        assert "decisionsmith_teacher_errors_total 1" in metrics
        assert 'decisionsmith_decisions_total{sure="false"} 1' in metrics
    with ds.harness(Ticket, teacher=down, log=db) as h, TestClient(srv.app(h)) as c:
        r = c.post("/v1/decide", json={"text": TEXT})
        assert r.status_code == 503 and r.json()["error"]["code"] == "engine"
        assert "no engine could answer" in r.json()["error"]["message"] and "doctor" in r.json()["error"]["fix"]
        s1 = c.post("/v1/systemone", json={"state": TEXT})
        assert s1.status_code == 503


def test_unexpected_errors_are_500_json_without_details(h, monkeypatch, caplog):
    async def boom(text):
        raise RuntimeError("secret path /srv/x")

    monkeypatch.setattr(h, "adecide", boom)
    with TestClient(srv.app(h)) as c:
        r = c.post("/v1/decide", json={"text": TEXT})
    assert r.status_code == 500
    assert r.json()["error"] == {
        "code": "error",
        "message": "RuntimeError inside the harness",
        "fix": "see the server log",
    }
    assert TEXT not in caplog.text


def test_engine_error_without_fix(h, monkeypatch):
    async def fail(text):
        raise EngineError("x", "down")

    monkeypatch.setattr(h, "adecide", fail)
    with TestClient(srv.app(h)) as c:
        body = c.post("/v1/decide", json={"text": TEXT}).json()["error"]
    assert body == {"code": "engine", "message": "down", "fix": "check the engines: decisionsmith doctor"}


def test_metrics(client):
    client.post("/v1/decide", json={"text": TEXT})
    client.post("/v1/decide", json={"txt": TEXT})
    client.get("/nope")
    text = client.get("/metrics")
    assert text.headers["content-type"].startswith("text/plain")
    m = text.text
    assert 'decisionsmith_http_requests_total{path="/v1/decide",status="200"} 1' in m
    assert 'decisionsmith_http_requests_total{path="/v1/decide",status="422"} 1' in m
    assert 'decisionsmith_http_requests_total{path="other",status="404"} 1' in m
    assert 'decisionsmith_field_decisions_total{field="team",source="teacher"} 1' in m
    assert 'decisionsmith_decision_latency_seconds_bucket{le="+Inf"} 1' in m
    assert "decisionsmith_decision_latency_seconds_count 1" in m
    assert "decisionsmith_teacher_calls_total 1" in m and 'decisionsmith_teacher_tokens_total{kind="input"} 5' in m
    assert "decisionsmith_teacher_cost_usd_total 0.0" in m and "decisionsmith_teacher_unpriced_calls_total 0" in m
    assert "# TYPE decisionsmith_decision_latency_seconds histogram" in m
    assert TEXT not in m


def test_metrics_render_edges():
    m = srv._Metrics()
    m.decision({"a": "student"}, True, 20_000.0)
    m.decision({"a": "student"}, True, 3.0)
    m.teacher_call({"input_tokens": True, "output_tokens": 2.0}, None)
    m.teacher_call({}, 0.25)
    out = m.render()
    assert 'decisionsmith_decision_latency_seconds_bucket{le="0.005"} 1' in out
    assert 'decisionsmith_decision_latency_seconds_bucket{le="10.0"} 1' in out
    assert 'decisionsmith_decision_latency_seconds_bucket{le="+Inf"} 2' in out
    assert "decisionsmith_teacher_unpriced_calls_total 1" in out and "decisionsmith_teacher_cost_usd_total 0.25" in out
    assert 'decisionsmith_teacher_tokens_total{kind="input"} 0' in out
    assert 'decisionsmith_teacher_tokens_total{kind="output"} 2' in out
    assert srv._labels(path='a"b\\c\nd') == '{path="a\\"b\\\\c\\nd"}'


def test_counted_teacher_sync_async_and_errors():
    metrics = srv._Metrics()

    class Async:
        name = "async"
        extra = "kept"

        async def aask(self, text, questions):
            if text == "fail":
                raise EngineError("async", "down")
            return {"answers": {}, "usage": {"input_tokens": 3, "output_tokens": 1, "cost_usd": 0.5}}

        def ask(self, text, questions):
            raise AssertionError("not used")

    a = srv._CountedTeacher(Async(), metrics)
    assert a.name == "async" and a.extra == "kept"
    token = srv._usage.set({"input_tokens": 0, "output_tokens": 0})
    assert asyncio.run(a.aask("x", {}))["usage"]["input_tokens"] == 3
    assert srv._usage.get() == {"input_tokens": 3, "output_tokens": 1}
    srv._usage.reset(token)
    with pytest.raises(EngineError):
        asyncio.run(a.aask("fail", {}))
    s = srv._CountedTeacher(FakeEngine(), metrics)
    assert asyncio.run(s.aask("x y", {"q": {"type": "noul"}}))["usage"]["input_tokens"] == 2
    odd = srv._CountedTeacher(type("Odd", (), {"name": "odd", "ask": lambda self, t, q: {"usage": "?"}})(), metrics)
    odd.ask("x", {})
    with pytest.raises(EngineError):
        srv._CountedTeacher(FakeEngine(error="nope"), metrics).ask("x", {})
    assert metrics.teacher["calls"] == 5 and metrics.teacher["errors"] == 2
    assert metrics.cost == 0.5 and metrics.teacher["unpriced"] == 1 and metrics.teacher["input_tokens"] == 5


def test_concurrent_requests_do_not_block_the_event_loop(db):
    slow = FakeEngine(confidence=0.9, latency=0.3)
    with ds.harness(Ticket, student=slow, log=db) as h:
        api = srv.app(h)

        async def go():
            transport = httpx.ASGITransport(app=api)
            async with httpx.AsyncClient(transport=transport, base_url="http://ds.test") as c:
                t0 = time.perf_counter()
                decide = [c.post("/v1/decide", json={"text": "text %d" % i}) for i in range(6)]
                health = c.get("/health")
                *rs, hr = await asyncio.gather(*decide, health)
                return time.perf_counter() - t0, rs, hr

        elapsed, rs, hr = asyncio.run(go())
    assert all(r.status_code == 200 for r in rs) and hr.status_code == 200
    assert len({r.json()["id"] for r in rs}) == 6
    assert elapsed < 6 * 0.3 * 0.6, elapsed


def test_app_from_a_saved_model_folder(tiny, tmp_path):
    path = saved(tiny, tmp_path)
    log = str(tmp_path / "served.db")
    api = srv.app(path, teacher=teacher(), mode="shadow", log=log, collect=0.0)
    with TestClient(api) as c:
        assert c.get("/health").json()["model"] == "team-v1"
        r = c.post("/v1/decide", json={"text": "you charged me twice"}).json()
        assert set(r["value"]) == {"label"} and r["source"] == {"label": "teacher"}
        one = c.post("/v1/systemone", json={"state": "the app crashes", "questions": {"label": {}}}).json()
        assert one["model"] == "team-v1" and one["answers"]["label"]["type"] == "choice"
    rows = ds.load(path)
    assert rows.simple
    import sqlite3

    con = sqlite3.connect(log)
    assert con.execute("SELECT count(*), count(text) FROM decisions").fetchone() == (2, 0)
    con.close()


def test_app_from_a_model_and_bad_arguments(tiny, tmp_path, h):
    m = ds.load(saved(tiny, tmp_path))
    with TestClient(srv.app(m, log=None)) as c:
        assert c.post("/v1/decide", json={"text": "sync is broken"}).json()["source"] == {"label": "student"}
        assert c.get("/health").json()["modes"] == {"label": "student"}
    with pytest.raises(ValueError, match="drop teacher, collect, log"):
        srv.app(h, teacher="fake", collect=0.1, log=None)
    with pytest.raises(FileNotFoundError, match="no saved model"):
        srv.app(str(tmp_path / "missing"))
    api = srv.app(h)
    assert isinstance(h.teacher, srv._CountedTeacher)
    first = h.teacher
    srv.app(h)
    assert h.teacher is not first and not isinstance(h.teacher.engine, srv._CountedTeacher)
    assert api.title == "decisionsmith: Ticket"


def test_owned_harness_is_closed_on_shutdown(tiny, tmp_path, monkeypatch):
    closed = []
    monkeypatch.setattr(ds.core.Harness, "close", lambda self: closed.append(self))
    with TestClient(srv.app(saved(tiny, tmp_path), log=None)):
        pass
    assert len(closed) == 1


def test_needs_fastapi(h, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", None)
    with pytest.raises(ImportError, match=r"uv add 'decisionsmith\[serve\]'"):
        srv.app(h)


def test_cli_serve(tiny, tmp_path, monkeypatch, capsys):
    import uvicorn

    calls = []

    def run(api, **kw):
        with TestClient(api) as c:
            calls.append((c.get("/health").json(), kw))

    monkeypatch.setattr(uvicorn, "run", run)
    path = saved(tiny, tmp_path)
    assert cli.main(["serve", path, "--log", "none", "--port", "8123"]) == 0
    health, kw = calls[-1]
    assert kw == {"host": "127.0.0.1", "port": 8123, "log_level": "info"} and health["modes"] == {"label": "student"}
    log = str(tmp_path / "d.db")
    args = ["serve", path, "--teacher", "fake", "--mode", "label=shadow", "--log", log, "--collect", "0.1"]
    assert cli.main(args) == 0 and calls[-1][0]["modes"] == {"label": "shadow"}
    assert cli.main(["serve", path, "--teacher", "fake", "--mode", "teacher", "--log", log]) == 0
    assert calls[-1][0]["modes"] == {"label": "teacher"}
    assert cli.main(["serve", str(tmp_path / "missing")]) == cli.INVALID
    assert "no saved model" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert cli.main(["serve", path]) == cli.INVALID
    assert "needs Uvicorn" in capsys.readouterr().err
