"""`decisionsmith serve`: a saved model (or any harness) behind HTTP, with a Jev-compatible `/v1/systemone`.

    from decisionsmith.serve import app
    api = app("models/ticket-v2", teacher="claude-haiku-4-5", collect=0.1)   # an ASGI app (FastAPI)

Needs the `[serve]` extra (FastAPI + Uvicorn); `import decisionsmith` never loads it.
"""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
import logging
import os
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict

from .core import Harness, Mode, harness
from .engines import EngineError
from .schema import confidence, option_keys, top

API_KEY_ENV = "DECISIONSMITH_API_KEY"
MAX_BODY_BYTES = 1024 * 1024
MAX_TEXT_CHARS = 50_000
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
INSTALL = "uv add 'decisionsmith[serve]'"

logger = logging.getLogger("decisionsmith.serve")

_usage: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar("decisionsmith_usage", default=None)


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"text": "I was charged twice"}]})
    text: str


class DecideResponse(BaseModel):
    id: str | None
    value: dict[str, Any]
    source: dict[str, str]
    confidence: dict[str, float]
    sure: bool
    latency_ms: float


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"examples": [{"state": "I was charged twice", "questions": {"label": {"type": "choice"}}}]},
    )
    state: str
    questions: dict[str, dict[str, Any]] | None = None
    model: str | None = None


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]
    routing: dict[str, Any]


class LabelRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"id": "3f2a...", "labels": {"label": "billing"}}]}
    )
    id: str
    labels: dict[str, Any]


class _Problem(Exception):
    """An HTTP error with the JSON body `{"error": {"code", "message", "fix"}}`."""

    def __init__(self, status: int, code: str, message: str, fix: str = "") -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.fix = status, code, message, fix

    def body(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "fix": self.fix}}


class _Metrics:
    """Counters and one histogram, rendered in the Prometheus text format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = {}
        self.decisions: dict[bool, int] = {True: 0, False: 0}
        self.fields: dict[tuple[str, str], int] = {}
        self.buckets = [0] * len(LATENCY_BUCKETS)
        self.latency_sum = 0.0
        self.latency_count = 0
        self.teacher = {"calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0, "unpriced": 0}
        self.cost = 0.0

    def request(self, path: str, status: int) -> None:
        with self._lock:
            self.requests[(path, status)] = self.requests.get((path, status), 0) + 1

    def decision(self, source: dict[str, str], sure: bool, latency_ms: float) -> None:
        seconds = latency_ms / 1000
        with self._lock:
            self.decisions[sure] += 1
            for field, src in source.items():
                self.fields[(field, src)] = self.fields.get((field, src), 0) + 1
            for i, le in enumerate(LATENCY_BUCKETS):
                self.buckets[i] += seconds <= le
            self.latency_sum += seconds
            self.latency_count += 1

    def teacher_call(self, usage: dict[str, Any] | None, cost: float | None) -> None:
        with self._lock:
            self.teacher["calls"] += 1
            if usage is None:
                self.teacher["errors"] += 1
                return
            for kind in ("input_tokens", "output_tokens"):
                self.teacher[kind] += _tokens(usage, kind)
            if cost is None:
                self.teacher["unpriced"] += 1
            else:
                self.cost += cost

    def render(self) -> str:
        out: list[str] = []

        def metric(name: str, kind: str, help: str, samples: list[tuple[str, float]]) -> None:
            out.extend(["# HELP decisionsmith_%s %s" % (name, help), "# TYPE decisionsmith_%s %s" % (name, kind)])
            out.extend("decisionsmith_%s%s %s" % (name, labels, _num(v)) for labels, v in samples)

        with self._lock:
            metric(
                "http_requests_total",
                "counter",
                "HTTP requests by path and status code.",
                [(_labels(path=p, status=str(s)), n) for (p, s), n in sorted(self.requests.items())],
            )
            metric(
                "decisions_total",
                "counter",
                "Decisions made, by whether every field was sure.",
                [(_labels(sure=str(s).lower()), self.decisions[s]) for s in (True, False)],
            )
            metric(
                "field_decisions_total",
                "counter",
                "Field answers by field and the engine that gave them (student or teacher).",
                [(_labels(field=f, source=s), n) for (f, s), n in sorted(self.fields.items())],
            )
            hist = [(_labels(le=_num(le)), n) for le, n in zip(LATENCY_BUCKETS, self.buckets)]
            metric(
                "decision_latency_seconds",
                "histogram",
                "Time the harness took per decision.",
                [],
            )
            out.extend("decisionsmith_decision_latency_seconds_bucket%s %d" % (lab, n) for lab, n in hist)
            out.append('decisionsmith_decision_latency_seconds_bucket{le="+Inf"} %d' % self.latency_count)
            out.append("decisionsmith_decision_latency_seconds_sum %s" % _num(self.latency_sum))
            out.append("decisionsmith_decision_latency_seconds_count %d" % self.latency_count)
            t = self.teacher
            metric("teacher_calls_total", "counter", "Calls to the teacher LLM.", [("", t["calls"])])
            metric("teacher_errors_total", "counter", "Teacher calls that failed.", [("", t["errors"])])
            metric(
                "teacher_tokens_total",
                "counter",
                "Tokens the teacher reported, by kind.",
                [(_labels(kind="input"), t["input_tokens"]), (_labels(kind="output"), t["output_tokens"])],
            )
            metric(
                "teacher_cost_usd_total",
                "counter",
                "Teacher cost in USD, from calls whose cost is known (see teacher_unpriced_calls_total).",
                [("", self.cost)],
            )
            metric(
                "teacher_unpriced_calls_total",
                "counter",
                "Successful teacher calls whose cost is unknown (the provider reported no cost).",
                [("", t["unpriced"])],
            )
        return "\n".join(out) + "\n"


def _labels(**labels: str) -> str:
    esc = {k: v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") for k, v in labels.items()}
    return "{%s}" % ",".join('%s="%s"' % kv for kv in esc.items())


def _num(v: float) -> str:
    return repr(float(v)) if isinstance(v, float) else str(v)


def _tokens(usage: dict[str, Any], kind: str) -> int:
    v = usage.get(kind)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


class _CountedTeacher:
    """Wraps the harness's teacher to count its calls, tokens and cost for `/metrics` and `usage`."""

    def __init__(self, engine: Any, metrics: _Metrics) -> None:
        self.engine, self.metrics, self.name = engine, metrics, engine.name

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def ask(self, text: str, questions: dict[str, dict[str, Any]]) -> Any:
        return self._record(lambda: self.engine.ask(text, questions))

    async def aask(self, text: str, questions: dict[str, dict[str, Any]]) -> Any:
        aask = getattr(self.engine, "aask", None)
        if not callable(aask):
            return await asyncio.to_thread(self.ask, text, questions)
        try:
            raw = await aask(text, questions)
        except Exception:
            self.metrics.teacher_call(None, None)
            raise
        self._count(raw)
        return raw

    def _record(self, call: Callable[[], Any]) -> Any:
        try:
            raw = call()
        except Exception:
            self.metrics.teacher_call(None, None)
            raise
        self._count(raw)
        return raw

    def _count(self, raw: Any) -> None:
        from .benchmark import _cost

        usage = raw.get("usage") if isinstance(raw, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        self.metrics.teacher_call(usage, _cost(self.engine, usage))
        mine = _usage.get()
        if mine is not None:
            for kind in ("input_tokens", "output_tokens"):
                mine[kind] += _tokens(usage, kind)


def _harness(
    target: Any, teacher: Any, mode: Any, log: Any, collect: float | None, device: str | None
) -> tuple[Harness[Any], str, bool]:
    from .predictor import Model, load

    if isinstance(target, Harness):
        given = {"teacher": teacher, "mode": mode, "collect": collect, "device": device}
        extra = [k for k, v in given.items() if v is not None] + (["log"] if log != "decisions.db" else [])
        if extra:
            raise ValueError(
                "app(harness) takes its settings from the harness; drop %s, or pass a model path instead"
                % ", ".join(extra)
            )
        return target, target.schema.name, False
    model = target if isinstance(target, Model) else load(target, device=device)
    name = str((model.meta or {}).get("name") or model.schema.name)
    h: Harness[Any] = harness(model, teacher=teacher, mode=mode, log=log, collect=1.0 if collect is None else collect)
    return h, name, True


def _answer(schema: Any, name: str, dist: dict[str, float]) -> dict[str, Any]:
    """One field's probabilities as a Jev `/v1/systemone` answer (`choice`, `score` or `noul`)."""
    f = schema.fields[name]
    conf = confidence(dist)
    if f.kind == "bool":
        return {"type": "noul", "noul": dist["true"], "confidence": conf}
    if f.kind == "choice":
        return {"type": "choice", "choice": top(dist), "probabilities": dict(dist), "confidence": conf}
    probs = [dist[label] for label in f.labels]
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs)),
        "legend": {str(i): c for i, c in enumerate(f.question["criteria"])},
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "confidence": conf,
    }


def _check_questions(schema: Any, questions: dict[str, dict[str, Any]] | None) -> list[str]:
    if not questions:
        return list(schema.fields)
    unknown = [q for q in questions if q not in schema.fields]
    if unknown:
        raise _Problem(
            422,
            "invalid",
            "this server answers the questions %s; unknown: %s" % (list(schema.fields), unknown),
            "use the saved model's field names as question ids (GET /health lists them), or leave questions out",
        )
    for qid, q in questions.items():
        ours = schema.fields[qid].question
        if "type" in q and q["type"] != ours["type"]:
            raise _Problem(
                422,
                "invalid",
                "question %r is type %r here, not %r" % (qid, ours["type"], q["type"]),
                'send "type": %r or leave type out' % ours["type"],
            )
        if q.get("criteria") and sorted(option_keys({**ours, **q})) != sorted(option_keys(ours)):
            raise _Problem(
                422,
                "invalid",
                "question %r has options %s here, not %s" % (qid, option_keys(ours), option_keys({**ours, **q})),
                "send the saved model's options, or leave criteria out",
            )
    return list(questions)


def _check_text(text: str, what: str) -> None:
    if len(text) > MAX_TEXT_CHARS:
        raise _Problem(
            413,
            "too_large",
            "%s is %d characters; the limit is %d" % (what, len(text), MAX_TEXT_CHARS),
            "send a shorter text (split long documents)",
        )
    if not text.strip():
        raise _Problem(422, "invalid", "%s is empty" % what, "send a non-empty text")


def app(
    target: Any,
    *,
    teacher: Any = None,
    mode: Mode | dict[str, Mode] | None = None,
    log: str | os.PathLike[str] | None = "decisions.db",
    collect: float | None = None,
    device: str | None = None,
    api_key: str | None = None,
) -> Any:
    """An ASGI app (FastAPI) serving a saved model folder, a `ds.model`, or a ready `ds.harness(...)`.

    Endpoints: `POST /v1/decide`, `POST /v1/systemone` (Jev-compatible), `POST /v1/label`, `GET /v1/status`,
    `GET /health`, `GET /metrics` (Prometheus) and `/docs`. With an `api_key` (default: `DECISIONSMITH_API_KEY` from
    the environment), every endpoint but `/health` and the docs needs `Authorization: Bearer <key>`.
    """
    try:
        from fastapi import Depends, FastAPI, Header, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse, PlainTextResponse
        from starlette.exceptions import HTTPException
    except ImportError:
        raise ImportError("decisionsmith.serve needs FastAPI: %s" % INSTALL) from None
    from . import __version__

    h, name, owned = _harness(target, teacher, mode, log, collect, device)
    metrics = _Metrics()
    if h.teacher is not None:
        inner = h.teacher.engine if isinstance(h.teacher, _CountedTeacher) else h.teacher
        h.teacher = _CountedTeacher(inner, metrics)
    api_key = api_key or os.environ.get(API_KEY_ENV) or None
    expected = ("Bearer " + api_key).encode("utf-8", "surrogateescape") if api_key else b""

    @asynccontextmanager
    async def lifespan(_: Any) -> Any:
        try:
            yield
        finally:
            if owned:
                h.close()

    api = FastAPI(
        title="decisionsmith: %s" % name,
        summary="Typed decisions from a System One model, with an LLM behind it for unsure cases.",
        version=__version__,
        lifespan=lifespan,
    )

    def auth(authorization: str | None = Header(default=None, include_in_schema=False)) -> None:
        if api_key is None:
            return
        given = (authorization or "").encode("utf-8", "surrogateescape")
        if not hmac.compare_digest(given, expected):
            raise _Problem(
                401,
                "unauthorized",
                "missing or wrong API key",
                "send the header Authorization: Bearer <the server's %s>" % API_KEY_ENV,
            )

    async def call(fn: Callable[[], Awaitable[Any]]) -> Any:
        try:
            return await fn()
        except EngineError as e:
            raise _Problem(503, "engine", e.message, e.fix or "check the engines: decisionsmith doctor") from None
        except KeyError:
            raise _Problem(
                404, "not_found", "no decision with that id in the log", "use an id a decision returned"
            ) from None
        except ValueError as e:
            fix = "start the server with a log (--log decisions.db)" if "log=None" in str(e) else ""
            raise _Problem(409 if fix else 422, "no_log" if fix else "invalid", str(e), fix) from None
        except Exception as e:
            logger.exception("request failed")
            raise _Problem(500, "error", "%s inside the harness" % type(e).__name__, "see the server log") from None

    async def decide(text: str) -> Any:
        r = await call(lambda: h.adecide(text))
        metrics.decision(r.source, r.sure, r.latency_ms)
        return r

    @api.exception_handler(_Problem)
    async def _problem(_: Request, e: _Problem) -> Any:
        headers = {"WWW-Authenticate": "Bearer"} if e.status == 401 else None
        return JSONResponse(e.body(), status_code=e.status, headers=headers)

    @api.exception_handler(RequestValidationError)
    async def _invalid(request: Request, e: RequestValidationError) -> Any:
        where = "; ".join(
            "%s: %s" % (".".join(str(p) for p in err.get("loc", ()) if p != "body") or "body", err.get("msg"))
            for err in e.errors()
        )
        return JSONResponse(
            _Problem(422, "invalid", where, "see the request examples at /docs").body(), status_code=422
        )

    @api.exception_handler(HTTPException)
    async def _http(_: Request, e: HTTPException) -> Any:
        code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http")
        body = _Problem(e.status_code, code, str(e.detail), "see the endpoints at /docs").body()
        return JSONResponse(body, status_code=e.status_code)

    @api.middleware("http")
    async def _count(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        route = request.scope.get("route")
        metrics.request(getattr(route, "path", "other"), response.status_code)
        return response

    @api.get("/health", summary="Liveness and what this server answers")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": name,
            "fields": {n: list(f.labels) for n, f in h.schema.fields.items()},
            "modes": dict(h.modes),
            "version": __version__,
        }

    @api.post("/v1/decide", response_model=DecideResponse, dependencies=[Depends(auth)], summary="Decide one text")
    async def decide_route(body: DecideRequest) -> Any:
        _check_text(body.text, "text")
        r = await decide(body.text)
        value = r.value.model_dump(mode="json")
        return DecideResponse(
            id=r.id,
            value={n: value[n] for n in r.source},
            source=r.source,
            confidence=r.confidence,
            sure=r.sure,
            latency_ms=r.latency_ms,
        )

    @api.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        dependencies=[Depends(auth)],
        summary="Jev-compatible decision (state + questions in, answers + usage out)",
    )
    async def systemone(body: SystemOneRequest) -> Any:
        _check_text(body.state, "state")
        ids = _check_questions(h.schema, body.questions)
        token = _usage.set({"input_tokens": 0, "output_tokens": 0})
        try:
            r = await decide(body.state)
            usage = dict(_usage.get() or {})
        finally:
            _usage.reset(token)
        return SystemOneResponse(
            model=name,
            answers={q: _answer(h.schema, q, r.probabilities[q]) for q in ids},
            usage=usage,
            routing={"id": r.id, "source": {q: r.source[q] for q in ids}, "sure": r.sure, "latency_ms": r.latency_ms},
        )

    @api.post("/v1/label", dependencies=[Depends(auth)], summary="Record the right answer for a decision id")
    async def label(body: LabelRequest) -> dict[str, Any]:
        unknown = sorted(set(body.labels) - set(h.schema.fields))
        if not body.labels or unknown:
            raise _Problem(
                422,
                "invalid",
                "unknown fields %s" % unknown if unknown else "labels is empty",
                'send {"id": "...", "labels": {"field": "value"}} with fields from %s' % list(h.schema.fields),
            )

        async def run() -> None:
            await asyncio.to_thread(h.label, body.id, **body.labels)

        await call(run)
        return {"id": body.id, "labelled": sorted(body.labels)}

    @api.get("/v1/status", dependencies=[Depends(auth)], summary="Per field: agreement, sure rate, advice")
    async def status() -> dict[str, Any]:
        async def run() -> Any:
            return await asyncio.to_thread(h.status)

        s = await call(run)
        return {"model": name, "modes": dict(h.modes), "teacher": h.teacher.name if h.teacher else None, **s.to_dict()}

    @api.get("/metrics", dependencies=[Depends(auth)], response_class=PlainTextResponse, summary="Prometheus metrics")
    async def prometheus() -> Any:
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    api.add_middleware(_BodyLimit, limit=MAX_BODY_BYTES)
    return api


class _BodyLimit:
    """ASGI wrapper: reads at most `limit` bytes of a request body (chunked or not), else answers 413."""

    def __init__(self, app: Any, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.limit:
            await self._too_large(send)
            return
        chunks: list[bytes] = []
        size = 0
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                return
            chunks.append(message.get("body", b""))
            size += len(chunks[-1])
            if size > self.limit:
                await self._too_large(send)
                return
            more = message.get("more_body", False)
        body = b"".join(chunks)
        sent = False

        async def replay() -> dict[str, Any]:
            nonlocal sent
            if sent:
                return dict(await receive())
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    async def _too_large(self, send: Any) -> None:
        payload = json.dumps(
            _Problem(
                413,
                "too_large",
                "request body is over %d bytes" % self.limit,
                "send one text per request, under %d characters" % MAX_TEXT_CHARS,
            ).body()
        ).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": payload})
