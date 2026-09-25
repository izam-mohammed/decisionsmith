# Serve a model over HTTP

`decisionsmith serve models/<name>-vN` puts a saved model (and, if you like, an LLM for the unsure cases) behind a
small HTTP server, so any app in any language can ask it for decisions.

**Flow:** production flow. It loads what step 6 of the developer flow saved, logs every decision, and collects real
samples for the next [golden dataset](golden.md).

```bash
uv add "decisionsmith[serve,laya]"
uv run decisionsmith serve models/ticket-v2 --teacher claude-haiku-4-5 --collect 0.1
# Uvicorn listens on http://127.0.0.1:8000; the API docs are at http://127.0.0.1:8000/docs
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{"text": "I was charged twice"}'
# {"id": "3f2a...", "value": {"label": "billing"}, "source": {"label": "student"},
#  "confidence": {"label": 0.97}, "sure": true, "latency_ms": 18.4}
```

The numbers in that output are an illustration of the shape, not a measurement.

## The same thing from Python

`decisionsmith.serve.app(...)` returns the server as an ASGI app (FastAPI). Give it a saved model folder, a
`ds.model`, or a ready `ds.harness(...)`. `TestClient` calls it in-process, with no port, which is handy in tests.

```python
from fastapi.testclient import TestClient

import decisionsmith as ds
from decisionsmith.serve import app

model = ds.model(["billing", "technical", "sales"])
model.train("tickets.csv")
path = model.save("models/ticket")  # models/ticket-v1

api = app(path, teacher="claude-haiku-4-5", collect=0.1)
with TestClient(api) as client:
    r = client.post("/v1/decide", json={"text": "I was charged twice"}).json()
    print(r["value"], r["source"], r["sure"])
    client.post("/v1/label", json={"id": r["id"], "labels": {"label": "billing"}})  # a human correction
    print(client.get("/v1/status").json()["fields"]["label"]["advice"])
```

To run it for real from your own file, point Uvicorn at it:

```bash
# app.py:  from decisionsmith.serve import app;  api = app("models/ticket-v1", teacher="claude-haiku-4-5")
uv run uvicorn app:api --host 127.0.0.1 --port 8000
```

## Endpoints

| endpoint | what |
|---|---|
| `POST /v1/decide` | `{"text": "..."}` in; `id`, `value` per field, `source` per field (`student` or `teacher`), `confidence` per field, `sure`, `latency_ms` out |
| `POST /v1/systemone` | the Jev wire format: `{"state": "...", "questions": {...}}` in; `{"model", "answers", "usage", "routing"}` out (see below) |
| `POST /v1/label` | `{"id": "<decision id>", "labels": {"field": "right value"}}`: record a human correction for `status()`, `golden` and training |
| `GET /v1/status` | per field: agreement with the teacher, sure rate, accuracy when sure, and the next step (`h.status()`) |
| `GET /metrics` | Prometheus text format (below) |
| `GET /health` | `{"status": "ok", "model", "fields", "modes", "version"}`; never needs the API key |
| `GET /docs`, `GET /openapi.json` | interactive API docs and the OpenAPI schema |

## Jev-compatible `/v1/systemone`

The request and response follow the `/v1/systemone` format that TypeSafe's Jev and Laya's own `laya-serve` speak,
so a client written for them can switch by changing its base URL.

- Request: `{"state": "<text>", "questions": {"<field>": {"type": ..., "criteria": ...}}, "model": "..."}`.
  Question ids must be the saved model's field names (`label` for a labels model; `GET /health` lists them). Leave
  `questions` out to get every field. `type` and `criteria`, if you send them, must match the saved model; the
  instructions text is ignored, because the model answers the question it was trained on. `model` is ignored.
- Response: `answers` per question in the Jev shape (`choice` + `probabilities` + `confidence`; `score` +
  `probabilities` + `legend` + `confidence`; `noul` + `confidence`), `usage` (`input_tokens`, `output_tokens`:
  what the teacher LLM used for this request, 0 when the local model answered alone), `model` (the saved folder's
  name) and `routing` (`id`, `source` per field, `sure`, `latency_ms`; Jev clients ignore it).

What is tested: decisionsmith's own Jev client (`SystemOneEngine`, the code behind `"systemone:<url>"`) decodes
every answer, and each answer carries exactly the keys Laya's `/v1/systemone` answer carries (Laya's extras
`answer_confidence` and `action` aside). typesafe-sdk and other Jev clients have not been run against it yet.

A decisionsmith harness elsewhere can use this server as its student:

<!-- no-test: needs a running server on port 8000 -->
```python
import decisionsmith as ds

remote = ds.model(["billing", "technical", "sales"], "systemone:http://localhost:8000")
h = ds.harness(remote, teacher="claude-haiku-4-5")
```

(`SYSTEMONE_API_KEY` is sent as the bearer token if the server has an API key.)

Plain Python with `httpx`:

<!-- no-test: needs a running server on port 8000 -->
```python
import httpx

r = httpx.post("http://localhost:8000/v1/systemone", json={"state": "I was charged twice", "questions": {"label": {}}})
print(r.json()["answers"]["label"]["choice"])
```

## Options

| option (CLI / Python) | default | change it when |
|---|---|---|
| `--teacher` / `teacher=` | none: the model decides alone | you want an LLM to answer the cases the model is unsure of (cascade), or to measure it (shadow) |
| `--mode` / `mode=` | `cascade` with a teacher, `student` without | `shadow` while you are still measuring; per field: `--mode team=cascade,wants_refund=student` |
| `--collect` / `collect=` | `1.0` (keep every text in the log) | real users: `0.1` keeps a 10% share plus every unsure or disputed text; `0` keeps none ([collect](collect.md)) |
| `--log` / `log=` | `decisions.db` | another path; `--log none` (`log=None`) logs nothing, which also turns off `/v1/label` and `/v1/status` |
| `--host`, `--port` | `127.0.0.1`, `8000` | `--host 0.0.0.0` only inside a container or behind a reverse proxy |

`app(harness)` takes every setting from the harness you pass; giving `teacher=` or the others too is an error.

## Auth

Set `DECISIONSMITH_API_KEY` before starting the server (or pass `app(..., api_key=...)` in Python), and every endpoint except `/health` and the API docs needs
`Authorization: Bearer <key>` (compared in constant time). Without it the server is open, which is fine on
`127.0.0.1` and not fine on a public address.

```bash
export DECISIONSMITH_API_KEY=$(openssl rand -hex 32)
uv run decisionsmith serve models/ticket-v2
curl -s localhost:8000/v1/decide -H "authorization: Bearer $DECISIONSMITH_API_KEY" \
  -H 'content-type: application/json' -d '{"text": "the app crashes on login"}'
```

## Metrics

`GET /metrics` (Prometheus text format; needs the API key if one is set). Counts start at zero when the server
starts and are per process.

| metric | what |
|---|---|
| `decisionsmith_http_requests_total{path, status}` | requests by endpoint and status code |
| `decisionsmith_decisions_total{sure}` | decisions, by whether every field was sure (the sure rate is `sure="true"` over the total) |
| `decisionsmith_field_decisions_total{field, source}` | answers per field from the `student` or the `teacher` |
| `decisionsmith_decision_latency_seconds` | histogram of the time the harness took per decision |
| `decisionsmith_teacher_calls_total`, `decisionsmith_teacher_errors_total` | teacher LLM calls, and how many failed |
| `decisionsmith_teacher_tokens_total{kind}` | `input` and `output` tokens the teacher reported |
| `decisionsmith_teacher_cost_usd_total` | teacher cost from calls whose cost is known (a provider that reports it, or Jev's list price) |
| `decisionsmith_teacher_unpriced_calls_total` | teacher calls whose cost is unknown; most LLM providers don't report cost, so check your provider's bill |

No text is ever put in a metric or in the server's own log lines. Text is stored only in the decision log, and only
for the rows `collect` keeps.

## Limits and errors

A request body is at most 1 MiB and a text at most 50,000 characters. Every error is JSON:
`{"error": {"code": "...", "message": "...", "fix": "..."}}`.

| status | code | what happened | fix |
|---|---|---|---|
| 401 | `unauthorized` | missing or wrong API key | send `Authorization: Bearer <key>` |
| 404 | `not_found` | `/v1/label` with an id the log doesn't have, or an unknown path | use the `id` a decision returned; see `/docs` |
| 409 | `no_log` | `/v1/label` or `/v1/status` on a server started with `--log none` | start it with a log |
| 413 | `too_large` | body over 1 MiB or text over 50,000 characters | send one shorter text per request |
| 422 | `invalid` | bad JSON, a missing field, an empty text, a question id or option the model doesn't have | the message names the problem |
| 503 | `engine` | no engine could answer a field (the teacher failed and there is no student answer to fall back to) | `decisionsmith doctor --engines <teacher>` |
| 500 | `error` | a bug; the server log has the traceback | please open an issue |

If the teacher fails (a timeout, a rate limit, a bad reply) and the model has an answer, the server returns the
model's answer with `sure: false` instead of an error. The teacher's failure shows in
`decisionsmith_teacher_errors_total`.

## Deploy notes

- Requests run concurrently: the model runs in a worker thread and LLM calls are async, so a slow LLM call doesn't
  hold up other requests or `/health`. CPU inference still shares the machine's cores.
- The model loads on the first request, so that one is slower; send one warm-up request after a start (a
  `/health` call doesn't load it).
- One process per decision log. `uvicorn --workers N` starts N copies of the model (N times the memory) that share
  the SQLite log file; metrics are then per worker.
- Keep the server on `127.0.0.1` behind your reverse proxy (TLS, rate limits), or bind `0.0.0.0` inside a container.
  Set `DECISIONSMITH_API_KEY` whenever anything but your own machine can reach it.
- Keys for the teacher come from the usual variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...). No key is
  written to the log or to `/health`.
- A new version: save `models/ticket-v3`, start a server on it next to the old one, move traffic, stop the old one.
  The log can stay the same file.

## Next

- [Collect real samples](collect.md): what `--collect` keeps, retention and `h.forget`.
- [Golden dataset](golden.md): turn the served log into labelled data (`decisionsmith golden --log decisions.db`).
- Runnable examples: [examples/serve/](../examples/serve/).

Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai Innovations.
decisionsmith is an independent project, not affiliated with TypeSafe AI or Convai Innovations.
