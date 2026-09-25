"""Any Jev-compatible `/v1/systemone` endpoint over HTTP, and TypeSafe's hosted Jev."""

from __future__ import annotations

import os
from typing import Any

from .. import offline
from .base import EngineError, response
from .http import Clients, apost, post

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"


class SystemOneEngine:
    """Any Jev-compatible `/v1/systemone` endpoint: laya-serve, laya.cpp, `decisionsmith serve`, or Jev itself."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        retries: int = 2,
        name: str | None = None,
    ) -> None:
        base = url.rstrip("/")
        self.url = base if base.endswith("/v1/systemone") else base + "/v1/systemone"
        self.model = model
        self.retries = retries
        self.name = name or "systemone:%s" % url
        self._clients = Clients(timeout, {"Authorization": "Bearer " + api_key} if api_key else {})

    def _body(self, text: str, questions: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"state": text, "questions": questions}
        if self.model:
            body["model"] = self.model
        return body

    def ask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if offline.enabled():
            return offline.answer(text, questions, self.name)
        r = post(self._clients.sync(), self.url, self._body(text, questions), self.name, self.retries)
        return self._read(r, questions)

    async def aask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if offline.enabled():
            return offline.answer(text, questions, self.name)
        r = await apost(self._clients.async_(), self.url, self._body(text, questions), self.name, self.retries)
        return self._read(r, questions)

    def _read(self, r: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if r.status_code in (401, 403):
            raise EngineError(self.name, "authentication failed (HTTP %d)" % r.status_code, self._auth_fix())
        if r.status_code >= 400:
            raise EngineError(self.name, "HTTP %d: %s" % (r.status_code, r.text[:200]))
        try:
            data = r.json()
        except ValueError:
            raise EngineError(self.name, "response is not JSON: %r" % r.text[:120]) from None
        response(self, data, questions)
        return dict(data)

    def _auth_fix(self) -> str:
        return "set SYSTEMONE_API_KEY to the server's bearer token"


class JevEngine(SystemOneEngine):
    """TypeSafe's hosted Jev. Key from `TYPESAFE_API_KEY`. Jev can be a teacher or a student, not fine-tuned."""

    def __init__(self, model: str = JEV_MODEL, *, api_key: str | None = None, timeout: float = 60.0) -> None:
        key = api_key or os.environ.get("TYPESAFE_API_KEY") or ("offline" if offline.enabled() else None)
        name = "jev" if model == JEV_MODEL else "jev:%s" % model
        if not key:
            raise EngineError(name, "no API key", "export TYPESAFE_API_KEY=... (get one from TypeSafe)")
        super().__init__(JEV_URL, api_key=key, model=model, timeout=timeout, name=name)

    def _auth_fix(self) -> str:
        return "check TYPESAFE_API_KEY"
