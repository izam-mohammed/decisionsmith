"""Any Jev-compatible `/v1/systemone` endpoint over HTTP, and TypeSafe's hosted Jev."""

from __future__ import annotations

import os
import time
from typing import Any

from .base import EngineError, response

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class SystemOneEngine:
    """Any Jev-compatible `/v1/systemone` endpoint: laya-serve, laya.cpp, or Jev itself."""

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
        import httpx

        base = url.rstrip("/")
        self.url = base if base.endswith("/v1/systemone") else base + "/v1/systemone"
        self.model = model
        self.retries = retries
        self.name = name or "systemone:%s" % url
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def _sleep(self, attempt: int) -> None:
        time.sleep(0.5 * 2**attempt)

    def ask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        import httpx

        body: dict[str, Any] = {"state": text, "questions": questions}
        if self.model:
            body["model"] = self.model
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(self.url, json=body)
            except httpx.HTTPError as e:
                last = "%s: %s" % (type(e).__name__, e)
                if attempt < self.retries:
                    self._sleep(attempt)
                continue
            if r.status_code in _RETRY_STATUS and attempt < self.retries:
                last = "HTTP %d" % r.status_code
                self._sleep(attempt)
                continue
            if r.status_code in (401, 403):
                raise EngineError(self.name, "authentication failed (HTTP %d)" % r.status_code, self._auth_fix())
            if r.status_code >= 400:
                raise EngineError(self.name, "HTTP %d: %s" % (r.status_code, r.text[:200]))
            try:
                data = r.json()
            except ValueError:
                raise EngineError(self.name, "response is not JSON: %r" % r.text[:120]) from None
            response(self, data, questions)
            return data
        raise EngineError(
            self.name,
            "no response after %d attempts (%s)" % (self.retries + 1, last),
            "check the URL and network",
        )

    def _auth_fix(self) -> str:
        return "set SYSTEMONE_API_KEY to the server's bearer token"


class JevEngine(SystemOneEngine):
    """TypeSafe's hosted Jev. Key from `TYPESAFE_API_KEY`. Jev can be a teacher or a student, not fine-tuned."""

    def __init__(self, model: str = JEV_MODEL, *, api_key: str | None = None, timeout: float = 60.0) -> None:
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        name = "jev" if model == JEV_MODEL else "jev:%s" % model
        if not key:
            raise EngineError(name, "no API key", "export TYPESAFE_API_KEY=... (get one from TypeSafe)")
        super().__init__(JEV_URL, api_key=key, model=model, timeout=timeout, name=name)

    def _auth_fix(self) -> str:
        return "check TYPESAFE_API_KEY"
