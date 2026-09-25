"""POST with retries (timeouts, 408/429/5xx, exponential backoff), sync and async, on httpx."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .base import EngineError

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def delay(attempt: int, response: Any = None) -> float:
    after = response.headers.get("retry-after") if response is not None else None
    try:
        return min(float(after), 20.0) if after else 0.5 * 2**attempt
    except ValueError:
        return 0.5 * 2**attempt


def _gave_up(name: str, retries: int, last: str) -> EngineError:
    return EngineError(name, "no response after %d attempts (%s)" % (retries + 1, last), "check the URL and network")


def post(client: Any, url: str, body: dict[str, Any], name: str, retries: int) -> Any:
    import httpx

    last = ""
    for attempt in range(retries + 1):
        try:
            r = client.post(url, json=body)
        except httpx.HTTPError as e:
            last = "%s: %s" % (type(e).__name__, e)
            if attempt < retries:
                time.sleep(delay(attempt))
            continue
        if r.status_code in RETRY_STATUS and attempt < retries:
            last = "HTTP %d" % r.status_code
            time.sleep(delay(attempt, r))
            continue
        return r
    raise _gave_up(name, retries, last)


async def apost(client: Any, url: str, body: dict[str, Any], name: str, retries: int) -> Any:
    import httpx

    last = ""
    for attempt in range(retries + 1):
        try:
            r = await client.post(url, json=body)
        except httpx.HTTPError as e:
            last = "%s: %s" % (type(e).__name__, e)
            if attempt < retries:
                await asyncio.sleep(delay(attempt))
            continue
        if r.status_code in RETRY_STATUS and attempt < retries:
            last = "HTTP %d" % r.status_code
            await asyncio.sleep(delay(attempt, r))
            continue
        return r
    raise _gave_up(name, retries, last)


class Clients:
    """One sync httpx client, and one async client per running event loop."""

    def __init__(self, timeout: float, headers: dict[str, str]) -> None:
        self.timeout, self.headers = timeout, headers
        self._sync: Any = None
        self._async: tuple[Any, Any] | None = None

    def sync(self) -> Any:
        import httpx

        if self._sync is None:
            self._sync = httpx.Client(timeout=self.timeout, headers=self.headers)
        return self._sync

    def async_(self) -> Any:
        import httpx

        loop = asyncio.get_running_loop()
        if self._async is None or self._async[0] is not loop:
            self._async = (loop, httpx.AsyncClient(timeout=self.timeout, headers=self.headers))
        return self._async[1]
