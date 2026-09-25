"""OpenAI SDK (`uv add "decisionsmith[openai]"`): any `openai` client as the teacher, and `wrap(client, x)`.

ds.harness(Ticket, teacher=teacher(OpenAI(base_url=..., api_key=...), "llama-3.3-70b-versatile"))
client = wrap(OpenAI(), ds.model(Ticket))   # parse(response_format=Ticket): the model answers when it is sure
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from .. import offline
from ..engines.structured import TextEngine
from ._base import resolve, run_sync
from .litellm import last_text


def _read(response: Any) -> tuple[str, dict[str, Any]]:
    u = getattr(response, "usage", None)
    usage = {"input_tokens": getattr(u, "prompt_tokens", None), "output_tokens": getattr(u, "completion_tokens", None)}
    return response.choices[0].message.content or "", usage


def teacher(client: Any, model: str, **options: Any) -> TextEngine:
    """An `openai.OpenAI` / `AsyncOpenAI` client (any `base_url`) and a model id as a teacher; `options` go to
    `chat.completions.create`."""
    import openai

    create = client.chat.completions.create

    def kwargs(system: str, user: str) -> dict[str, Any]:
        return {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

    name = "openai-sdk:%s" % model
    if isinstance(client, openai.AsyncOpenAI):

        async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
            return _read(await create(**kwargs(system, user), **options))

        return TextEngine(name, lambda s, u: run_sync(lambda: acomplete(s, u)), acomplete)
    return TextEngine(name, lambda s, u: _read(create(**kwargs(s, u), **options)))


def as_harness(x: Any) -> Any:
    import decisionsmith as ds

    resolve(x)
    return ds.harness(x, log=None) if isinstance(x, ds.Model) else x


def _names(fmt: Any) -> tuple[list[str], dict[str, Any]] | None:
    if isinstance(fmt, type) and issubclass(fmt, BaseModel):
        return list(fmt.model_fields), {}
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        props = (fmt.get("json_schema") or {}).get("schema", {}).get("properties") or {}
        return list(props), props
    return None


def fits(h: Any, fmt: Any) -> bool:
    """Can the harness answer this `response_format` (a Pydantic class or a `json_schema` dict) on its own?"""
    names = (_names(fmt) or ([], {}))[0]
    return len(names) == 1 if h.simple else bool(names) and set(names) <= set(h.schema.fields)


def fill(h: Any, fmt: Any, result: Any) -> tuple[str, Any] | None:
    """The reply `(json, parsed)` for a fitting `response_format` from a harness `Result`, if the harness is sure
    (or offline: `DS_OFFLINE=1` never calls the client) and the value is allowed by the format."""
    if not (result.sure or offline.enabled()):
        return None
    names, props = _names(fmt) or ([], {})
    value = result.value.model_dump(mode="json")
    data = {names[0]: value["label"]} if h.simple else {n: value[n] for n in names}
    if isinstance(fmt, dict):
        if any(data[n] not in props[n].get("enum", [data[n]]) for n in names):
            return None
        return json.dumps(data), None
    try:
        parsed = fmt.model_validate(data)
    except ValidationError:
        return None
    return parsed.model_dump_json(), parsed


def _completion(kw: dict[str, Any], hit: tuple[str, Any], name: str) -> Any:
    from openai.types.chat import ChatCompletion, ParsedChatCompletion

    message: dict[str, Any] = {"role": "assistant", "content": hit[0]}
    cls: Any = ChatCompletion
    if name == "parse":
        message["parsed"] = hit[1]
        cls = ParsedChatCompletion[Any if hit[1] is None else type(hit[1])]
    choice = {"index": 0, "finish_reason": "stop", "message": message}
    data = {"id": "decisionsmith", "object": "chat.completion", "created": int(time.time()), "model": kw.get("model")}
    return cls.model_validate({**data, "choices": [choice]})


class Proxy:
    def __init__(self, real: Any, **overrides: Any) -> None:
        self._real, self._overrides = real, overrides

    def __getattr__(self, name: str) -> Any:
        return self._overrides[name] if name in self._overrides else getattr(self._real, name)


def wrap(client: Any, x: Any) -> Any:
    """`client` with `chat.completions.create` / `.parse` answered by `x` (a `ds.model` or `ds.harness`) when the
    request's `response_format` only asks for its fields and it is sure; otherwise the real call is made.
    Replies from decisionsmith have `id == "decisionsmith"`."""
    import openai

    h, real = as_harness(x), client.chat.completions
    is_async = isinstance(client, openai.AsyncOpenAI)

    def method(name: str) -> Any:
        def text(kw: dict[str, Any]) -> str:
            return last_text(kw.get("messages")) if fits(h, kw.get("response_format")) else ""

        def sync(**kw: Any) -> Any:
            t = text(kw)
            hit = fill(h, kw["response_format"], h.decide(t)) if t.strip() else None
            return _completion(kw, hit, name) if hit else getattr(real, name)(**kw)

        async def async_(**kw: Any) -> Any:
            t = text(kw)
            hit = fill(h, kw["response_format"], await h.adecide(t)) if t.strip() else None
            return _completion(kw, hit, name) if hit else await getattr(real, name)(**kw)

        return async_ if is_async else sync

    completions = Proxy(real, create=method("create"), parse=method("parse"))
    return Proxy(client, chat=Proxy(client.chat, completions=completions))
