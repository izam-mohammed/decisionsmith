"""Any LLM as a teacher, with no LLM library: a built-in OpenAI-compatible client on httpx, and Claude through the
official `anthropic` SDK (`decisionsmith[anthropic]`). Answers are one-hot. LiteLLM lives in `integrations.litellm`.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, ValidationError

from .. import offline
from .base import EngineError, need
from .http import Clients, apost, post
from .structured import (
    FIX,
    SYSTEM,
    WRITER,
    answers_model,
    checked,
    fake_reply,
    json_schema,
    one_hot,
    parse,
    prompt,
    retry_prompt,
    texts_model,
    write_prompt,
)

PROVIDERS: dict[str, tuple[str, str | None]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.ai/v1", "TOGETHER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}
_OPENAI_NAMES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_FORMATS = ("json_schema", "json_object", None)
_FORMAT_WORDS = ("response_format", "json_schema", "json_object", "structured output")
Flow = Generator[dict[str, Any], Any, tuple[BaseModel, dict[str, Any]]]


def resolve(model: str, url: str | None) -> tuple[str, str, str | None, str | None]:
    """`model` -> (provider, model id to send, base URL, key env var)."""
    head, sep, rest = model.partition("/")
    if url:
        return "custom", rest if head == "openai" and sep else model, url.rstrip("/"), None
    if model.startswith("claude-") or (head == "anthropic" and rest):
        return "anthropic", rest if head == "anthropic" else model, None, "ANTHROPIC_API_KEY"
    if sep and head in PROVIDERS and rest:
        return (head, rest, *PROVIDERS[head])
    if model.startswith(_OPENAI_NAMES):
        return ("openai", model, *PROVIDERS["openai"])
    if model.startswith("gemini-"):
        return ("gemini", model, *PROVIDERS["gemini"])
    raise ValueError(
        "don't know where %r runs; use a provider prefix (%s/<model>), ds.LLM(%r, url=...) for any "
        "OpenAI-compatible server, or 'litellm/<model>'" % (model, "/".join(PROVIDERS), model)
    )


class LLMEngine:
    """Any LLM as a teacher (or engine). No LLM library needed.

        ds.LLM("claude-haiku-4-5")                     # Claude: pip install 'decisionsmith[anthropic]'
        ds.LLM("gpt-5-mini") · ds.LLM("gemini-2.5-flash") · ds.LLM("groq/llama-3.3-70b-versatile")
        ds.LLM("ollama/qwen3")                                             # local, no key
        ds.LLM("my-model", url="http://localhost:8000/v1", api_key="...")  # any OpenAI-compatible server
        ds.LLM("litellm/bedrock/...")                                      # anything LiteLLM supports
        ds.LLM("gpt-5", max_tokens=200)                                    # extra request options

    Keys come from the provider's environment variable (OPENAI_API_KEY, GEMINI_API_KEY, ...). Answers are one-hot.
    """

    def __new__(cls, model: str, **kwargs: Any) -> Any:
        if isinstance(model, str) and model.startswith("litellm/"):
            from ..integrations.litellm import teacher

            return teacher(model[len("litellm/") :], **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        model: str,
        *,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        retries: int = 2,
        **options: Any,
    ) -> None:
        if not model:
            raise ValueError("give the model name, e.g. ds.LLM('claude-haiku-4-5')")
        self.provider, self.model, base, self.key_env = resolve(model, url)
        self.url: str = base or ""
        self.name = model if not url else "%s@%s" % (model, url)
        self.api_key = api_key
        self.timeout, self.retries, self.options = timeout, retries, options
        self._clients: Clients | None = None
        self._claude_clients: dict[bool, Any] = {}

    def __repr__(self) -> str:
        return "ds.LLM(%r)" % self.name

    def _key(self) -> str | None:
        key = self.api_key or (os.environ.get(self.key_env) if self.key_env else None)
        if not key and self.key_env and self.provider != "anthropic":
            raise EngineError(self.name, "no API key", "export %s=..." % self.key_env)
        return key

    def _http(self) -> Clients:
        if self._clients is None:
            key = self._key()
            self._clients = Clients(self.timeout, {"Authorization": "Bearer " + key} if key else {})
        return self._clients

    def _flow(self, system: str, user: str, model: type[BaseModel], temperature: float) -> Flow:
        """Yields request bodies, receives responses: drops unsupported options on a 400, retries a bad reply."""
        defaults: dict[str, Any] = {} if "temperature" in self.options else {"temperature": temperature}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        fmt, repaired = 0, False
        usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": None}
        while True:
            body: dict[str, Any] = {"model": self.model, "messages": messages, **defaults, **self.options}
            if _FORMATS[fmt] == "json_schema" and "response_format" not in self.options:
                schema = {"name": model.__name__.lower(), "strict": True, "schema": json_schema(model)}
                body["response_format"] = {"type": "json_schema", "json_schema": schema}
            elif _FORMATS[fmt] == "json_object" and "response_format" not in self.options:
                body["response_format"] = {"type": "json_object"}
            r = yield body
            if r.status_code in (400, 422):
                err = r.text.lower()
                if "temperature" in err and "temperature" in defaults:
                    del defaults["temperature"]
                    continue
                if fmt < len(_FORMATS) - 1 and any(w in err for w in _FORMAT_WORDS):
                    fmt += 1
                    continue
            if r.status_code in (401, 403):
                raise EngineError(self.name, "authentication failed (HTTP %d)" % r.status_code, self._fix())
            if r.status_code >= 400:
                raise EngineError(self.name, "HTTP %d: %s" % (r.status_code, r.text[:300]), FIX)
            try:
                data = r.json()
                content = data["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError):
                raise EngineError(self.name, "unexpected response: %r" % r.text[:200]) from None
            u = data.get("usage") or {}
            usage["input_tokens"] += int(u.get("prompt_tokens") or 0)
            usage["output_tokens"] += int(u.get("completion_tokens") or 0)
            try:
                return parse(content, model), usage
            except (ValidationError, ValueError) as e:
                if repaired:
                    return checked(self.name, content, model), usage
                repaired = True
                messages = [messages[0], {"role": "user", "content": retry_prompt(user, content, e)}]

    def _fix(self) -> str:
        return "check %s" % (self.key_env or "the api_key you passed")

    def _structured(self, system: str, user: str, model: type[BaseModel], temperature: float) -> Any:
        if offline.enabled():
            return model.model_validate_json(fake_reply(user, model)), {}
        if self.provider == "anthropic":
            client = self._claude(False)
            try:
                reply = client.messages.parse(**self._claude_request(system, user, model))
            except Exception as e:
                raise self._claude_error(e) from e
            return self._read_claude(reply)
        http = self._http()
        flow = self._flow(system, user, model, temperature)
        body = next(flow)
        while True:
            try:
                body = flow.send(post(http.sync(), self.url + "/chat/completions", body, self.name, self.retries))
            except StopIteration as done:
                return done.value

    async def _astructured(self, system: str, user: str, model: type[BaseModel], temperature: float) -> Any:
        if offline.enabled():
            return model.model_validate_json(fake_reply(user, model)), {}
        if self.provider == "anthropic":
            client = self._claude(True)
            try:
                reply = await client.messages.parse(**self._claude_request(system, user, model))
            except Exception as e:
                raise self._claude_error(e) from e
            return self._read_claude(reply)
        http = self._http()
        flow = self._flow(system, user, model, temperature)
        body = next(flow)
        while True:
            try:
                r = await apost(http.async_(), self.url + "/chat/completions", body, self.name, self.retries)
                body = flow.send(r)
            except StopIteration as done:
                return done.value

    def _claude(self, is_async: bool) -> Any:
        if is_async not in self._claude_clients:
            (anthropic,) = need(self.name, "Claude needs the anthropic SDK", "anthropic", "anthropic")
            cls = anthropic.AsyncAnthropic if is_async else anthropic.Anthropic
            kwargs = {"api_key": self.api_key} if self.api_key else {}
            self._claude_clients[is_async] = cls(max_retries=self.retries, timeout=self.timeout, **kwargs)
        return self._claude_clients[is_async]

    def _claude_request(self, system: str, user: str, model: type[BaseModel]) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": 4096 if "texts" in model.model_fields else 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_format": model,
            **self.options,
        }

    def _claude_error(self, e: Exception) -> EngineError:
        if isinstance(e, ValidationError):
            return EngineError(self.name, "Claude's reply did not match the schema (refusals look like this too)")
        return EngineError(self.name, "%s: %s" % (type(e).__name__, str(e)[:300]), FIX)

    def _read_claude(self, reply: Any) -> tuple[BaseModel, dict[str, Any]]:
        if getattr(reply, "stop_reason", None) == "refusal":
            raise EngineError(self.name, "Claude declined to answer (stop_reason=refusal)", "check the text")
        parsed = getattr(reply, "parsed_output", None)
        if parsed is None:
            raise EngineError(self.name, "no structured answer (stop_reason=%s)" % getattr(reply, "stop_reason", None))
        u = getattr(reply, "usage", None)
        usage = {
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "cost_usd": None,
        }
        return parsed, usage

    def ask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        obj, usage = self._structured(SYSTEM, prompt(text, questions), answers_model(questions), 0.0)
        return one_hot(obj, questions, usage, self.name)

    async def aask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        obj, usage = await self._astructured(SYSTEM, prompt(text, questions), answers_model(questions), 0.0)
        return one_hot(obj, questions, usage, self.name)

    def write(self, request: str, n: int) -> list[str]:
        obj, _ = self._structured(WRITER, write_prompt(request, n), texts_model(n), 1.0)
        return list(obj.texts)


def write(engine: Any, prompt: str, n: int) -> list[str]:
    """Ask a teacher that can write (any LLM teacher) for `n` new texts."""
    custom = getattr(engine, "write", None)
    if not callable(custom):
        raise EngineError(
            getattr(engine, "name", "?"), "can't write examples", "use an LLM, e.g. teacher='claude-haiku-4-5'"
        )
    return [str(t).strip() for t in custom(prompt, n) if str(t).strip()]
