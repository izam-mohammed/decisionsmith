"""Outlines (`uv add "decisionsmith[outlines]"`): any Outlines model as a teacher, with structured generation.

Outlines constrains the model to the answer schema (enum options and booleans), so local models always reply with
valid JSON. Transformers, llama.cpp, MLX, vLLM, Ollama, SGLang, TGI and API models all work the same way.

ds.harness(Ticket, teacher=teacher(outlines.from_ollama(ollama.Client(), "qwen3")), student="laya")
"""

from __future__ import annotations

import asyncio
from typing import Any

from .. import offline
from ..engines.structured import SYSTEM, WRITER, answers_model, fake_reply, one_hot, prompt, texts_model, write_prompt
from ._base import model_name, run_sync


class OutlinesEngine:
    """A decisionsmith engine on an Outlines model: one structured call answers every field."""

    def __init__(self, model: Any, **options: Any) -> None:
        from outlines.models.base import AsyncModel

        self.model, self.options = model, options
        self.is_async = isinstance(model, AsyncModel)
        self.name = "outlines:%s" % (getattr(model, "model_name", None) or model_name(model))

    def _chat(self, system: str, user: str) -> Any:
        from outlines.inputs import Chat

        return Chat([{"role": "system", "content": system}, {"role": "user", "content": user}])

    def _generate(self, system: str, user: str, schema: Any) -> Any:
        if offline.enabled():
            return schema.model_validate_json(fake_reply(user, schema))
        if self.is_async:
            return run_sync(lambda: self._agenerate(system, user, schema))
        return schema.model_validate_json(self.model(self._chat(system, user), schema, **self.options))

    async def _agenerate(self, system: str, user: str, schema: Any) -> Any:
        if offline.enabled() or not self.is_async:
            return await asyncio.to_thread(self._generate, system, user, schema)
        return schema.model_validate_json(await self.model(self._chat(system, user), schema, **self.options))

    def ask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        answers = self._generate(SYSTEM, prompt(text, questions), answers_model(questions))
        return one_hot(answers, questions, {}, self.name)

    async def aask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        answers = await self._agenerate(SYSTEM, prompt(text, questions), answers_model(questions))
        return one_hot(answers, questions, {}, self.name)

    def write(self, request: str, n: int) -> list[str]:
        return list(self._generate(WRITER, write_prompt(request, n), texts_model(n)).texts)


def teacher(model: Any, **options: Any) -> OutlinesEngine:
    """An Outlines model (`outlines.from_transformers(...)`, `from_llamacpp`, `from_ollama`, `from_vllm`,
    `from_openai`, ...) as a teacher; `options` go to each generation call (`max_new_tokens=200`, ...)."""
    return OutlinesEngine(model, **options)
