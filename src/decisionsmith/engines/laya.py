"""Laya in-process via `laya.load`."""

from __future__ import annotations

import os
import threading
from typing import Any

from .base import EngineError, need

LAYA_REPO = "convaiinnovations/laya"
_ALIASES: dict[str, str | None] = {
    "laya": None,
    "english": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}
_AGENTS: dict[tuple[str, str | None, str | None], Any] = {}
_AGENTS_LOCK = threading.Lock()


def resolve_laya(spec: str) -> tuple[str, str | None]:
    """`laya` / `english` / `multilingual` / `typed-decisions` / a local dir / an HF repo id."""
    if spec in _ALIASES:
        return LAYA_REPO, _ALIASES[spec]
    if os.path.isdir(spec):
        return os.path.abspath(spec), None
    if spec.count("/") == 1 and not spec.startswith((".", "/", "~")):
        return spec, None
    raise EngineError(
        "laya:%s" % spec,
        "no such checkpoint",
        "use laya, laya:multilingual, laya:typed-decisions, laya:<dir> or laya:<org>/<repo>",
    )


class LayaEngine:
    """Laya in-process via `laya.load`. Checkpoints are loaded once per process and shared."""

    def __init__(self, model: str = "laya", *, device: str | None = None, batch_size: int = 32) -> None:
        self.spec = model
        self.model_id, self.subfolder = resolve_laya(model)
        self.device = device
        self.batch_size = batch_size
        self.name = "laya" if model == "laya" else "laya:%s" % model

    @property
    def agent(self) -> Any:
        key = (self.model_id, self.subfolder, self.device)
        with _AGENTS_LOCK:
            if key not in _AGENTS:
                (laya,) = need(self.name, "laya is not installed", "laya", "laya")
                try:
                    _AGENTS[key] = laya.load(self.model_id, device=self.device, subfolder=self.subfolder)
                except Exception as e:
                    raise EngineError(self.name, "could not load: %s" % e, "check the checkpoint path or id") from e
            return _AGENTS[key]

    def ask(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return self.ask_many([text], questions)[0]

    def ask_many(self, texts: list[str], questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        agent = self.agent
        try:
            return list(agent.predict_batch(texts, questions, batch_size=self.batch_size))
        except ValueError as e:
            raise EngineError(self.name, str(e), "check the schema's options fit the model") from e
