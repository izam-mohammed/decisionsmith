"""smolagents (`pip install "decisionsmith[smolagents]"`): any smolagents model as the teacher.

from smolagents import OpenAIServerModel
ds.harness(Ticket, teacher=OpenAIServerModel(model_id="gpt-4o-mini"))  # detected automatically
"""

from __future__ import annotations

from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


def teacher(model: Any) -> TextEngine:
    """A smolagents model (OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel, ...)."""

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        message = model.generate(
            [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ]
        )
        u = getattr(message, "token_usage", None)
        usage = {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)}
        return _text(message.content), usage

    return TextEngine("smolagents:%s" % model_name(model), complete)
