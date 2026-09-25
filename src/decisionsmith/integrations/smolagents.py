"""smolagents (`uv add "decisionsmith[smolagents]"`): any smolagents model as the teacher, and a `Tool`.

ds.harness(Ticket, teacher=OpenAIServerModel(model_id="gpt-5-mini"))   # detected automatically
CodeAgent(tools=[tool(h, "route_ticket")], model=model)                # the agent asks your model
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..engines.structured import TextEngine
from ._base import model_name, resolve

_CLASSES: dict[str, type] = {}


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


def _tool_class() -> type:
    from smolagents import Tool

    class DecisionTool(Tool):
        """A smolagents `Tool` taking `text` and returning the decision: a dict of fields, or the label for a
        `ds.model(labels)`."""

        def __init__(self, x: Any, name: str = "decide", description: str | None = None) -> None:
            self.decider = resolve(x)
            self.name = name
            self.description = description or "Decide %s for a text (%s)." % (
                ", ".join(self.decider.fields),
                self.decider.name,
            )
            self.inputs = {"text": {"type": "string", "description": "The text to decide."}}
            self.output_type = "string" if self.decider.simple else "object"
            super().__init__()

        def forward(self, text: str) -> Any:
            value = self.decider.call(text)
            return value.model_dump(mode="json") if isinstance(value, BaseModel) else value

    return DecisionTool


def __getattr__(name: str) -> Any:
    if name == "DecisionTool":
        return _CLASSES.setdefault(name, _tool_class())
    raise AttributeError(name)


def tool(x: Any, name: str = "decide", description: str | None = None) -> Any:
    """A `DecisionTool` for `CodeAgent(tools=[...])` / `ToolCallingAgent(tools=[...])`."""
    return __getattr__("DecisionTool")(x, name, description)
