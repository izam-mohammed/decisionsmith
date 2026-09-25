"""simonw/llm (`uv add "decisionsmith[llm-plugin]"`): an `llm decide` command, and any `llm` model as the teacher.

llm decide "you charged me twice" -l billing -l technical -l sales             # base Laya decides
llm decide "..." -l billing -l technical --engine laya:./runs/v1 --teacher gpt-5-mini   # an llm model when unsure
ds.harness(Ticket, teacher=teacher(llm.get_model("gpt-5-mini")), student="laya")
"""

from __future__ import annotations

import sys
from typing import Any

from ..engines.structured import TextEngine
from ._base import model_name, run_sync


def teacher(model: Any, **options: Any) -> TextEngine:
    """An `llm` model (`llm.get_model(...)` or `llm.get_async_model(...)`) as a teacher; `options` are its
    model options (`temperature=0`, ...)."""
    import llm

    def read(text: str, usage: Any) -> tuple[str, dict[str, Any]]:
        return text, {"input_tokens": usage.input, "output_tokens": usage.output}

    name = "llm:%s" % getattr(model, "model_id", model_name(model))
    if isinstance(model, (llm.AsyncModel, llm.AsyncKeyModel)):

        async def acomplete(system: str, user: str) -> tuple[str, dict[str, Any]]:
            r = model.prompt(user, system=system, stream=False, **options)
            return read(await r.text(), await r.usage())

        return TextEngine(name, lambda s, u: run_sync(lambda: acomplete(s, u)), acomplete)

    def complete(system: str, user: str) -> tuple[str, dict[str, Any]]:
        r = model.prompt(user, system=system, stream=False, **options)
        return read(r.text(), r.usage())

    return TextEngine(name, complete)


def _register_commands() -> Any:
    import click
    import llm

    @llm.hookimpl
    def register_commands(cli: Any) -> None:
        @cli.command(name="decide")
        @click.argument("text", required=False)
        @click.option("labels", "-l", "--label", multiple=True, required=True, help="A label (repeat: -l a -l b)")
        @click.option("--engine", default="laya", help="The decisionsmith model: laya, laya:<dir>, jev, ...")
        @click.option("--teacher", "teacher_id", help="An llm model id that answers when the engine is unsure")
        @click.option("--question", help="What to decide, e.g. 'Which team should handle this?'")
        def decide(text: str | None, labels: tuple[str, ...], engine: str, teacher_id: str | None, question: Any):
            """Decide TEXT (or stdin) with a decisionsmith model: prints one of the labels."""
            import decisionsmith as ds

            text = text if text is not None else sys.stdin.read()
            m: Any = ds.model(list(labels), engine, question=question)
            if teacher_id:
                m = ds.harness(m, teacher=teacher(llm.get_model(teacher_id)), log=None)
            click.echo(m(text.strip()) if teacher_id else m.predict(text.strip()))

    return register_commands


def __getattr__(name: str) -> Any:
    # llm loads this module through its `llm` entry point; the hook is built on first access, so importing
    # this module (and decisionsmith) never imports llm or click.
    if name == "register_commands":
        return _register_commands()
    raise AttributeError(name)


def __dir__() -> list[str]:
    return [*globals(), "register_commands"]
