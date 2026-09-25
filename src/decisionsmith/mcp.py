"""MCP server (`decisionsmith mcp`): the harness, bench, status, export, label and finetune as agent tools.

Secrets come from the environment only, never from tool arguments.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from .schema import compile_schema

_HARNESSES: dict[tuple[Any, ...], Any] = {}
_LOCK = threading.Lock()
_MODELS: dict[str, type[BaseModel]] = {}


def model_from(schema: Any) -> type[BaseModel]:
    """`module:Model` / `file.py:Model`, or a JSON Schema object with enum / boolean properties."""
    if isinstance(schema, str):
        from .cli import load_schema

        if schema not in _MODELS:
            _MODELS[schema] = load_schema(schema)
        return _MODELS[schema]
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict) or not schema["properties"]:
        raise ValueError("schema must be 'module:Model' or a JSON Schema object with properties")
    key = repr(sorted(schema.items(), key=str))
    if key not in _MODELS:
        fields: dict[str, Any] = {}
        for name, prop in schema["properties"].items():
            desc = prop.get("description")
            if prop.get("type") == "boolean":
                ann: Any = bool
            elif isinstance(prop.get("enum"), list) and prop["enum"]:
                ann = Literal[tuple(str(v) for v in prop["enum"])]
            else:
                raise ValueError("property %r: only enum and boolean properties can be decided" % name)
            fields[name] = (ann, Field(description=desc))
        _MODELS[key] = create_model(str(schema.get("title") or "Decision"), **fields)
    return _MODELS[key]


def _harness(schema: Any, teacher: str | None, student: str | None, mode: str | None, log: str | None) -> Any:
    from .core import Harness

    model = model_from(schema)
    key = (model, teacher, student, mode, log)
    with _LOCK:
        if key not in _HARNESSES:
            _HARNESSES[key] = Harness(model, teacher=teacher, student=student, mode=mode, log=log)  # type: ignore[arg-type]
        return _HARNESSES[key]


def decide(
    text: str,
    schema: Any,
    teacher: str | None = None,
    student: str | None = None,
    mode: str | None = None,
    log: str | None = "decisions.db",
) -> dict[str, Any]:
    """Decide one text. Returns value per field, the engine that answered each, confidence and the log id."""
    r = _harness(schema, teacher, student, mode, log).decide(text)
    return {
        "value": r.value.model_dump(mode="json"),
        "source": r.source,
        "confidence": r.confidence,
        "sure": r.sure,
        "id": r.id,
    }


def label(id: str, fields: dict[str, Any], schema: Any, log: str = "decisions.db") -> dict[str, Any]:
    """Record the correct answer for a logged decision."""
    from .log import Log

    compiled = compile_schema(model_from(schema))
    with Log(log) as db:
        db.label(id, {n: compiled.label_of(n, v) for n, v in fields.items()})
    return {"ok": True}


def status(schema: Any, log: str = "decisions.db", student: str | None = None) -> dict[str, Any]:
    """Per-field agreement, sure rate, accuracy when sure and the next step, from a harness log."""
    from .cli import _status_dict

    return _status_dict(compile_schema(model_from(schema)), log, student, 0.8)


def export(schema: Any, out: str, log: str = "decisions.db", format: str = "answers") -> dict[str, Any]:
    """Write labelled decisions from a harness log as training JSONL."""
    from .cli import open_log
    from .training.export import export as _export

    with open_log(log) as db:
        n = _export(compile_schema(model_from(schema)), db, out, format)
    return {"rows": n, "path": os.path.abspath(out), "format": format}


def bench(data: str, schema: Any, engines: list[str]) -> dict[str, Any]:
    """Compare engines on labelled data (CSV/JSONL): accuracy, macro-F1, ECE, latency, cost, per field."""
    from .benchmark import bench as _bench

    return _bench(model_from(schema), data, engines).to_dict()


def finetune(
    data: str, schema: Any = None, base: str = "laya", train: str = "auto", out: str = "runs/v1"
) -> dict[str, Any]:
    """Fine-tune Laya on labelled data. Returns base vs fine-tuned metrics, go/no-go and the checkpoint path."""
    from .training.finetuning import finetune as _finetune

    model = model_from(schema) if schema is not None else None
    return _finetune(data, model, base=base, train=train, out=out, verbose=False).to_dict()


def engines_check(engines: list[str]) -> dict[str, Any]:
    """Check engines answer a tiny request: reachable, authenticated, latency, and the fix when not."""
    import argparse
    import contextlib
    import io
    import json

    from .cli import _doctor

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _doctor(argparse.Namespace(engines=",".join(engines), json=True))
    return json.loads(buf.getvalue())


def schema_compile(schema: Any) -> dict[str, Any]:
    """Show the Jev-format questions a schema compiles to, and any warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = compile_schema(model_from(schema))
    return {"questions": compiled.questions(), "warnings": [str(w.message) for w in caught]}


TOOLS = (decide, label, status, export, bench, finetune, engines_check, schema_compile)


def server() -> Any:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        raise SystemExit("MCP support is not installed: pip install 'decisionsmith[mcp]'") from None
    app = MCPServer(
        "decisionsmith",
        instructions="Decisions (classify, route, screen, score) with an LLM or Jev as teacher and Laya as a fast "
        "student. Ask the user before calling paid engines or sending data to hosted ones.",
    )
    for fn in TOOLS:
        app.tool()(fn)
    return app


def serve() -> None:  # pragma: no cover
    server().run("stdio")
