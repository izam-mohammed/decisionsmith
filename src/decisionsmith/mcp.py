"""MCP server (`decisionsmith mcp`): the harness, bench, status, export, label and finetune as agent tools, plus
the tools a coding agent uses to build a model itself (golden_*, data_check, evaluate, model_info).

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


def _spec(schema: Any, labels: list[str] | None) -> Any:
    """Labels, a golden session file (its schema), `module:Model` or a JSON Schema -> what `ds.model` takes."""
    from . import golden_session

    if labels:
        return [str(x) for x in labels]
    if isinstance(schema, str) and schema.lower().endswith(".json"):
        return golden_session.spec_of(golden_session.load(schema))
    return model_from(schema)


def _class(spec: Any) -> type[BaseModel]:
    from .predictor import labels_model

    return labels_model(spec) if isinstance(spec, list) else spec


def finetune(
    data: str,
    schema: Any = None,
    base: str = "laya",
    train: str = "auto",
    out: str = "runs/v1",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Fine-tune Laya on labelled data (split=test rows are held out). `schema` may be a golden session file.
    Returns base vs fine-tuned metrics, go/no-go and the checkpoint path."""
    from .training.finetuning import finetune as _finetune

    model = _class(_spec(schema, labels)) if schema is not None or labels else None
    return _finetune(data, model, base=base, train=train, out=out, verbose=False).to_dict()


def golden_start(
    source: str,
    schema: Any = None,
    labels: list[str] | None = None,
    n: int = 200,
    strategy: str = "diverse",
    test: float = 0.2,
    out: str = "golden.csv",
    overwrite: bool = False,
    agent: str | None = None,
) -> dict[str, Any]:
    """Pick the rows worth labelling (from a harness log .db or a file of texts) into a labelling session file.
    strategy: diverse | random | uncertain | disagree. Then label them with golden_batch / golden_submit."""
    from . import golden_session
    from .golden_set import golden

    spec = _spec(schema, labels)
    teacher = "agent:%s" % agent if agent else "agent"
    rows = golden(source, teacher, n, strategy, schema=spec, test=test, out=out, overwrite=overwrite, verbose=False)
    path = golden_session.session_path(out)
    return {
        "session": path,
        "rows": len(rows),
        "test": sum(r["split"] == "test" for r in rows),
        "next": "golden_batch(%r)" % path,
    }


def golden_batch(session: str, size: int = 20) -> dict[str, Any]:
    """The next texts to label with the options and instructions. A share of texts comes back under a new id for an
    independent second answer; give each batch to a fresh labeller that can't see earlier answers."""
    from .golden_session import batch

    return batch(session, size)


def golden_submit(session: str, answers: list[dict[str, Any]], agent: str | None = None) -> dict[str, Any]:
    """Store answers: [{"id": ..., "answers": {field: option}}] or [{"id": ..., "skip": "why"}]. Each is checked
    against the schema and each id takes one answer; bad items come back in `rejected` with the reason. `agent`
    names you, e.g. claude-code."""
    from .golden_session import submit

    return submit(session, answers, agent)


def golden_add(session: str, examples: list[dict[str, Any]], agent: str | None = None) -> dict[str, Any]:
    """Add examples you wrote when real data is short: [{"text": ..., "answers": {field: option}}]. They are marked
    synthetic, train rows only, and count only after a second, independent answer agrees. A text that repeats a
    session text or shares 80% or more of its words with a test text is rejected."""
    from .golden_session import add

    return add(session, examples, agent)


def golden_status(session: str) -> dict[str, Any]:
    """Progress, label balance, agreement between the passes, disagreements and skipped rows to show the user."""
    from .golden_session import status

    return status(session)


def golden_finish(session: str, out: str | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Write golden.csv from the session (same format as an LLM-labelled one, plus a `checked` column). The session
    is then finished: no more batches, answers or examples."""
    from .golden_session import finish

    return finish(session, out, overwrite)


def data_check(path: str, schema: Any = None, labels: list[str] | None = None) -> dict[str, Any]:
    """Check a data file before training: counts per option, repeated texts, test/train leaks and near copies
    (training texts sharing 80% or more of their words with a test text), lengths, advice.
    Without schema/labels every column other than id/text/split/labelled_by is read as a field."""
    from .checks import check

    given = schema is not None or bool(labels)
    return check(path, compile_schema(_class(_spec(schema, labels))) if given else None)


def evaluate(
    model: str,
    data: str,
    schema: Any = None,
    labels: list[str] | None = None,
    target: float = 0.97,
    save: str | None = None,
) -> dict[str, Any]:
    """Evaluate a saved model folder, or a checkpoint (e.g. finetune's runs/v1, with schema or labels), on labelled
    data (split=test rows of a golden.csv): per field numbers, and by who labelled the rows (accuracy on rows a
    person labelled, agreement with an agent's or LLM's labels), go/no-go. `save` (e.g.
    models/ticket) then writes the next versioned model folder with its thresholds and this report."""
    from .artifact import META
    from .predictor import Model, load

    given = schema is not None or bool(labels)
    if os.path.isfile(os.path.join(model, META)):
        m = load(model, _spec(schema, labels) if given else None)
    elif given:
        m = Model(_spec(schema, labels), model)
    else:
        raise ValueError("%s is not a saved model folder; pass schema=... or labels=[...] with a checkpoint" % model)
    out = m.evaluate(data, target=target).to_dict()
    if save:
        out["saved"] = m.save(save, verbose=False)
        out["load"] = "ds.load(%r)" % out["saved"]
    return out


def model_info(path: str) -> dict[str, Any]:
    """What a saved model folder holds: labels or fields, thresholds, the saved report's go/no-go, how to load it."""
    from .artifact import read_meta

    meta = read_meta(path)
    report = os.path.join(path, "report.json")
    saved: Any = None
    if os.path.exists(report):
        import json

        with open(report, encoding="utf-8") as f:
            r = json.load(f)
        saved = {"go": r.get("go"), "reasons": r.get("reasons"), "rows": r.get("rows")}
    return {
        "name": meta["name"],
        "kind": meta["kind"],
        "labels": meta["labels"],
        "fields": {f["name"]: f["labels"] for f in meta["schema"]["fields"]},
        "thresholds": meta["thresholds"],
        "base_model": meta.get("base_model"),
        "report": saved,
        "load": "ds.load(%r)" % path,
    }


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


TOOLS = (
    decide,
    label,
    status,
    export,
    bench,
    finetune,
    engines_check,
    schema_compile,
    golden_start,
    golden_batch,
    golden_submit,
    golden_add,
    golden_status,
    golden_finish,
    data_check,
    evaluate,
    model_info,
)


def server() -> Any:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        raise SystemExit("MCP support is not installed: uv add 'decisionsmith[mcp]'") from None
    app = MCPServer(
        "decisionsmith",
        instructions="Decisions (classify, route, screen, score) with an LLM or Jev as teacher and Laya as a fast "
        "student. Ask the user before calling paid engines or sending data to hosted ones. To build a model "
        "yourself: golden_start, then golden_batch / golden_submit until done (a share of texts comes back under "
        "new ids for an independent second answer; use a fresh labeller per batch), golden_finish, finetune, "
        "evaluate (with save=...). Quote measured numbers only; flag synthetic data.",
    )
    for fn in TOOLS:
        app.tool()(fn)
    return app


def serve() -> None:  # pragma: no cover
    server().run("stdio")
