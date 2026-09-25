"""The saved model folder: a Laya checkpoint plus `decisionsmith.json`, `report.json` and `MODEL_CARD.md`."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field, create_model

from .report import Report
from .schema import Options, Scale, Schema, compile_schema

if TYPE_CHECKING:
    from .predictor import Model

FORMAT = "decisionsmith.model/1"
META = "decisionsmith.json"
_VERSION = re.compile(r"-v(\d+)$")


def stem_of(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60].strip("-") or "model"


def _next(path: str) -> str:
    folder, base = os.path.split(path)
    taken = [
        int(m.group(1))
        for entry in (os.listdir(folder or ".") if os.path.isdir(folder or ".") else [])
        if (m := re.fullmatch(re.escape(base) + r"-v(\d+)", entry))
    ]
    return "%s-v%d" % (path, max(taken, default=0) + 1)


def version_path(path: str | os.PathLike[str] | None, name: str) -> tuple[str, bool]:
    """`models/ticket` -> `models/ticket-v<highest + 1>`; a path already ending in `-vN` is used as is.

    Returns the path and whether it was numbered automatically (so a clash can move on to the next number).
    """
    if path is None:
        path = os.path.join("models", stem_of(name))
    path = os.path.normpath(os.path.expanduser(os.fspath(path)))
    if _VERSION.search(path):
        if os.path.exists(path):
            raise FileExistsError(
                "%s already exists and saved versions are never overwritten; "
                "use a new version number, or model.save() for the next free one" % path
            )
        return path, False
    return _next(path), True


def describe(schema: Schema) -> dict[str, Any]:
    fields = []
    for f in schema.fields.values():
        q = f.question
        crit = q.get("criteria")
        if isinstance(crit, dict):
            descriptions = dict(crit)
        elif f.kind == "scale":
            descriptions = {
                label: c[len(label) + 2 :] for label, c in zip(f.labels, crit or []) if c.startswith(label + ": ")
            }
        else:
            descriptions = {}
        plain = all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in f.values)
        fields.append(
            {
                "name": f.name,
                "kind": f.kind,
                "labels": list(f.labels),
                "values": list(f.values) if f.kind != "bool" and plain else None,
                "instructions": q["instructions"],
                "descriptions": descriptions,
            }
        )
    return {"name": schema.name, "fingerprint": schema.fingerprint, "fields": fields}


def rebuild(desc: dict[str, Any]) -> type[BaseModel]:
    """The saved schema -> a Pydantic class that asks exactly the same questions."""
    fields: dict[str, Any] = {}
    for f in desc["fields"]:
        base: Any = bool if f["kind"] == "bool" else Literal[tuple(f.get("values") or f["labels"])]
        meta: list[Any] = [Scale] if f["kind"] == "scale" else []
        if f["descriptions"]:
            meta.append(Options(**f["descriptions"]))
        annotation = Annotated[(base, *meta)] if meta else base
        fields[f["name"]] = (annotation, Field(description=f["instructions"]))
    return create_model(desc["name"], **fields)


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _card(meta: dict[str, Any], report: Report | None, where: str) -> str:
    what = (
        "labels: %s" % ", ".join(meta["labels"])
        if meta["kind"] == "labels"
        else "fields: %s" % ", ".join("%s (%s)" % (f["name"], "/".join(f["labels"])) for f in meta["schema"]["fields"])
    )
    lines = [
        "---",
        "library_name: laya",
        "tags: [laya, system-one, decisionsmith]",
        "---",
        "",
        "# %s" % meta["name"],
        "",
        "A decision model made with decisionsmith, fine-tuned from `%s`." % _base_name(meta.get("base_model")),
        "",
        "- %s" % what,
        "- saved: %s, decisionsmith %s" % (meta["created"][:10], meta["decisionsmith_version"]),
    ]
    if report is not None:
        lines.append("- evaluation (report.json): %s" % ("go" if report.go else "no-go"))
        for row in report.rows:
            acc = row.get("accuracy")
            lines.append(
                "  - %s: accuracy %s on %s decisions"
                % (row["field"], "-" if acc is None else "%.3f" % acc, row.get("decisions"))
            )
    lines += [
        "",
        'Load it with `ds.load("%s")` (or the folder\'s path wherever you copy it). The folder is also a plain Laya '
        "checkpoint (`laya.load`)." % where.replace("\\", "/"),
        "",
        "Licence: choose one for your model and its training data. The Laya base weights it was fine-tuned from are "
        "Apache-2.0.",
        "",
        "Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai "
        "Innovations. Not affiliated with TypeSafe AI or Convai Innovations.",
        "",
    ]
    return "\n".join(lines)


def _base_name(base: Any) -> str:
    if not base:
        return "laya"
    text = str(base)
    return os.path.basename(os.path.normpath(text)) if os.path.isabs(text) or os.sep in text else text


def shown_path(path: str) -> str:
    """How a saved folder is named in its model card: relative to the working folder if inside it, else its name."""
    full, here = os.path.abspath(path), os.path.abspath(os.getcwd())
    try:
        inside = os.path.commonpath([full, here]) == here
    except ValueError:
        inside = False
    return os.path.relpath(full, here) if inside else os.path.basename(full)


def _public(report: dict[str, Any]) -> dict[str, Any]:
    """A report as saved in the folder: no texts, no local paths, no training row ids."""
    from .evaluation import short_name

    details = {k: v for k, v in (report.get("details") or {}).items() if k not in ("worst", "train_ids")}
    for key in ("base", "finetuned"):
        if isinstance(details.get(key), dict):
            details[key] = {k: v for k, v in details[key].items() if k != "worst"}
    title = " ".join(short_name(w) if ":" in w else w for w in str(report.get("title", "")).split(" "))
    if "->" in title:
        head, _, tail = title.partition("-> ")
        title = head + "-> " + os.path.basename(os.path.normpath(tail))
    return {**report, "title": title, "path": None, "details": details}


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str)


def _source(model: Model) -> str:
    from .engines import LayaEngine

    if model.trained or model.path:
        return str(model.trained or model.path)
    spec = model.engine.spec if isinstance(model.engine, LayaEngine) else None
    if spec and os.path.isdir(os.path.expanduser(spec)):
        return os.path.expanduser(spec)
    raise ValueError("nothing to save yet: train it first (model.train(...)), or load a saved one (ds.load)")


def _default_path(model: Model) -> str | None:
    if model.path and _VERSION.search(os.path.normpath(model.path)):
        return _VERSION.sub("", os.path.normpath(model.path))
    return None


def save(model: Model, path: str | os.PathLike[str] | None, *, verbose: bool = True) -> str:
    from . import __version__

    source = _source(model)
    name = "-".join(model.schema.fields["label"].labels) if model.simple else model.schema.name
    dest, numbered = version_path(path if path is not None else _default_path(model), name)
    parent_dir = os.path.dirname(os.path.abspath(dest))
    os.makedirs(parent_dir, exist_ok=True)
    tmp = os.path.join(parent_dir, ".%s.tmp-%s" % (os.path.basename(dest), uuid.uuid4().hex))
    shutil.copytree(source, tmp, ignore=shutil.ignore_patterns("checkpoint_latest", META, "report.*", "*.tmp-*"))
    cfg_path = os.path.join(source, "rl_agent_config.json")
    trained = (_json(cfg_path).get("decisionsmith") or {}) if os.path.exists(cfg_path) else {}
    training = trained or (model.meta.get("training") if model.meta else None) or {}
    labels = list(model.schema.fields["label"].labels) if model.simple else None
    calibration = model.calibration
    report = model.report
    if report is None and os.path.exists(os.path.join(source, "report.json")):
        _write_json(os.path.join(tmp, "train_report.json"), _public(_json(os.path.join(source, "report.json"))))
    if report is not None:
        _write_json(os.path.join(tmp, "report.json"), _public(report.to_dict()))
    while True:
        version = _VERSION.search(dest)
        meta = {
            "format": FORMAT,
            "name": os.path.basename(dest),
            "version": int(version.group(1)) if version else None,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "decisionsmith_version": __version__,
            "kind": "labels" if model.simple else "class",
            "labels": labels,
            "question": model.schema.fields["label"].question["instructions"] if model.simple else None,
            "schema": describe(model.schema),
            "questions": model.schema.questions(),
            "calibration": {n: c["temperature"] for n, c in calibration.items() if c.get("temperature") is not None},
            "thresholds": {n: c.get("threshold") for n, c in calibration.items() if "threshold" in c},
            "base_model": training.get("base"),
            "data_hash": training.get("data_hash"),
            "training": {k: v for k, v in training.items() if k != "text_hashes"},
        }
        _write_json(os.path.join(tmp, META), meta)
        with open(os.path.join(tmp, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
            f.write(_card(meta, report, shown_path(dest)))
        try:
            if os.path.exists(dest):
                raise FileExistsError(dest)
            os.rename(tmp, dest)
            break
        except OSError:
            if not numbered or not os.path.exists(dest):
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            dest = _next(_VERSION.sub("", dest))
    if verbose:
        print("saved to %s" % dest)
    return dest


REQUIRED = ("format", "name", "kind", "labels", "question", "schema", "questions", "calibration", "thresholds")


def read_meta(path: str) -> dict[str, Any]:
    meta_path = os.path.join(path, META)
    try:
        meta = _json(meta_path)
    except ValueError as e:
        raise ValueError("%s is not valid JSON (%s); save the model again" % (meta_path, e)) from None
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        found = meta.get("format") if isinstance(meta, dict) else None
        raise ValueError("%s has format %r; this decisionsmith reads %r" % (meta_path, found, FORMAT))
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ValueError("%s is missing %s; save the model again with model.save()" % (meta_path, ", ".join(missing)))
    return meta


def _mismatch(given: Schema, meta: dict[str, Any]) -> str | None:
    saved, got = meta["questions"], given.questions()
    if saved == got:
        return None
    out = []
    for name in sorted(set(saved) | set(got)):
        a, b = saved.get(name), got.get(name)
        if a is None or b is None:
            out.append("%s: %s" % (name, "only in the saved model" if b is None else "not in the saved model"))
            continue
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                out.append("%s.%s: saved %r, given %r" % (name, key, a.get(key), b.get(key)))
    return "; ".join(out)


def load(path: str | os.PathLike[str], schema: Any = None, *, device: str | None = None) -> Model:
    from .predictor import Model, labels_model

    path = os.path.expanduser(os.fspath(path))
    if not os.path.exists(os.path.join(path, META)):
        if os.path.exists(os.path.join(path, "rl_agent_config.json")):
            raise ValueError(
                "%s is a plain Laya checkpoint (no %s); use ds.model(labels_or_class, %r)" % (path, META, path)
            )
        raise FileNotFoundError("no saved model at %s (expected %s); save one with model.save()" % (path, META))
    meta = read_meta(path)
    if schema is None:
        spec: Any = meta["labels"] if meta["kind"] == "labels" else rebuild(meta["schema"])
    else:
        spec = schema
        given = compile_schema(
            labels_model(list(schema), meta.get("question")) if isinstance(schema, (list, tuple)) else schema
        )
        diff = _mismatch(given, meta)
        if diff:
            raise ValueError(
                "%s does not ask the same questions as the saved model (%s); load without it, or change it to match"
                % (given.name, diff)
            )
    m = Model(spec, path, question=meta.get("question"), device=device)
    m.path, m.meta = path, meta
    for name, temperature in meta["calibration"].items():
        m.calibration.setdefault(name, {})["temperature"] = temperature
    for name, threshold in meta["thresholds"].items():
        m.calibration.setdefault(name, {})["threshold"] = threshold
    report_path = os.path.join(path, "report.json")
    if os.path.exists(report_path):
        r = _json(report_path)
        m.report = Report(
            r["kind"], r["title"], r["rows"], go=r["go"], reasons=r["reasons"], path=r["path"], details=r["details"]
        )
    return m
