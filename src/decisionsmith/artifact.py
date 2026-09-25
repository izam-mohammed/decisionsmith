"""The saved model folder: a Laya checkpoint plus `decisionsmith.json`, `report.json` and `MODEL_CARD.md`."""

from __future__ import annotations

import json
import os
import re
import shutil
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


def version_path(path: str | None, name: str) -> str:
    """`models/ticket` -> the first free `models/ticket-vN`; a path already ending in `-vN` is used as is."""
    if path is None:
        stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
        path = os.path.join("models", stem)
    path = os.path.normpath(path)
    if _VERSION.search(path):
        if os.path.exists(path):
            raise FileExistsError(
                "%s already exists and saved versions are never overwritten; "
                "use a new version number, or model.save() for the next free one" % path
            )
        return path
    n = 1
    while os.path.exists("%s-v%d" % (path, n)):
        n += 1
    return "%s-v%d" % (path, n)


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
        fields.append(
            {
                "name": f.name,
                "kind": f.kind,
                "labels": list(f.labels),
                "instructions": q["instructions"],
                "descriptions": descriptions,
            }
        )
    return {"name": schema.name, "fingerprint": schema.fingerprint, "fields": fields}


def rebuild(desc: dict[str, Any]) -> type[BaseModel]:
    """The saved schema -> a Pydantic class that asks exactly the same questions."""
    fields: dict[str, Any] = {}
    for f in desc["fields"]:
        base: Any = bool if f["kind"] == "bool" else Literal[tuple(f["labels"])]
        meta: list[Any] = [Scale] if f["kind"] == "scale" else []
        if f["descriptions"]:
            meta.append(Options(**f["descriptions"]))
        annotation = Annotated[(base, *meta)] if meta else base
        fields[f["name"]] = (annotation, Field(description=f["instructions"]))
    return create_model(desc["name"], **fields)


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _card(meta: dict[str, Any], report: Report | None) -> str:
    what = (
        "labels: %s" % ", ".join(meta["labels"])
        if meta["kind"] == "labels"
        else "fields: %s" % ", ".join("%s (%s)" % (f["name"], "/".join(f["labels"])) for f in meta["schema"]["fields"])
    )
    lines = [
        "---",
        "license: apache-2.0",
        "library_name: laya",
        "tags: [laya, system-one, decisionsmith]",
        "---",
        "",
        "# %s" % meta["name"],
        "",
        "A decision model made with decisionsmith, fine-tuned from `%s`." % (meta.get("base_model") or "laya"),
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
        'Load it with `ds.load("%s")`; the folder is also a plain Laya checkpoint (`laya.load`).' % meta["name"],
        "",
        "Built on [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) by Nandakishor M / Convai "
        "Innovations. Not affiliated with TypeSafe AI or Convai Innovations.",
        "",
    ]
    return "\n".join(lines)


def save(model: Model, path: str | None) -> str:
    from . import __version__

    source = model.trained or model.path
    if source is None:
        raise ValueError("nothing to save yet: train it first (model.train(...)), or load a saved one (ds.load)")
    dest = version_path(path, model.schema.name)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("checkpoint_latest", META, "report.*"))
    cfg_path = os.path.join(source, "rl_agent_config.json")
    trained = (_json(cfg_path).get("decisionsmith") or {}) if os.path.exists(cfg_path) else {}
    previous = model.info.get("training") if model.info else None
    training = trained or previous or {}
    version = _VERSION.search(dest)
    labels = list(model.schema.fields["label"].labels) if model.simple else None
    calibration = model.calibration
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
        "training": training,
    }
    with open(os.path.join(dest, META), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    report = model.report
    if report is None and os.path.exists(os.path.join(source, "report.json")):
        shutil.copy(os.path.join(source, "report.json"), os.path.join(dest, "train_report.json"))
    if report is not None:
        report.save(os.path.join(dest, "report.json"))
    with open(os.path.join(dest, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.write(_card(meta, report))
    return dest


def load(path: str | os.PathLike[str], schema: Any = None, *, device: str | None = None) -> Model:
    from .predictor import Model, labels_model

    path = os.fspath(path)
    meta_path = os.path.join(path, META)
    if not os.path.exists(meta_path):
        if os.path.exists(os.path.join(path, "rl_agent_config.json")):
            raise ValueError(
                "%s is a plain Laya checkpoint (no %s); use ds.model(labels_or_class, %r)" % (path, META, path)
            )
        raise FileNotFoundError("no saved model at %s (expected %s); save one with model.save()" % (path, META))
    meta = _json(meta_path)
    if meta.get("format") != FORMAT:
        raise ValueError("%s has format %r; this decisionsmith reads %r" % (meta_path, meta.get("format"), FORMAT))
    saved = meta["schema"]
    if schema is None:
        spec: Any = meta["labels"] if meta["kind"] == "labels" else rebuild(saved)
    else:
        spec = schema
        given = compile_schema(
            labels_model(list(schema), meta.get("question")) if isinstance(schema, (list, tuple)) else schema
        )
        if given.fingerprint != saved["fingerprint"]:
            want = {f["name"]: f["labels"] for f in saved["fields"]}
            got = {n: list(f.labels) for n, f in given.fields.items()}
            raise ValueError("%s does not match the saved model: saved %s, given %s" % (given.name, want, got))
    m = Model(spec, path, question=meta.get("question"), device=device)
    m.path, m.info = path, meta
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
