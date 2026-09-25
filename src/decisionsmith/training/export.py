"""Training rows from the decision log: human labels first, else the teacher's answer."""

from __future__ import annotations

import json
from typing import Any

from ..files import write_jsonl
from ..log import Log
from ..schema import Distribution, Schema, top

FORMATS = ("answers", "typed-decisions")


def gold(schema: Schema, row: dict[str, Any]) -> dict[str, Distribution]:
    """Per field: a human label (one-hot) if there is one, else the teacher's answer."""
    out: dict[str, Distribution] = {}
    teacher = row.get("teacher_dists") or {}
    for name, f in schema.fields.items():
        human = row.get("labels", {}).get(name)
        if human in f.labels:
            out[name] = {k: float(k == human) for k in f.labels}
        elif name in teacher and set(teacher[name]) == set(f.labels):
            out[name] = {k: float(teacher[name][k]) for k in f.labels}
    return out


def gold_rows(schema: Schema, log: Log) -> list[dict[str, Any]]:
    rows = ((row, gold(schema, row)) for row in log.rows(schema.name))
    return [{"id": row["id"], "text": row["text"], "answers": answers} for row, answers in rows if answers]


def export(schema: Schema, log: Log, path: str, format: str = "answers") -> int:
    if format not in FORMATS:
        raise ValueError("format must be 'answers' or 'typed-decisions', got %r" % format)
    rows = gold_rows(schema, log)
    if format == "typed-decisions":
        rows = [_typed(schema, r) for r in rows]
    return write_jsonl(path, rows)


def _typed(schema: Schema, row: dict[str, Any]) -> dict[str, Any]:
    answers = row["answers"]
    return {
        "id": row["id"],
        "state": row["text"],
        "questions": json.dumps(schema.questions(answers), ensure_ascii=False),
        "gold": json.dumps({n: gold_entry(schema, n, d) for n, d in answers.items()}, ensure_ascii=False),
    }


def gold_entry(schema: Schema, name: str, dist: Distribution) -> dict[str, Any]:
    f = schema.fields[name]
    keys = list(f.labels) if f.kind != "scale" else [str(i) for i in range(len(f.labels))]
    probs = {k: dist[label] for k, label in zip(keys, f.labels)}
    label = keys[f.labels.index(top(dist))]
    return {"probabilities": probs, "label": label}
