"""Training data: CSV, JSONL (schema + answers, or Laya's typed-decisions) or rows -> one internal row; splits."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..schema import Schema, compile_schema, option_keys

MIN_ROWS = 20


@dataclass
class Row:
    id: str
    text: Any
    questions: dict[str, dict[str, Any]]
    targets: dict[str, list[float]]
    group: str | None = None


class DataError(ValueError):
    pass


def _normalise(probs: list[float], where: str) -> list[float]:
    if any((not isinstance(p, (int, float))) or isinstance(p, bool) or not math.isfinite(p) or p < 0 for p in probs):
        raise DataError("%s: probabilities must be non-negative numbers, got %r" % (where, probs))
    total = float(sum(probs))
    if total <= 0:
        raise DataError("%s: empty distribution" % where)
    return [float(p) / total for p in probs]


def _schema_target(schema: Schema, name: str, value: Any, where: str) -> list[float]:
    f = schema.fields[name]
    if isinstance(value, dict):
        unknown = set(map(str, value)) - set(f.labels)
        if unknown:
            raise DataError("%s: %s not options of %r (%s)" % (where, sorted(unknown), name, list(f.labels)))
        return _normalise([value.get(k, 0.0) for k in f.labels], where)
    try:
        label = schema.label_of(name, value)
    except ValueError as e:
        raise DataError("%s: %s" % (where, e)) from None
    return [1.0 if k == label else 0.0 for k in f.labels]


def _from_answers(schema: Schema, rec: dict[str, Any], rid: str, where: str, group: str | None) -> Row:
    answers = {} if rec.get("answers") is None else rec["answers"]
    if not isinstance(answers, dict):
        raise DataError("%s: 'answers' must be an object" % where)
    unknown = set(answers) - set(schema.fields)
    if unknown:
        raise DataError("%s: unknown fields %s; the schema has %s" % (where, sorted(unknown), list(schema.fields)))
    targets = {n: _schema_target(schema, n, v, "%s field %r" % (where, n)) for n, v in answers.items() if v is not None}
    return Row(rid, _text(rec.get("text"), where), schema.questions(targets), targets, group)


def _from_typed(rec: dict[str, Any], rid: str, where: str, group: str | None) -> Row:
    def obj(key: str) -> Any:
        v = rec.get(key)
        try:
            return json.loads(v) if isinstance(v, str) and key != "state" else v
        except ValueError:
            raise DataError("%s: %r is not valid JSON" % (where, key)) from None

    state, questions, gold = rec.get("state"), obj("questions"), obj("gold")
    if isinstance(state, str) and state[:1] in "{[":
        try:
            state = json.loads(state)
        except ValueError:
            pass
    if not isinstance(questions, dict) or not isinstance(gold, dict):
        raise DataError("%s: typed-decisions rows need 'questions' and 'gold' objects" % where)
    targets, qs = {}, {}
    for qid, g in gold.items():
        q = questions.get(qid)
        if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
            raise DataError("%s question %r: unknown or missing question type" % (where, qid))
        keys = option_keys(q)
        if not keys:
            raise DataError("%s question %r: no options" % (where, qid))
        probs = g.get("probabilities") if isinstance(g, dict) else None
        if isinstance(probs, dict) and probs:
            target = [probs.get(k, 0.0) for k in keys]
        elif isinstance(g, dict) and str(g.get("label")).lower() in [k.lower() for k in keys]:
            label = str(g["label"]).lower()
            target = [1.0 if k.lower() == label else 0.0 for k in keys]
        else:
            raise DataError("%s question %r: gold needs 'probabilities' or a valid 'label'" % (where, qid))
        targets[qid] = _normalise(target, "%s question %r" % (where, qid))
        qs[qid] = q
    return Row(rid, state if state is not None else "", qs, targets, group)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataError("%s: missing 'text'" % where)
    return value


def _records(data: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        if not os.path.exists(path):
            raise DataError("no such file: %s" % path)
        if path.endswith(".csv"):
            with open(path, newline="", encoding="utf-8-sig") as f:
                for i, rec in enumerate(csv.DictReader(f), start=2):
                    yield "%s line %d" % (os.path.basename(path), i), {"_csv": True, **rec}
            return
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        yield "%s line %d" % (os.path.basename(path), i), json.loads(line)
                    except ValueError:
                        raise DataError("%s line %d: not valid JSON" % (os.path.basename(path), i)) from None
        return
    for i, rec in enumerate(data):
        yield "row %d" % i, rec


def load(data: Any, schema: type[BaseModel] | Schema | None = None, group_by: str | None = None) -> list[Row]:
    """Read training data. CSV needs a `text` column and one column per schema field (blank = unlabelled)."""
    compiled = compile_schema(schema) if isinstance(schema, type) else schema
    rows: list[Row] = []
    for n, (where, rec) in enumerate(_records(data)):
        if not isinstance(rec, dict):
            raise DataError("%s: expected an object" % where)
        rid = str(rec.get("id") or "row%d" % n)
        group = None if group_by is None else str(rec.get(group_by, "")) or None
        if "questions" in rec and "gold" in rec:
            row = _from_typed(rec, rid, where, group)
        else:
            if compiled is None:
                raise DataError("%s: this format needs a schema, e.g. ds.finetune(data, Ticket)" % where)
            if rec.pop("_csv", False):
                if "text" not in rec:
                    raise DataError("%s: the CSV needs a 'text' column" % where)
                rec = {
                    "id": rec.get("id"),
                    "text": rec["text"],
                    "answers": {k: v for k, v in rec.items() if k in compiled.fields and v not in (None, "")},
                }
            row = _from_answers(compiled, rec, rid, where, group)
        if row.targets:
            rows.append(row)
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        raise DataError("row ids must be unique")
    return rows


def split(
    rows: Sequence[Row], seed: int = 0, test: float = 0.15, calib: float = 0.10, calib_max: int = 400
) -> tuple[list[Row], list[Row], list[Row]]:
    """Seeded train / calib / test split; rows sharing a `group` stay on one side."""
    if len(rows) < MIN_ROWS:
        raise DataError("need at least %d labelled rows to fine-tune, have %d" % (MIN_ROWS, len(rows)))
    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r.group or "id:" + r.id, []).append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(rows)
    want_test = max(1, round(test * n))
    want_calib = max(1, min(calib_max, round(calib * n)))
    out: tuple[list[Row], list[Row], list[Row]] = ([], [], [])
    tr, ca, te = out
    for k in keys:
        target = te if len(te) < want_test else ca if len(ca) < want_calib else tr
        target.extend(groups[k])
    if not tr or not ca or not te:
        raise DataError("groups are too large to split into train/calib/test; use fewer rows per group")
    return tr, ca, te
