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
    split: str | None = None
    labelled_by: str | None = None


SPLITS = ("train", "calib", "test")
ALIASES = {"dev": "calib", "val": "calib", "valid": "calib", "validation": "calib"}
FORMULA = ("=", "+", "-", "@")


def same_text(text: Any) -> str:
    """The form two texts must share to count as the same: NFKC, case-folded, punctuation dropped, spaces collapsed."""
    import unicodedata

    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    kept = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in folded)
    return " ".join(kept.split())


def text_hash(text: Any) -> str:
    """A short, one-way fingerprint of a text (see `same_text`): provenance without storing the text."""
    import hashlib

    return hashlib.sha256(same_text(text).encode()).hexdigest()[:16]


def _formula(value: str) -> bool:
    return value.startswith(FORMULA) or (len(value) > 1 and value[0] == "'" and value[1] in FORMULA)


def safe_cell(value: str) -> str:
    """Quote a CSV cell a spreadsheet would run as a formula (`=`, `+`, `-`, `@` first), and one that already
    starts with `'` plus one of those, so reading it back gives the original text."""
    return "'" + value if _formula(value) else value


def unsafe_cell(value: Any) -> Any:
    """Undo `safe_cell` when reading a CSV back."""
    if isinstance(value, str) and value[:1] == "'" and _formula(value[1:]):
        return value[1:]
    return value


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
        if path.lower().endswith(".csv"):
            with open(path, newline="", encoding="utf-8-sig") as f:
                for i, rec in enumerate(csv.DictReader(f), start=2):
                    rec = {k: unsafe_cell(v) for k, v in rec.items()}
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


def _split_of(rec: dict[str, Any], where: str) -> str:
    held = str(rec.get("split") or "").strip().lower()
    held = ALIASES.get(held, held)
    if held and held not in SPLITS:
        raise DataError("%s: split must be one of %s (or blank), got %r" % (where, SPLITS, held))
    return held


def load(
    data: Any, schema: type[BaseModel] | Schema | None = None, group_by: str | None = None, split: str = "train"
) -> list[Row]:
    """Read training data. CSV needs a `text` column and one column per schema field (blank = unlabelled).

    A `split` column (a golden dataset: `train`, `calib` or `test`; `dev`/`val` mean `calib`) picks rows:
    `split="train"` never returns `test` rows; `split="test"` returns only `test` rows, or every row when no row
    has a split. A blank split counts as `train` whenever any row has one. `split="all"` returns everything.
    """
    if split not in ("train", "test", "all"):
        raise ValueError("split must be 'train', 'test' or 'all', got %r" % split)
    compiled = compile_schema(schema) if isinstance(schema, type) else schema
    records = [(where, rec) for where, rec in _records(data)]
    for where, rec in records:
        if not isinstance(rec, dict):
            raise DataError("%s: expected an object" % where)
    marked = any(_split_of(rec, where) for where, rec in records)
    everything: list[Row] = []
    for n, (where, rec) in enumerate(records):
        held = _split_of(rec, where) or ("train" if marked else "")
        by = str(rec.get("labelled_by") or "").strip() or None
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
        row.split, row.labelled_by = held or None, by
        if row.targets:
            everything.append(row)
    return [r for r in _merge_repeats(everything) if _keep(r, split, marked)]


def _keep(row: Row, split: str, marked: bool) -> bool:
    if split == "train":
        return row.split != "test"
    if split == "test":
        return not marked or row.split == "test"
    return True


def _merge_repeats(rows: list[Row]) -> list[Row]:
    """The same row twice (joined golden runs) counts once, and as `test` if any copy is `test`."""
    by_id: dict[str, Row] = {}
    for r in rows:
        first = by_id.get(r.id)
        if first is None:
            by_id[r.id] = r
            continue
        if first.text != r.text or first.targets != r.targets:
            raise DataError(
                "row id %r appears twice with different text or labels; give each row its own id, or fix one copy"
                % r.id
            )
        if r.split == "test":
            first.split = "test"
    return list(by_id.values())


def split(
    rows: Sequence[Row], seed: int = 0, test: float = 0.15, calib: float = 0.10, calib_max: int = 400
) -> tuple[list[Row], list[Row], list[Row]]:
    """Seeded train / calib / test split; rows sharing a `group` stay on one side.

    Rows marked `split=calib` (a golden dataset) are always the calibration split; the rest are split as usual.
    """
    marked = [r for r in rows if r.split == "calib"]
    if marked:
        tr, ca, te = split([r for r in rows if r.split != "calib"], seed, test, calib, calib_max)
        return tr + ca, marked, te
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
