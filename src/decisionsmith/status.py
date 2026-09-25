"""`status()`: where each field stands, from the log, and the one next step."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .schema import Schema, top
from .training.export import gold

CASCADE_ROWS, CASCADE_AGREE, CASCADE_ACC = 200, 0.90, 0.95
STUDENT_ROWS, STUDENT_AGREE = 500, 0.97
FINETUNE_ROWS, FINETUNE_PER_OPTION = 300, 20
HUMAN_MIN = 30


@dataclass
class FieldStatus:
    field: str
    mode: str
    rows: int
    both: int
    agreement: float | None
    sure_rate: float | None
    accuracy_when_sure: float | None
    accuracy_source: str | None
    labelled: int
    human_labels: int
    advice: str


class Status:
    """Per-field numbers and advice. `print(h.status())`, or `.to_dict()` for agents."""

    def __init__(self, schema: str, student: str | None, fields: dict[str, FieldStatus]) -> None:
        self.schema, self.student, self.fields = schema, student, fields

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "student": self.student,
            "fields": {n: asdict(f) for n, f in self.fields.items()},
        }

    def __str__(self) -> str:
        width = max(len(n) for n in self.fields) + 1
        lines = []
        for n, f in self.fields.items():
            parts = []
            if f.agreement is not None:
                parts.append("student agrees %s" % _pct(f.agreement))
            if f.sure_rate is not None:
                parts.append("sure on %s" % _pct(f.sure_rate))
            if f.accuracy_when_sure is not None:
                parts.append("accuracy when sure %s (vs %s)" % (_pct(f.accuracy_when_sure), f.accuracy_source))
            parts.append("%d labelled" % f.labelled)
            lines.append("%s %s · %s -> %s" % ((n + ":").ljust(width), f.mode, " · ".join(parts), f.advice))
        return "\n".join(lines)

    __repr__ = __str__


def _pct(v: float) -> str:
    return "%d%%" % round(100 * v)


def compute_status(
    schema: Schema,
    rows: list[dict[str, Any]],
    modes: Mapping[str, str],
    student: str | None,
    threshold: Callable[[str], float],
) -> Status:
    golds = [gold(schema, r) for r in rows]
    mine = [r for r in rows if student is not None and r["student"] == student]
    fields = {}
    for name, f in schema.fields.items():
        labels = set(f.labels)
        sdist = [(r, d) for r in mine if (d := (r["student_dists"] or {}).get(name)) and set(d) == labels]
        sure_rows = [(r, d) for r, d in sdist if max(d.values()) >= threshold(name)]
        agree = [_agrees(r, d, name) for r, d in sdist if name in (r["teacher_dists"] or {})]
        teacher_sure = [_agrees(r, d, name) for r, d in sure_rows if name in (r["teacher_dists"] or {})]
        human_sure = [top(d) == r["labels"][name] for r, d in sure_rows if name in r["labels"]]
        if len(human_sure) >= HUMAN_MIN:
            acc, source = sum(human_sure) / len(human_sure), "human"
        elif teacher_sure:
            acc, source = sum(teacher_sure) / len(teacher_sure), "teacher"
        else:
            acc, source = None, None
        per_option = dict.fromkeys(f.labels, 0)
        for g in golds:
            if name in g:
                per_option[top(g[name])] += 1
        fs = FieldStatus(
            field=name,
            mode=modes[name],
            rows=len(sdist),
            both=len(agree),
            agreement=sum(agree) / len(agree) if agree else None,
            sure_rate=len(sure_rows) / len(sdist) if sdist else None,
            accuracy_when_sure=acc,
            accuracy_source=source,
            labelled=sum(per_option.values()),
            human_labels=sum(1 for r in rows if name in r["labels"]),
            advice="",
        )
        fs.advice = _advice(fs, student, min(per_option.values()))
        fields[name] = fs
    return Status(schema.name, student, fields)


def _agrees(row: dict[str, Any], dist: dict[str, float], name: str) -> bool:
    return top(dist) == top(row["teacher_dists"][name])


def _advice(f: FieldStatus, student: str | None, rarest: int) -> str:
    if student is None:
        return "no student yet: add student='laya' in shadow mode"
    agree = f.agreement or 0.0
    acc_ok = f.accuracy_when_sure is not None and f.accuracy_when_sure >= CASCADE_ACC
    if f.both >= STUDENT_ROWS and agree >= STUDENT_AGREE and acc_ok:
        return "ready for student" if f.mode != "student" else "in student mode; keep labelling a sample"
    if f.both >= CASCADE_ROWS and agree >= CASCADE_AGREE and acc_ok:
        return "ready for cascade" if f.mode in ("teacher", "shadow") else "cascade is right; keep auditing"
    if f.labelled >= FINETUNE_ROWS and rarest >= FINETUNE_PER_OPTION:
        return "ready to finetune: run h.finetune()"
    if f.both >= CASCADE_ROWS:
        return "student disagrees often: collect %d labels (have %d), then h.finetune()" % (
            FINETUNE_ROWS,
            f.labelled,
        )
    if f.mode in ("teacher",) and f.both == 0:
        return "use mode='shadow' to measure the student"
    return "needs more data (have %d, want %d)" % (f.both, CASCADE_ROWS)
