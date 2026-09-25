"""`adapt()`: per-field temperature and cascade threshold for any student, fitted from the decision log."""

from __future__ import annotations

import math
from typing import Any

from ..schema import Distribution, Field, Schema, argmax
from . import calibrate

MIN_ADAPT_ROWS = 100

Pair = tuple[str, list[float], list[float]]


def adapt_key(schema: str, student: str | None) -> str:
    return "adapt:%s:%s" % (schema, student or "")


def threshold_for(calib: dict[str, dict[str, Any]], name: str, default: float) -> float:
    """The adapted threshold for a field; a field adapt() found no safe threshold for never trusts the student."""
    c = calib.get(name)
    if c is None or "threshold" not in c:
        return default
    return 1.01 if c.get("threshold") is None else float(c["threshold"])


def calibrated(calib: dict[str, dict[str, Any]], name: str, dist: Distribution) -> Distribution:
    c = calib.get(name)
    if not c or c.get("temperature") in (None, 1.0):
        return dist
    labels = list(dist)
    return dict(zip(labels, calibrate.scale([dist[k] for k in labels], float(c["temperature"]))))


def fit(
    schema: Schema,
    rows: list[tuple[dict[str, Any], dict[str, Distribution]]],
    previous: dict[str, dict[str, Any]],
    target: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """(student row, gold) pairs -> (calibration per field, report rows). Fields with too few rows keep `previous`."""
    calib: dict[str, dict[str, Any]] = {}
    table = []
    for name, f in schema.fields.items():
        pairs = _pairs(f, rows)
        if len(pairs) < MIN_ADAPT_ROWS:
            table.append({"field": name, "rows": len(pairs), "note": "needs %d labelled student rows" % MIN_ADAPT_ROWS})
            if name in previous:
                calib[name] = previous[name]
            continue
        fit_on = [p for p in pairs if int(p[0][-1], 16) % 2 == 0] or pairs
        held = [p for p in pairs if int(p[0][-1], 16) % 2 == 1] or pairs
        temp = calibrate.fit_temperature([_logit(p[1]) for p in fit_on], [p[2] for p in fit_on])
        before = [(max(p[1]), float(argmax(p[1]) == argmax(p[2]))) for p in held]
        scaled = [calibrate.scale(p[1], temp) for p in held]
        after = [(max(q), float(argmax(q) == argmax(p[2]))) for q, p in zip(scaled, held)]
        thr = calibrate.threshold([c for c, _ in after], [k for _, k in after], target)
        covered = [k for c, k in after if thr is not None and c >= thr]
        calib[name] = {"temperature": temp, "threshold": thr}
        table.append(
            {
                "field": name,
                "rows": len(pairs),
                "temperature": temp,
                "threshold": thr,
                "coverage": len(covered) / len(after),
                "accuracy_when_sure": (sum(covered) / len(covered)) if covered else None,
                "ece_before": calibrate.ece([c for c, _ in before], [k for _, k in before]),
                "ece_after": calibrate.ece([c for c, _ in after], [k for _, k in after]),
                "confused": _confused(f.labels, pairs),
            }
        )
    return calib, table


def _pairs(f: Field, rows: list[tuple[dict[str, Any], dict[str, Distribution]]]) -> list[Pair]:
    pairs = []
    for r, g in rows:
        want = g.get(f.name)
        dist = (r["student_dists"] or {}).get(f.name)
        if want and dist and set(dist) == set(f.labels):
            pairs.append((r["id"], [dist[k] for k in f.labels], [want[k] for k in f.labels]))
    return pairs


def _logit(p: list[float]) -> list[float]:
    return [math.log(max(v, 1e-12)) for v in p]


def _confused(labels: tuple[str, ...], pairs: list[Pair]) -> str | None:
    counts: dict[tuple[str, str], int] = {}
    wrong = 0
    for _, pred, want in pairs:
        a, b = labels[argmax(want)], labels[argmax(pred)]
        if a != b:
            wrong += 1
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    (a, b), c = max(counts.items(), key=lambda kv: kv[1])
    return "%s/%s (%d of %d errors)" % (a, b, c, wrong)
