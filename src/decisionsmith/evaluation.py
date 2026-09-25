"""`model.evaluate(data)`: held-out numbers per field and a go/no-go for putting the model behind the harness."""

from __future__ import annotations

import time
from collections import Counter
from typing import TYPE_CHECKING, Any

from .engines import response
from .report import Report, metrics
from .schema import argmax
from .training import adapt as adapting
from .training import calibrate
from .training import data as data_mod

if TYPE_CHECKING:
    from .predictor import Model

MIN_DECISIONS, MIN_PER_OPTION, MAX_ECE = 100, 10, 0.10


def evaluate(model: Model, data: Any, target: float = 0.97) -> Report:
    schema = model.schema
    rows = data_mod.load(model._rows(data), schema, split="test")
    if not rows:
        raise ValueError("no labelled rows to evaluate; give text plus a label per field")
    questions = schema.questions()
    t0 = time.perf_counter()
    raws = model.ask_many([str(r.text) for r in rows], questions)
    ms = (time.perf_counter() - t0) * 1000 / len(rows)
    preds = []
    for raw in raws:
        answers, _ = response(model, raw, questions)
        preds.append(
            {n: adapting.calibrated(model.calibration, n, d) for n, d in schema.distributions(answers).items()}
        )

    table: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    reasons: list[str] = []
    fields: dict[str, Any] = {}
    for name, f in schema.fields.items():
        pairs = [(p[name], r) for p, r in zip(preds, rows) if name in r.targets]
        pred = [[d[k] for k in f.labels] for d, _ in pairs]
        gold = [r.targets[name] for _, r in pairs]
        m = metrics(pred, gold, ordinal=f.kind == "scale")
        if not pairs:
            table.append({"field": name, "decisions": 0})
            reasons.append("%s: no labelled rows for this field" % name)
            continue
        conf = [max(p) for p in pred]
        right = [argmax(p) == argmax(g) for p, g in zip(pred, gold)]
        thr = calibrate.threshold(conf, right, target)
        sure = [k for c, k in zip(conf, right) if thr is not None and c >= thr]
        confusions = Counter(
            "%s -> %s" % (f.labels[argmax(g)], f.labels[argmax(p)]) for p, g, k in zip(pred, gold, right) if not k
        )
        m.update({"threshold": thr, "coverage": len(sure) / len(pairs), "confusions": dict(confusions.most_common())})
        fields[name] = m
        c = model.calibration.setdefault(name, {})
        c["threshold"] = thr
        worst += [
            {
                "field": name,
                "text": str(r.text)[:200],
                "gold": f.labels[argmax(g)],
                "pred": f.labels[argmax(p)],
                "confidence": max(p),
            }
            for p, g, k, (_, r) in zip(pred, gold, right, pairs)
            if not k
        ]
        table.append(
            {
                "field": name,
                "decisions": m["n"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "ece": m["ece"],
                "threshold": thr,
                "coverage": m["coverage"],
            }
        )
        counts = Counter(f.labels[argmax(g)] for g in gold)
        low = [k for k in f.labels if counts[k] < MIN_PER_OPTION]
        if low:
            reasons.append("%s: options %s have fewer than %d test rows" % (name, low, MIN_PER_OPTION))
        if m["ece"] > MAX_ECE:
            reasons.append("%s: calibration error %.3f is above %.2f" % (name, m["ece"], MAX_ECE))
        if thr is None:
            reasons.append(
                "%s: no confidence level reaches %.0f%% accuracy, so the harness would send every %s decision to "
                "the teacher" % (name, target * 100, name)
            )
    total = sum(r.get("decisions", 0) for r in table)
    if total < MIN_DECISIONS:
        reasons.insert(0, "only %d test decisions; want at least %d (add data)" % (total, MIN_DECISIONS))
    go = not reasons
    worst.sort(key=lambda w: -w["confidence"])
    details = {"fields": fields, "worst": worst[:10], "ms_per_text": ms, "target": target, "rows": len(rows)}
    title = "evaluate: %s on %d rows" % (model.name, len(rows))
    report = Report("evaluate", title, table, go=go, reasons=reasons or ["ready for the harness"], details=details)
    model.report = report
    return report
