"""`model.evaluate(data)`: held-out numbers per field and a go/no-go for putting the model behind the harness."""

from __future__ import annotations

import os
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


def short_name(name: str) -> str:
    """`laya:/home/me/models/ticket-v2` -> `laya:ticket-v2`, so reports don't carry local paths."""
    kind, sep, rest = name.partition(":")
    if sep and ("/" in rest or "\\" in rest):
        return "%s:%s" % (kind, os.path.basename(os.path.normpath(rest)))
    return name


def training_hashes(model: Model) -> set[str]:
    """Fingerprints of the texts the model was trained on, from its checkpoint's provenance (empty if unknown)."""
    import json

    folders = [model.trained, model.path, getattr(model.engine, "model_id", None)]
    for folder in [f for f in folders if isinstance(f, str)]:
        cfg = os.path.join(folder, "rl_agent_config.json")
        if os.path.exists(cfg):
            with open(cfg, encoding="utf-8") as f:
                return set((json.load(f).get("decisionsmith") or {}).get("text_hashes") or [])
    return set()


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
        pick = [int(data_mod.text_hash(r.text)[:8], 16) % 2 == 0 for _, r in pairs]
        separate = any(pick) and not all(pick)
        chosen_on = [(c, k) for c, k, x in zip(conf, right, pick) if x or not separate]
        reported_on = [(c, k) for c, k, x in zip(conf, right, pick) if not x or not separate]
        thr = calibrate.threshold([c for c, _ in chosen_on], [k for _, k in chosen_on], target)
        sure = [k for c, k in reported_on if thr is not None and c >= thr]
        confusions = Counter(
            "%s -> %s" % (f.labels[argmax(g)], f.labels[argmax(p)]) for p, g, k in zip(pred, gold, right) if not k
        )
        m.update(
            {
                "threshold": thr,
                "coverage": len(sure) / len(reported_on),
                "accuracy_when_sure": sum(sure) / len(sure) if sure else None,
                "threshold_rows": len(chosen_on),
                "coverage_rows": len(reported_on),
                "confusions": dict(confusions.most_common()),
            }
        )
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
                "accuracy_when_sure": m["accuracy_when_sure"],
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
    seen = training_hashes(model)
    overlap = sum(data_mod.text_hash(r.text) in seen for r in rows)
    if overlap:
        reasons.insert(
            0,
            "%d of %d test texts were in the training data, so these numbers are optimistic; evaluate on rows the "
            "model never saw (a golden.csv keeps them apart with split=test)" % (overlap, len(rows)),
        )
    go = not reasons
    worst.sort(key=lambda w: -w["confidence"])
    details = {
        "overlap": overlap,
        "fields": fields,
        "worst": worst[:10],
        "ms_per_text": ms,
        "target": target,
        "rows": len(rows),
    }
    title = "evaluate: %s on %d rows" % (short_name(model.name), len(rows))
    report = Report("evaluate", title, table, go=go, reasons=reasons or ["ready for the harness"], details=details)
    model.report = report
    return report
