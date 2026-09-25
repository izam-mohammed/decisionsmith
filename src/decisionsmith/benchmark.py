"""`ds.bench`: every engine on the same labelled data, per field: accuracy, calibration, latency, cost."""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .engines import Engine, EngineError, JevEngine, LayaEngine, from_string, response
from .report import Report, metrics, percentile
from .schema import argmax, compile_schema
from .testing import FakeEngine
from .training import data as data_mod

# Jev list price per 1M input tokens (output free) as published by TypeSafe, recorded 23/09/2026; check current pricing.
JEV_USD_PER_M_INPUT = 0.042


def _cost(engine: Engine, usage: dict[str, Any]) -> float | None:
    if isinstance(usage.get("cost_usd"), (int, float)):
        return float(usage["cost_usd"])
    if isinstance(engine, JevEngine) and isinstance(usage.get("input_tokens"), (int, float)):
        return float(usage["input_tokens"]) * JEV_USD_PER_M_INPUT / 1e6
    if isinstance(engine, (LayaEngine, FakeEngine)):
        return 0.0
    return None


def _run(engine: Engine, texts: list[str], questions: dict[str, Any]) -> list[tuple[Any, float]]:
    if callable(getattr(engine, "ask_many", None)):
        out: list[tuple[Any, float]] = []
        for i in range(0, len(texts), 32):
            chunk = texts[i : i + 32]
            t0 = time.perf_counter()
            try:
                answers: Sequence[Any] = engine.ask_many(chunk, questions)  # type: ignore[attr-defined]
            except Exception as e:
                answers = [e] * len(chunk)
            ms = (time.perf_counter() - t0) * 1000 / len(chunk)
            out.extend((a, ms) for a in answers)
        return out

    def one(text: str) -> tuple[Any, float]:
        t0 = time.perf_counter()
        try:
            return engine.ask(text, questions), (time.perf_counter() - t0) * 1000
        except Exception as e:
            return e, (time.perf_counter() - t0) * 1000

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, texts))


def bench(
    schema: Any,
    data: Any,
    engines: Any,
    *,
    threshold: float = 0.8,
    limit: int | None = None,
    out: str | None = None,
) -> Report:
    """Run each engine zero-shot on labelled data (CSV/JSONL) and compare them field by field.

    With a `split` column (a golden dataset) only the `split=test` rows are used, so a model trained on the rest is
    compared fairly; without one, every row is used.
    """
    compiled = compile_schema(schema)
    rows = [r for r in data_mod.load(data, compiled, split="test") if isinstance(r.text, str)]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no labelled rows to bench on")
    specs = [e.strip() for e in engines.split(",")] if isinstance(engines, str) else list(engines)
    questions = compiled.questions()
    table: list[dict[str, Any]] = []
    details: dict[str, Any] = {"rows": len(rows), "engines": {}}
    for spec in specs:
        try:
            engine = from_string(spec)
        except (EngineError, ValueError, TypeError) as e:
            table.append({"engine": str(spec), "field": "-", "error": str(e).splitlines()[0]})
            continue
        results = _run(engine, [r.text for r in rows], questions)
        preds: dict[str, list[tuple[list[float], list[float]]]] = {n: [] for n in compiled.fields}
        errors, costs, latency = [], [], []
        for row, (raw, ms) in zip(rows, results):
            latency.append(ms)
            try:
                if isinstance(raw, BaseException):
                    raise raw
                answers, usage = response(engine, raw, questions)
                costs.append(_cost(engine, usage))
                for name, target in row.targets.items():
                    dist = compiled.distribution(name, answers[name])
                    preds[name].append(([dist[k] for k in compiled.fields[name].labels], target))
            except Exception as e:
                errors.append(str(e).splitlines()[0])
        known = [c for c in costs if c is not None]
        cost_1k = 1000 * sum(known) / len(known) if known and len(known) == len(costs) else None
        p50, p95 = percentile(latency, 50), percentile(latency, 95)
        details["engines"][engine.name] = {"errors": len(errors), "first_error": errors[0] if errors else None}
        for name, pairs in preds.items():
            m = metrics([p for p, _ in pairs], [g for _, g in pairs], ordinal=compiled.fields[name].kind == "scale")
            sure = [float(argmax(p) == argmax(g)) for p, g in pairs if max(p) >= threshold]
            table.append(
                {
                    "engine": engine.name,
                    "field": name,
                    "n": m["n"],
                    "accuracy": m.get("accuracy"),
                    "macro_f1": m.get("macro_f1"),
                    "ece": m.get("ece"),
                    "coverage": len(sure) / m["n"] if m["n"] else None,
                    "acc_when_sure": sum(sure) / len(sure) if sure else None,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "cost_per_1k": cost_1k,
                    "errors": len(errors),
                }
            )
            details["engines"][engine.name][name] = m
    reasons = [
        "zero-shot on %d rows; coverage = share of answers with confidence >= %.2f" % (len(rows), threshold),
        "cost is USD per 1k decisions from reported usage; laya and fake count compute as 0",
    ]
    report = Report("bench", "bench: %s" % compiled.name, table, reasons=reasons, path=out, details=details)
    if out:
        report.save(out)
    return report
