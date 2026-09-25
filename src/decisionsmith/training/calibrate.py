"""Temperature scaling, ECE and confidence thresholds in plain Python, so `adapt()` needs no numpy or torch."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

TEMP_MIN, TEMP_MAX = 0.5, 5.0


def softmax(z: Sequence[float], t: float = 1.0) -> list[float]:
    m = max(z)
    e = [math.exp((float(v) - m) / t) for v in z]
    total = sum(e)
    return [v / total for v in e]


def _log_softmax(z: list[float]) -> list[float]:
    m = max(z)
    log_total = math.log(sum(math.exp(v - m) for v in z))
    return [v - m - log_total for v in z]


def nll(logits: Sequence[Any], targets: Sequence[Any], t: float) -> float:
    losses = [-sum(g * lp for g, lp in zip(tg, _log_softmax([v / t for v in z]))) for z, tg in zip(logits, targets)]
    return sum(losses) / len(losses)


def fit_temperature(
    logits: Sequence[Any],
    targets: Sequence[Any],
    lo: float = TEMP_MIN,
    hi: float = TEMP_MAX,
) -> float:
    """The temperature minimising NLL (convex in 1/T), by golden-section search on log T, within [lo, hi]."""
    zs = [[float(v) for v in z] for z in logits]
    ts = [[float(v) for v in t] for t in targets]
    if not zs:
        return 1.0
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(zs, ts, math.exp(c)), nll(zs, ts, math.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(zs, ts, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(zs, ts, math.exp(d))
    return float(min(hi, max(lo, math.exp((a + b) / 2))))


def scale(probs: Sequence[Any], t: float) -> list[float]:
    """Apply a temperature to probabilities (as log-probability logits)."""
    return softmax([math.log(min(1.0, max(1e-12, float(p)))) / t for p in probs])


def ece(conf: Any, correct: Any, bins: int = 15) -> float:
    """Expected calibration error; same binning as `laya.common.ece_score`."""
    c, k = [float(v) for v in conf], [float(v) for v in correct]
    if not c:
        return float("nan")
    step = 1.0 / bins
    edges = [i * step for i in range(bins)] + [1.0]
    e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = [j for j, v in enumerate(c) if (v >= lo if i == 0 else v > lo) and v <= hi]
        if sel:
            e += len(sel) / len(c) * abs(sum(c[j] for j in sel) / len(sel) - sum(k[j] for j in sel) / len(sel))
    return float(e)


def threshold(conf: Sequence[Any], correct: Sequence[Any], target: float, min_rows: int = 20) -> float | None:
    """Smallest confidence threshold whose answers at or above it are at least `target` accurate."""
    pairs = sorted(zip(conf, correct), key=lambda p: -p[0])
    best: float | None = None
    n = right = 0
    i = 0
    while i < len(pairs):
        value = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == value:
            n += 1
            right += int(bool(pairs[i][1]))
            i += 1
        if n >= min_rows and right / n >= target:
            best = float(value)
    return best
