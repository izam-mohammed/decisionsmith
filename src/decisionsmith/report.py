"""`Report`: the one result type for finetune, adapt and bench, plus the shared metrics."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

from .files import parent
from .schema import argmax
from .training.calibrate import ece

CURVE = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between the closest ranks, like `numpy.percentile`."""
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def metrics(pred: Sequence[Any], gold: Sequence[Any], ordinal: bool = False) -> dict[str, Any]:
    """Accuracy, macro-F1, ECE, Brier and a coverage/accuracy curve for one field."""
    n = len(pred)
    if n == 0:
        return {"n": 0}
    p = [[float(v) for v in x] for x in pred]
    g = [[float(v) for v in x] for x in gold]
    pi, gi = [argmax(x) for x in p], [argmax(x) for x in g]
    right = [float(a == b) for a, b in zip(pi, gi)]
    conf = [max(x) for x in p]
    f1s = []
    for c in sorted(set(pi) | set(gi)):
        tp = sum(a == c and b == c for a, b in zip(pi, gi))
        wrong = sum((a == c) != (b == c) for a, b in zip(pi, gi))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + wrong))
    out: dict[str, Any] = {
        "n": n,
        "accuracy": mean(right),
        "macro_f1": mean(f1s),
        "ece": ece(conf, right),
        "brier": mean(sum((u - v) ** 2 for u, v in zip(a, b)) for a, b in zip(p, g)),
        "curve": [],
    }
    for t in CURVE:
        sure = [k for c, k in zip(conf, right) if c >= t]
        out["curve"].append({"threshold": t, "coverage": len(sure) / n, "accuracy": mean(sure) if sure else None})
    if ordinal:
        out["mae"] = mean(abs(_expected(a) - _expected(b)) for a, b in zip(p, g))
    return out


def _expected(dist: list[float]) -> float:
    return sum(i * v for i, v in enumerate(dist))


def _fmt(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return "-" if v is None else ("yes" if v else "no")
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return "%.3f" % v if abs(v) < 10 else "%.1f" % v
    return str(v)


class Report:
    """What a finetune, adapt or bench run found. `print(report)` for the table, `.to_dict()` for JSON."""

    def __init__(
        self,
        kind: str,
        title: str,
        rows: list[dict[str, Any]],
        *,
        go: bool | None = None,
        reasons: Sequence[str] = (),
        path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.kind, self.title, self.rows = kind, title, rows
        self.go, self.reasons, self.path = go, list(reasons), path
        self.details = details or {}

    @property
    def columns(self) -> list[str]:
        cols: list[str] = []
        for r in self.rows:
            cols.extend(k for k in r if k not in cols)
        return cols

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "go": self.go,
            "reasons": self.reasons,
            "path": self.path,
            "rows": self.rows,
            "details": self.details,
        }

    def __str__(self) -> str:
        cols = self.columns
        cells = [[_fmt(r.get(c)) for c in cols] for r in self.rows]
        widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
        lines = [self.title, ""]
        if cols:
            lines.append("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
            lines.append("  ".join("-" * w for w in widths))
            lines.extend("  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells)
        if self.go is not None:
            lines += ["", "go: %s" % ("yes" if self.go else "no")]
        lines.extend("  - %s" % r for r in self.reasons)
        if self.path:
            lines.append("saved: %s" % self.path)
        return "\n".join(lines)

    __repr__ = __str__

    def html(self) -> str:
        cols = self.columns
        head = "".join("<th>%s</th>" % html.escape(c) for c in cols)
        body = "".join(
            "<tr>%s</tr>" % "".join("<td>%s</td>" % html.escape(_fmt(r.get(c))) for c in cols) for r in self.rows
        )
        verdict = "" if self.go is None else "<p><b>go: %s</b></p>" % ("yes" if self.go else "no")
        reasons = "".join("<li>%s</li>" % html.escape(r) for r in self.reasons)
        return (
            "<!doctype html><meta charset=utf-8><title>%s</title>"
            "<style>body{font:14px system-ui;margin:2em;max-width:70em}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}</style>"
            "<h1>%s</h1><table><tr>%s</tr>%s</table>%s<ul>%s</ul>"
            % (html.escape(self.title), html.escape(self.title), head, body, verdict, reasons)
        )

    def save(self, path: str) -> str:
        with open(parent(path), "w", encoding="utf-8") as f:
            if path.endswith(".html"):
                f.write(self.html())
            else:
                json.dump(self.to_dict(), f, indent=2, default=str)
        return path
