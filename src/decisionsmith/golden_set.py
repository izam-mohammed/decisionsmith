"""`ds.golden`: pick the texts most worth labelling, have the main LLM label them, and write a reviewable CSV."""

from __future__ import annotations

import csv
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .engines import from_string, response
from .files import parent
from .schema import Distribution, Schema, compile_schema, confidence, top

STRATEGIES = ("uncertain", "disagree", "diverse", "random")


class _Candidate:
    def __init__(self, text: str, student: dict[str, Distribution] | None, teacher: dict[str, Distribution] | None):
        self.text, self.student, self.teacher = text, student, teacher

    def doubt(self) -> float:
        return min((confidence(d) for d in self.student.values()), default=1.0) if self.student else 1.0

    def disagrees(self) -> bool:
        if not self.student or not self.teacher:
            return False
        return any(n in self.teacher and top(d) != top(self.teacher[n]) for n, d in self.student.items())


def _schema_and_student(schema: Any, source: Any) -> tuple[Schema, Any]:
    from .core import Harness
    from .predictor import Model, labels_model

    if isinstance(source, Harness) and schema is None:
        return source.schema, None
    if isinstance(schema, Model):
        return schema.schema, schema
    if isinstance(schema, (list, tuple)):
        return compile_schema(labels_model(list(schema))), None
    if schema is None:
        raise ValueError(
            "ds.golden needs to know the answers: pass schema=ds.model([...]), a list of labels, or your class"
        )
    return compile_schema(schema), None


def _texts(source: Any) -> list[str]:
    if isinstance(source, (str, os.PathLike)):
        from .cli import read_texts

        return read_texts(os.fspath(source))
    texts = [x if isinstance(x, str) else x.get("text", "") for x in source]
    return [t for t in texts if isinstance(t, str) and t.strip()]


def _candidates(source: Any, schema: Schema, student: Any) -> list[_Candidate]:
    from .core import Harness
    from .log import Log

    log_path = source.log.path if isinstance(source, Harness) and source.log else None
    if isinstance(source, Harness) and log_path is None:
        raise ValueError("this harness has log=None, so there are no samples; give it a log path")
    if isinstance(source, (str, os.PathLike)) and os.fspath(source).endswith(".db"):
        log_path = os.fspath(source)
        if not os.path.exists(log_path):
            raise FileNotFoundError("no log at %s; run a harness with log=%r first" % (log_path, log_path))
    if log_path is not None:
        with Log(log_path) as log:
            rows = [r for r in log.rows(schema.name) if r["text"]]
        if not rows:
            raise ValueError(
                "%s has no %s decisions with text; the harness keeps text for collected rows (collect=...)"
                % (log_path, schema.name)
            )
        return [_Candidate(r["text"], r["student_dists"], r["teacher_dists"]) for r in rows]
    texts = _texts(source)
    if not texts:
        raise ValueError("no texts to choose from; give a harness log (.db), a CSV/JSONL/.txt of texts, or a list")
    if student is None:
        return [_Candidate(t, None, None) for t in texts]
    questions = schema.questions()
    dists = [schema.distributions(response(student, raw, questions)[0]) for raw in student.ask_many(texts, questions)]
    return [_Candidate(t, d, None) for t, d in zip(texts, dists)]


def _spread(items: list[_Candidate], k: int) -> list[_Candidate]:
    """All items, ordered so any first `k` of them spread evenly from shortest to longest."""
    by_length = sorted(items, key=lambda c: len(c.text))
    picks = sorted({round(i * (len(by_length) - 1) / max(1, k - 1)) for i in range(min(k, len(by_length)))})
    return [by_length[i] for i in picks] + [c for i, c in enumerate(by_length) if i not in set(picks)]


def _choose(pool: list[_Candidate], n: int, strategy: str, rng: random.Random) -> list[_Candidate]:
    if strategy == "random":
        return rng.sample(pool, min(n, len(pool)))
    if strategy == "uncertain":
        return sorted(pool, key=lambda c: c.doubt())[:n]
    if strategy == "disagree":
        return sorted(pool, key=lambda c: (not c.disagrees(), c.doubt()))[:n]
    buckets: dict[tuple[str, ...], list[_Candidate]] = {}
    for c in pool:
        buckets.setdefault(tuple(top(d) for d in (c.student or {}).values()), []).append(c)
    share = -(-n // len(buckets))
    spread = {key: _spread(items, share) for key, items in sorted(buckets.items())}
    out: list[_Candidate] = []
    while len(out) < min(n, len(pool)):
        out += [items.pop(0) for items in spread.values() if items]
    return out[:n]


def golden(
    source: Any,
    teacher: Any,
    n: int = 500,
    strategy: str = "uncertain",
    *,
    schema: Any = None,
    test: float = 0.2,
    out: str | None = "golden.csv",
    seed: int = 0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    engine = from_string(teacher)
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be one of %s, got %r" % (STRATEGIES, strategy))
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 <= test < 1.0:
        raise ValueError("test is the share of rows held out for evaluation, in [0, 1); got %r" % test)
    compiled, student = _schema_and_student(schema, source)
    firsts: dict[str, _Candidate] = {}
    for c in _candidates(source, compiled, student):
        firsts.setdefault(" ".join(c.text.lower().split()), c)
    pool = list(firsts.values())
    has_student = any(c.student for c in pool)
    if strategy in ("uncertain", "disagree") and not has_student:
        raise ValueError(
            "strategy=%r ranks texts by the student's confidence, and there is none here; use a harness log, "
            "pass schema=ds.model(...) to score the texts, or strategy='random'" % strategy
        )
    rng = random.Random(seed)
    chosen = _choose(pool, n, strategy, rng)
    questions = compiled.questions()

    def label(c: _Candidate) -> dict[str, Distribution] | None:
        try:
            return compiled.distributions(response(engine, engine.ask(c.text, questions), questions)[0])
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool_:
        answers = list(pool_.map(label, chosen))
    labelled = [(c, a) for c, a in zip(chosen, answers) if a is not None]
    order = list(range(len(labelled)))
    rng.shuffle(order)
    held = set(order[: round(test * len(labelled))])
    rows = [
        {
            "id": "g%04d" % i,
            "text": c.text,
            "answers": a,
            "split": "test" if i in held else "train",
            "labelled_by": engine.name,
        }
        for i, (c, a) in enumerate(labelled)
    ]
    if out:
        _write(rows, out, compiled)
    if verbose:
        failed = len(chosen) - len(labelled)
        print(
            "golden: %d rows (%s) labelled by %s%s · %d marked split=test%s"
            % (
                len(rows),
                strategy,
                engine.name,
                " · %d could not be labelled" % failed if failed else "",
                len(held),
                " · wrote %s; review it, then train on it" % out if out else "",
            )
        )
    return rows


def _write(rows: list[dict[str, Any]], path: str, schema: Schema) -> None:
    fields = list(schema.fields)
    with open(parent(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", *fields, "split", "labelled_by"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r["id"],
                    "text": r["text"],
                    **{k: top(v) for k, v in r["answers"].items()},
                    "split": r["split"],
                    "labelled_by": r["labelled_by"],
                }
            )
