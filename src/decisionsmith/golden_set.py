"""`ds.golden`: pick the texts most worth labelling, have the main LLM (or a coding agent) label them, write a CSV."""

from __future__ import annotations

import csv
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .engines import EngineError, from_string, response
from .files import parent
from .schema import Distribution, Schema, compile_schema, confidence, top
from .training.data import safe_cell, same_text, text_hash

STRATEGIES = ("uncertain", "disagree", "diverse", "random")
GIVE_UP_AFTER = 5


class _Candidate:
    def __init__(
        self,
        text: str,
        student: dict[str, Distribution] | None,
        teacher: dict[str, Distribution] | None,
        human: dict[str, str] | None = None,
        trained: bool = False,
    ) -> None:
        self.text, self.student, self.teacher = text, student, teacher
        self.human, self.trained = human or {}, trained

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
    if isinstance(source, (str, os.PathLike)) and os.fspath(source).lower().endswith(".db"):
        log_path = os.fspath(source)
        if not os.path.exists(log_path):
            raise FileNotFoundError("no log at %s; run a harness with log=%r first" % (log_path, log_path))
    if log_path is not None:
        with Log(log_path) as log:
            rows = [r for r in log.rows(schema.name) if r["text"]]
            trained = log.trained()
        if not rows:
            raise ValueError(
                "%s has no %s decisions with text; the harness keeps text for collected rows (collect=...)"
                % (log_path, schema.name)
            )
        return [
            _Candidate(r["text"], r["student_dists"], r["teacher_dists"], r["labels"], r["id"] in trained) for r in rows
        ]
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


def _position(text: str) -> float:
    return int(text_hash(text)[:8], 16) / 0xFFFFFFFF


def in_test(text: str, share: float) -> bool:
    """Whether a text belongs to the held-out test split: a stable function of the text, the same in every run."""
    return _position(text) < share


def _held(items: list[_Candidate], test: float) -> set[int]:
    """Which of `items` are held out for evaluation: by text hash, at least one when there are two or more."""
    held = {i for i, c in enumerate(items) if not c.trained and in_test(c.text, test)}
    eligible = [i for i, c in enumerate(items) if not c.trained]
    if test > 0 and len(items) >= 2 and not held and eligible:
        held = {min(eligible, key=lambda i: _position(items[i].text))}
    return held


def _check_out(out: str | None, overwrite: bool) -> None:
    if out is None:
        return
    if not out.lower().endswith((".csv", ".jsonl")):
        raise ValueError("out must end in .csv (easy to review) or .jsonl, got %r" % out)
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(
            "%s already exists and may hold your review; pass overwrite=True (CLI: --overwrite) or another out" % out
        )


def _label_all(
    chosen: list[_Candidate], engine: Any, schema: Schema
) -> tuple[list[tuple[_Candidate, dict[str, Distribution], str]], int, str | None]:
    questions = schema.questions()
    name = "llm:%s" % engine.name

    def one(c: _Candidate) -> tuple[dict[str, Distribution] | None, str | None]:
        human = {k: v for k, v in c.human.items() if k in schema.fields}
        if human and set(human) == set(schema.fields):
            return {n: {k: float(k == v) for k in schema.fields[n].labels} for n, v in human.items()}, "human"
        try:
            got = schema.distributions(response(engine, engine.ask(c.text, questions), questions)[0])
        except Exception as e:
            return None, e.message if isinstance(e, EngineError) else "%s: %s" % (type(e).__name__, e)
        for n, v in human.items():
            got[n] = {k: float(k == v) for k in schema.fields[n].labels}
        return got, name + ("+human" if human else "")

    done: list[tuple[_Candidate, dict[str, Distribution], str]] = []
    failed, first_error = 0, None
    starts = [0, *range(GIVE_UP_AFTER, len(chosen), 8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, start in enumerate(starts):
            batch = chosen[start : starts[i + 1] if i + 1 < len(starts) else len(chosen)]
            for c, (answers, by) in zip(batch, pool.map(one, batch)):
                if answers is None:
                    failed += 1
                    first_error = first_error or by
                else:
                    done.append((c, answers, str(by)))
            if not done and failed >= GIVE_UP_AFTER:
                break
    if not done:
        raise EngineError(
            engine.name,
            "could not label any of the %d texts tried; first error: %s" % (failed, first_error),
            "check the teacher with `decisionsmith doctor --engines %s`" % engine.name,
        )
    return done, failed, first_error


def golden(
    source: Any,
    teacher: Any,
    n: int = 500,
    strategy: str = "uncertain",
    *,
    schema: Any = None,
    test: float = 0.2,
    out: str | None = "golden.csv",
    overwrite: bool = False,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    from . import golden_session

    agent = golden_session.agent_of(teacher)
    engine = None if agent is not None else from_string(teacher)
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be one of %s, got %r" % (STRATEGIES, strategy))
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 <= test < 1.0:
        raise ValueError("test is the share of rows held out for evaluation, in [0, 1); got %r" % test)
    if agent is None:
        _check_out(out, overwrite)
    elif out is None:
        raise ValueError("teacher='agent' writes a labelling session next to out; give out='golden.csv'")
    else:
        _check_out(out, True)
        if os.path.exists(golden_session.session_path(out)) and not overwrite:
            raise FileExistsError(
                "%s already holds a labelling session; finish it (decisionsmith golden --finish %s), or pass "
                "overwrite=True (CLI: --overwrite) to start again" % ((golden_session.session_path(out),) * 2)
            )
    compiled, student = _schema_and_student(schema, source)
    firsts: dict[str, _Candidate] = {}
    for c in _candidates(source, compiled, student):
        firsts.setdefault(same_text(c.text), c)
    pool = list(firsts.values())
    has_student = any(c.student for c in pool)
    if strategy in ("uncertain", "disagree") and not has_student:
        raise ValueError(
            "strategy=%r ranks texts by the student's confidence, and there is none here; use a harness log, "
            "pass schema=ds.model(...) to score the texts, or strategy='random'" % strategy
        )
    rng = random.Random(seed)
    chosen = _choose(pool, n, strategy, rng)
    if engine is None:
        held = _held(chosen, test)
        return golden_session.start(
            [(c.text, c.human, "test" if i in held else "train") for i, c in enumerate(chosen)],
            compiled,
            str(out),
            agent=agent,
            strategy=strategy,
            overwrite=overwrite,
            verbose=verbose,
        )
    labelled, failed, first_error = _label_all(chosen, engine, compiled)
    held = _held([c for c, _, _ in labelled], test)
    rows = [
        {
            "id": "g" + text_hash(c.text)[:12],
            "text": c.text,
            "answers": a,
            "split": "test" if i in held else "train",
            "labelled_by": by,
        }
        for i, (c, a, by) in enumerate(labelled)
    ]
    if out:
        _write(rows, out, compiled)
    if verbose:
        print(
            "golden: %d rows (%s) labelled by %s%s · %d marked split=test%s"
            % (
                len(rows),
                strategy,
                engine.name,
                " · %d could not be labelled (first error: %s)" % (failed, first_error) if failed else "",
                len(held),
                " · wrote %s; review it, then train on it" % out if out else "",
            )
        )
    return rows


def _write(rows: list[dict[str, Any]], path: str, schema: Schema) -> None:
    tmp = parent(path) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        if path.lower().endswith(".jsonl"):
            for r in rows:
                rec = {**r, "answers": {k: top(v) for k, v in r["answers"].items()}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            extra = ["checked"] if any("checked" in r for r in rows) else []
            w = csv.DictWriter(f, fieldnames=["id", "text", *schema.fields, "split", "labelled_by", *extra])
            w.writeheader()
            for r in rows:
                cells = {"id": r["id"], "text": r["text"], **{k: top(v) for k, v in r["answers"].items()}}
                w.writerow(
                    {
                        **{k: safe_cell(v) for k, v in cells.items()},
                        "split": r["split"],
                        "labelled_by": r["labelled_by"],
                        **{k: r.get(k, "") for k in extra},
                    }
                )
    os.replace(tmp, path)
