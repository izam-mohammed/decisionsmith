"""The harness core (`ds.harness`): teacher + student behind one schema, routed per field, every decision logged."""

from __future__ import annotations

import logging
import os
import random
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Generic, Literal, TypeVar, cast

from pydantic import BaseModel

from .engines import Engine, EngineError, LayaEngine, ask_many, from_string, response
from .files import next_run
from .log import Log
from .report import Report
from .schema import Distribution, Schema, compile_schema, confidence, top
from .status import Status, compute_status
from .training import adapt as adapting
from .training.export import export, gold, gold_rows

T = TypeVar("T", bound=BaseModel)
Mode = Literal["teacher", "shadow", "cascade", "student"]
MODES = ("teacher", "shadow", "cascade", "student")
MIN_FINETUNE_ROWS = 50

logger = logging.getLogger("decisionsmith")


class Result(BaseModel, Generic[T]):
    """One decision: the typed value plus where each field came from and how sure it is."""

    id: str | None
    value: T
    source: dict[str, str]
    confidence: dict[str, float]
    probabilities: dict[str, dict[str, float]]
    sure: bool
    latency_ms: float


class _Answers:
    def __init__(self) -> None:
        self.dists: dict[str, Distribution] = {}
        self.error: str | None = None


class Harness(Generic[T]):
    """Use `ds.harness(...)`. Call it with text to get the typed decision."""

    def __init__(
        self,
        schema: type[T],
        teacher: Any = None,
        student: Any = None,
        mode: Mode | dict[str, Mode] | None = None,
        threshold: float = 0.8,
        log: str | os.PathLike[str] | None = "decisions.db",
        audit: float = 0.05,
    ) -> None:
        from .predictor import Model

        self.simple = False
        if isinstance(schema, Model):
            student = schema if student is None else student
            self.simple = schema.simple
            schema = schema.schema.model
        if isinstance(student, Model):
            student = student.engine
        self.schema: Schema = compile_schema(schema)
        self.teacher: Engine | None = from_string(teacher)
        self.student: Engine | None = from_string(student)
        if self.teacher is None and self.student is None:
            raise ValueError("give a teacher, a student, or both: ds.harness(Ticket, teacher='claude-sonnet-5')")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1], got %r" % threshold)
        if not 0.0 <= audit <= 1.0:
            raise ValueError("audit must be in [0, 1], got %r" % audit)
        self.threshold, self.audit = threshold, audit
        self.modes: dict[str, Mode] = self._modes(mode)
        self.log = Log(log) if log is not None else None
        self._pool: ThreadPoolExecutor | None = None
        self._random = random.Random()
        self._load_adapt()

    def _modes(self, mode: Mode | dict[str, Mode] | None) -> dict[str, Mode]:
        default: Mode = "cascade" if self.teacher and self.student else "teacher" if self.teacher else "student"
        if isinstance(mode, dict):
            unknown = set(mode) - set(self.schema.fields)
            if unknown:
                raise ValueError(
                    "mode names unknown fields %s; fields: %s" % (sorted(unknown), list(self.schema.fields))
                )
            modes = {n: mode.get(n, default) for n in self.schema.fields}
        else:
            modes = {n: mode or default for n in self.schema.fields}
        for name, m in modes.items():
            if m not in MODES:
                raise ValueError("mode for %r is %r; use one of %s" % (name, m, MODES))
            if m in ("teacher", "shadow", "cascade") and self.teacher is None:
                raise ValueError("mode %r for %r needs a teacher" % (m, name))
            if m in ("shadow", "cascade", "student") and self.student is None:
                raise ValueError("mode %r for %r needs a student" % (m, name))
        return modes

    def _adapt_key(self) -> str:
        return adapting.adapt_key(self.schema.name, self.student.name if self.student else None)

    def _load_adapt(self) -> None:
        self._calib: dict[str, dict[str, Any]] = (self.log.get(self._adapt_key(), {}) if self.log else {}) or {}

    def _threshold(self, name: str) -> float:
        return adapting.threshold_for(self._calib, name, self.threshold)

    def _calibrated(self, name: str, dist: Distribution) -> Distribution:
        return adapting.calibrated(self._calib, name, dist)

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="decisionsmith")
        return self._pool

    def _ask(self, engine: Engine, text: str, names: list[str], raw: Any = None) -> _Answers:
        out = _Answers()
        questions = self.schema.questions(names)
        try:
            if isinstance(raw, BaseException):
                raise raw
            answers, _ = response(engine, engine.ask(text, questions) if raw is None else raw, questions)
            out.dists = self.schema.distributions(answers, names)
        except Exception as e:
            out.error = str(e) if isinstance(e, EngineError) else "%s: %s" % (type(e).__name__, e)
            logger.warning("engine %s failed: %s", engine.name, out.error)
        return out

    def __call__(self, text: str) -> Any:
        return self._plain(self.decide(text).value)

    def _plain(self, value: Any) -> Any:
        return value.label if self.simple else value

    def decide(self, text: str) -> Result[T]:
        """Decide one text. The result says which engine answered each field and how sure it is."""
        return self._decide(text, None)

    def many(self, texts: list[str]) -> list[Any]:
        """Decide many texts; the student runs batched, teacher calls run in parallel. Order is kept."""
        texts = list(texts)
        student_fields = [n for n, m in self.modes.items() if m != "teacher"]
        pre: list[Any] = [None] * len(texts)
        if self.student is not None and student_fields and texts:
            try:
                pre = ask_many(self.student, texts, self.schema.questions(student_fields))
            except Exception as e:
                pre = [e] * len(texts)
        results = self._executor().map(lambda tp: self._decide(tp[0], tp[1], batched=True), zip(texts, pre))
        return [self._plain(r.value) for r in results]

    def _decide(self, text: str, pre: Any, batched: bool = False) -> Result[T]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        t0 = time.perf_counter()
        modes = self.modes
        student_fields = [n for n, m in modes.items() if m != "teacher"]
        first_teacher = [n for n, m in modes.items() if m in ("teacher", "shadow")]

        s = _Answers()
        t = _Answers()
        pending: Future[_Answers] | None = None
        if first_teacher and student_fields and not batched:
            assert self.teacher is not None
            pending = self._executor().submit(self._ask, self.teacher, text, first_teacher)
        elif first_teacher:
            assert self.teacher is not None
            t = self._ask(self.teacher, text, first_teacher)
        if student_fields:
            assert self.student is not None
            s = self._ask(self.student, text, student_fields, pre if batched else None)
        if pending is not None:
            t = pending.result()

        adapted = {n: self._calibrated(n, d) for n, d in s.dists.items()}
        sure = {n: confidence(d) >= self._threshold(n) for n, d in adapted.items()}
        more = [n for n, m in modes.items() if m == "cascade" and not sure.get(n, False)]
        more += [n for n, m in modes.items() if m == "student" and n not in adapted]
        confident = [n for n, m in modes.items() if m == "cascade" and sure.get(n, False)]
        if confident and self.audit and self._random.random() < self.audit:
            more += confident
        more = [n for n in more if n not in t.dists]
        if more and self.teacher is not None:
            extra = self._ask(self.teacher, text, more)
            t.dists.update(extra.dists)
            t.error = t.error or extra.error

        value: dict[str, Distribution] = {}
        source: dict[str, str] = {}
        unsure = False
        for n, m in modes.items():
            use_student = m == "student" or (m == "cascade" and sure.get(n, False))
            if use_student and n in adapted:
                value[n], source[n] = adapted[n], "student"
                unsure = unsure or not sure[n]
            elif n in t.dists:
                value[n], source[n] = t.dists[n], "teacher"
            elif n in adapted:
                value[n], source[n] = adapted[n], "student"
                unsure = True
            else:
                raise EngineError(
                    "+".join(e.name for e in (self.student, self.teacher) if e),
                    "no engine could answer %r (student: %s; teacher: %s)" % (n, s.error or "-", t.error or "-"),
                    "run `decisionsmith doctor` to check the engines",
                )

        obj = cast(T, self.schema.build(value))
        latency = (time.perf_counter() - t0) * 1000
        decision_id = uuid.uuid4().hex if self.log else None
        if self.log is not None:
            self.log.add(
                {
                    "id": decision_id,
                    "schema": self.schema.name,
                    "text": text,
                    "value": {n: top(d) for n, d in value.items()},
                    "source": source,
                    "teacher": self.teacher.name if self.teacher else None,
                    "student": self.student.name if self.student else None,
                    "teacher_dists": t.dists or None,
                    "student_dists": s.dists or None,
                    "latency_ms": latency,
                }
            )
        return Result(
            id=decision_id,
            value=obj,
            source=source,
            confidence={n: confidence(d) for n, d in value.items()},
            probabilities=value,
            sure=not unsure,
            latency_ms=latency,
        )

    def _require_log(self) -> Log:
        if self.log is None:
            raise ValueError("this harness has log=None; give it a log path to use this")
        return self.log

    def label(self, decision_id: str, **fields: Any) -> None:
        """Record the correct answer for a logged decision: `h.label(r.id, team="technical")`."""
        if not fields:
            raise ValueError("give at least one field, e.g. h.label(r.id, team='technical')")
        labels = {n: self.schema.label_of(n, v) for n, v in fields.items()}
        self._require_log().label(decision_id, labels)

    def status(self) -> Status:
        """Per field: how often the student agrees, how sure it is, how accurate when sure, and what to do next."""
        rows = self._require_log().rows(self.schema.name)
        return compute_status(
            self.schema, rows, self.modes, self.student.name if self.student else None, self._threshold
        )

    def export(self, path: str, format: Literal["answers", "typed-decisions"] = "answers") -> int:
        """Write every decision with a label (human first, else teacher) as training JSONL. Returns rows written."""
        return export(self.schema, self._require_log(), path, format)

    def adapt(self, target: float = 0.97) -> Report:
        """Fit per-field calibration and cascade thresholds from the log. Changes confidence, never answers."""
        log = self._require_log()
        if self.student is None:
            raise ValueError("adapt() calibrates the student; this harness has none")
        trained = log.trained()
        rows = [
            (r, gold(self.schema, r))
            for r in log.rows(self.schema.name)
            if r["student"] == self.student.name and r["id"] not in trained
        ]
        calib, table = adapting.fit(self.schema, rows, self._calib, target)
        log.set(self._adapt_key(), calib)
        self._calib = calib
        reasons = [
            "thresholds target %.0f%% accuracy on held-out rows; no threshold means the field always "
            "asks the teacher" % (target * 100)
        ]
        return Report(
            "adapt",
            "adapt: %s (student %s)" % (self.schema.name, self.student.name),
            table,
            reasons=reasons,
        )

    def finetune(self, out: str | None = None, **options: Any) -> Report:
        """Fine-tune the Laya student on the log (human labels first, else teacher answers).

        The harness switches to the new checkpoint only if it passes go/no-go and beats the current student.
        Modes never change.
        """
        from .training.finetuning import finetune

        if self.student is not None and not isinstance(self.student, LayaEngine):
            raise EngineError(
                self.student.name,
                "only Laya students can be fine-tuned",
                "use h.adapt() for this engine, or student='laya'",
            )
        rows = gold_rows(self.schema, self._require_log())
        if len(rows) < MIN_FINETUNE_ROWS:
            raise ValueError(
                "finetune needs at least %d labelled decisions in the log, have %d; keep running in "
                "teacher or shadow mode" % (MIN_FINETUNE_ROWS, len(rows))
            )
        base = self.student.spec if isinstance(self.student, LayaEngine) else "laya"
        out = out or self._next_run()
        report = finetune(rows, self.schema.model, base=base, out=out, **options)
        self._require_log().mark_trained(report.details.get("train_ids", []), out)
        if report.go:
            device = self.student.device if isinstance(self.student, LayaEngine) else None
            self.student = LayaEngine(out, device=device)
            self._load_adapt()
            report.reasons.append("the harness now uses student='laya:%s'; pass that next time you build it" % out)
            if all(m == "teacher" for m in self.modes.values()):
                report.reasons.append("all fields are in teacher mode; use mode='shadow' to start measuring it")
        else:
            report.reasons.append("kept the current student")
        return report

    def _next_run(self) -> str:
        return next_run(self.schema.name, os.path.dirname(os.path.abspath(self.log.path)) if self.log else os.getcwd())

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self.log is not None:
            self.log.close()

    def __enter__(self) -> Harness[T]:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def harness(
    schema: Any,
    teacher: Any = None,
    student: Any = None,
    mode: Mode | dict[str, Mode] | None = None,
    threshold: float = 0.8,
    log: str | os.PathLike[str] | None = "decisions.db",
    audit: float = 0.05,
) -> Harness[T]:
    """Connect a teacher (any LLM, `ds.LLM(...)`, or Jev) and a student (Laya, or Jev) behind one schema.

    h = ds.harness(Ticket, teacher="claude-sonnet-5", student="laya")
    h("You charged me twice, refund now!")   # -> Ticket(...)

    model = ds.model(["billing", "technical", "sales"])   # base or trained Laya
    h = ds.harness(model, teacher="claude-haiku-4-5")       # it becomes the student; h(text) -> "billing"
    """
    return Harness(schema, teacher=teacher, student=student, mode=mode, threshold=threshold, log=log, audit=audit)
