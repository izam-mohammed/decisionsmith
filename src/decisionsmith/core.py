"""The harness core (`ds.harness`): teacher + student behind one schema, routed per field, every decision logged."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
import uuid
import warnings
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
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
DEFAULT_THRESHOLD = 0.8
Job = tuple[Engine, str, list[str], Any]

logger = logging.getLogger("decisionsmith")


def sampled(decision_id: str, share: float) -> bool:
    """Whether `collect=share` keeps this decision's text by chance: a stable function of the id, not a random draw."""
    return int(hashlib.sha256(decision_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


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
        threshold: float | None = None,
        log: str | os.PathLike[str] | None = "decisions.db",
        audit: float = 0.05,
        collect: float = 1.0,
    ) -> None:
        from .predictor import Model

        self.simple = False
        if isinstance(schema, Model):
            student = schema if student is None else student
            self.simple = schema.simple
            schema = schema.schema.model
        self._model: Model | None = student if isinstance(student, Model) else None
        if isinstance(student, Model):
            student = student.engine
        self.schema: Schema = compile_schema(schema)
        self.teacher: Engine | None = from_string(teacher)
        self.student: Engine | None = from_string(student)
        if self.teacher is None and self.student is None:
            raise ValueError("give a teacher, a student, or both: ds.harness(Ticket, teacher='claude-sonnet-5')")
        self._explicit_threshold = threshold is not None
        threshold = DEFAULT_THRESHOLD if threshold is None else threshold
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1], got %r" % threshold)
        if not 0.0 <= audit <= 1.0:
            raise ValueError("audit must be in [0, 1], got %r" % audit)
        if not 0.0 <= collect <= 1.0:
            raise ValueError("collect must be in [0, 1] (the share of texts kept in the log), got %r" % collect)
        self.threshold, self.audit, self.collect = threshold, audit, collect
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
        saved = self._model.calibration if self._model is not None else {}
        if self._explicit_threshold:
            saved = {n: {k: v for k, v in c.items() if k != "threshold"} for n, c in saved.items()}
        self._calib: dict[str, dict[str, Any]] = (self.log.get(self._adapt_key(), {}) if self.log else {}) or saved

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
        flow = self._flow(text, pre if batched else None)
        jobs = next(flow)
        while True:
            pending = [self._executor().submit(self._ask, *job) for job in jobs[1:]] if not batched else []
            answers = [self._ask(*jobs[0]), *(p.result() for p in pending)]
            answers += [self._ask(*job) for job in jobs[1:]] if batched else []
            try:
                jobs = flow.send(answers)
            except StopIteration as done:
                return cast("Result[T]", done.value)

    async def adecide(self, text: str) -> Result[T]:
        """`decide` for async code: HTTP engines (LLMs, Jev, systemone) run natively async, others in a thread."""
        flow = self._flow(text, None)
        jobs = next(flow)
        while True:
            answers = await asyncio.gather(*(self._aask(*job) for job in jobs))
            try:
                jobs = flow.send(list(answers))
            except StopIteration as done:
                return cast("Result[T]", done.value)

    async def _aask(self, engine: Engine, text: str, names: list[str], raw: Any = None) -> _Answers:
        aask = getattr(engine, "aask", None)
        if raw is not None or not callable(aask):
            return await asyncio.to_thread(self._ask, engine, text, names, raw)
        try:
            reply = await aask(text, self.schema.questions(names))
        except Exception as e:
            reply = e
        if reply is None:
            reply = EngineError(engine.name, "aask returned None")
        return self._ask(engine, text, names, reply)

    async def acall(self, text: str) -> Any:
        """`h(text)` for async code."""
        return self._plain((await self.adecide(text)).value)

    def _flow(self, text: str, pre: Any) -> Generator[list[Job], list[_Answers], Result[T]]:
        """The routing: yields the engine calls to make (run in parallel), receives their answers."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        t0 = time.perf_counter()
        modes = self.modes
        student_fields = [n for n, m in modes.items() if m != "teacher"]
        first_teacher = [n for n, m in modes.items() if m in ("teacher", "shadow")]

        jobs: list[Job] = []
        if student_fields:
            assert self.student is not None
            jobs.append((self.student, text, student_fields, pre))
        if first_teacher:
            assert self.teacher is not None
            jobs.append((self.teacher, text, first_teacher, None))
        got = (yield jobs) if jobs else []
        s = got[0] if student_fields else _Answers()
        t = got[-1] if first_teacher else _Answers()

        adapted = {n: self._calibrated(n, d) for n, d in s.dists.items()}
        sure = {n: confidence(d) >= self._threshold(n) for n, d in adapted.items()}
        more = [n for n, m in modes.items() if m == "cascade" and not sure.get(n, False)]
        more += [n for n, m in modes.items() if m == "student" and n not in adapted]
        confident = [n for n, m in modes.items() if m == "cascade" and sure.get(n, False)]
        if confident and self.audit and self._random.random() < self.audit:
            more += confident
        more = [n for n in more if n not in t.dists]
        if more and self.teacher is not None:
            (extra,) = yield [(self.teacher, text, more, None)]
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
                    "text": text if self._keep_text(decision_id, adapted, sure, t.dists) else None,
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

    def _keep_text(
        self,
        decision_id: str | None,
        student: dict[str, Distribution],
        sure: dict[str, bool],
        teacher: dict[str, Distribution],
    ) -> bool:
        if self.collect >= 1.0:
            return True
        if self.collect <= 0.0:
            return False
        if not all(sure.values()):
            return True
        if any(n in teacher and top(d) != top(teacher[n]) for n, d in student.items()):
            return True
        return sampled(decision_id or "", self.collect)

    def forget(self, decision_id: str | None = None, *, older_than_days: float | None = None) -> int:
        """Delete a logged decision (text, answers, labels): `h.forget(r.id)`, or `h.forget(older_than_days=30)`."""
        if older_than_days is not None and older_than_days < 0:
            raise ValueError("older_than_days must be 0 or more, got %r" % older_than_days)
        if (decision_id is None) == (older_than_days is None):
            raise ValueError(
                "give a decision id or older_than_days, e.g. h.forget(r.id) or h.forget(older_than_days=30)"
            )
        log = self._require_log()
        if decision_id is not None:
            if not log.forget(decision_id):
                raise KeyError("no decision with id %r in %s" % (decision_id, log.path))
            return 1
        assert older_than_days is not None
        return log.forget(before=time.time() - older_than_days * 86400)

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
        """Write every decision with a label (human first, else teacher) as training JSONL. Returns rows written.

        Decisions logged without their text (see `collect=`) can't be exported.
        """
        log = self._require_log()
        n = export(self.schema, log, path, format)
        if n == 0 and any(r["text"] is None for r in log.rows(self.schema.name)):
            warnings.warn(
                "exported 0 rows: the logged decisions have no text (this harness or an earlier one used collect=0 "
                "or a low collect); raise collect= to keep texts for training",
                UserWarning,
                stacklevel=2,
            )
        return n

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
        if self._model is not None:
            self._model.calibration = {n: dict(c) for n, c in calib.items()}
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
            self._model = None
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
    threshold: float | None = None,
    log: str | os.PathLike[str] | None = "decisions.db",
    audit: float = 0.05,
    collect: float = 1.0,
) -> Harness[T]:
    """Connect a teacher (any LLM, `ds.LLM(...)`, or Jev) and a student (Laya, or Jev) behind one schema.

    h = ds.harness(Ticket, teacher="claude-sonnet-5", student="laya")
    h("You charged me twice, refund now!")   # -> Ticket(...)

    model = ds.model(["billing", "technical", "sales"])   # base or trained Laya
    h = ds.harness(model, teacher="claude-haiku-4-5")       # it becomes the student; h(text) -> "billing"

    `threshold` is how sure the student must be to answer in cascade. Per field, the first that exists wins: what
    `h.adapt()` fitted on this log, then a `threshold=` you pass, then the thresholds a loaded model was saved
    with (`model.evaluate`), then 0.8.

    `collect` is the share of texts kept in the log (1.0 keeps all). Below 1, a `collect` share (picked from the
    decision id) is kept plus every text the student was unsure of or answered differently from the teacher; the
    rest are logged without their text. With no student (teacher mode) only the share is kept. 0 keeps no text.
    """
    return Harness(
        schema,
        teacher=teacher,
        student=student,
        mode=mode,
        threshold=threshold,
        log=log,
        audit=audit,
        collect=collect,
    )
