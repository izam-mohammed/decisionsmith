"""`ds.model`: pick labels (or a Pydantic class), `.train()`, `.predict()`, plug it into a harness."""

from __future__ import annotations

import asyncio
import csv
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from .engines import Engine, EngineError, LayaEngine, ask_many, from_string, response, write
from .files import next_run, parent, write_jsonl
from .report import Report
from .schema import Schema, compile_schema, top

DEFAULT_QUESTION = "Which label fits this text?"
STYLES = (
    "short and casual",
    "long and detailed",
    "formal and polite",
    "frustrated or angry",
    "very brief, a few words",
    "with small typos and no punctuation",
    "rambling, with unrelated details mixed in",
    "matter-of-fact, like a notification",
)


def labels_model(labels: list[str], question: str | None = None) -> type[BaseModel]:
    labels = [str(x) for x in labels]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("give at least two different labels, e.g. ds.model(['spam', 'not spam'])")
    options: Any = Literal[tuple(labels)]
    return create_model("Label", label=(options, Field(description=question or DEFAULT_QUESTION)))


def _engine(spec: Any, device: str | None) -> Engine:
    if isinstance(spec, str) and os.path.isdir(spec):
        return LayaEngine(spec, device=device)
    if isinstance(spec, str) and (spec == "laya" or spec.startswith("laya:")):
        return LayaEngine(spec.partition(":")[2] or "laya", device=device)
    return from_string(spec)


class Model:
    """A decision model you can train and use, and hand to `ds.harness` as the student. Build it with `ds.model`."""

    def __init__(
        self, spec: Any, engine: Any = "laya", *, question: str | None = None, device: str | None = None
    ) -> None:
        self.simple = isinstance(spec, (list, tuple))
        self.schema: Schema = compile_schema(labels_model(list(spec), question) if self.simple else spec)
        self.engine: Engine = _engine(engine, device)
        self.device = device
        self.trained: str | None = None
        self.path: str | None = None
        self.info: dict[str, Any] = {}
        self.calibration: dict[str, dict[str, Any]] = {}
        self.report: Report | None = None

    @property
    def name(self) -> str:
        return self.engine.name

    def __repr__(self) -> str:
        what = list(self.schema.fields["label"].labels) if self.simple else self.schema.name
        return "ds.model(%r, engine=%r)" % (what, self.engine.name)

    def ask(self, text: str, questions: dict[str, Any]) -> Any:
        return self.engine.ask(text, questions)

    def ask_many(self, texts: list[str], questions: dict[str, Any]) -> list[Any]:
        return ask_many(self.engine, texts, questions)

    def value(self, obj: BaseModel) -> Any:
        return obj.label if self.simple else obj  # type: ignore[attr-defined]

    def predict(self, text: str | list[str]) -> Any:
        """One text -> the label (or your class). A list of texts -> a list, batched."""
        return self._values(text, self.ask_many(self._texts(text), self.schema.questions()))

    async def apredict(self, text: str | list[str]) -> Any:
        """`predict` for async code: HTTP engines run natively async, Laya runs batched in a thread."""
        texts, questions = self._texts(text), self.schema.questions()
        aask = getattr(self.engine, "aask", None)
        if callable(aask):
            raws = list(await asyncio.gather(*(aask(t, questions) for t in texts)))
        else:
            raws = await asyncio.to_thread(self.ask_many, texts, questions)
        return self._values(text, raws)

    def _texts(self, text: str | list[str]) -> list[str]:
        texts = [text] if isinstance(text, str) else list(text)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("predict() takes a non-empty string or a list of them")
        return texts

    def _values(self, text: str | list[str], raws: list[Any]) -> Any:
        questions = self.schema.questions()
        out = []
        for raw in raws:
            answers, _ = response(self, raw, questions)
            out.append(self.value(self.schema.build(self.schema.distributions(answers))))
        return out[0] if isinstance(text, str) else out

    def label(self, texts: list[str], teacher: Any, *, verbose: bool = True) -> list[dict[str, Any]]:
        """Have a teacher (an LLM name, `ds.LLM(...)`, or "jev") label your texts. Returns rows for `.train()`."""
        engine = from_string(teacher)
        questions = self.schema.questions()

        def one(text: str) -> dict[str, Any] | None:
            try:
                answers, _ = response(engine, engine.ask(text, questions), questions)
                return {"text": text, "answers": self.schema.distributions(answers)}
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = [r for r in pool.map(one, texts) if r is not None]
        if verbose and len(rows) < len(texts):
            print(
                "%s could not label %d of %d texts; using the rest" % (engine.name, len(texts) - len(rows), len(texts))
            )
        return rows

    def _describe(self, target: dict[str, str]) -> list[str]:
        lines = []
        for name, label in target.items():
            q = self.schema.fields[name].question
            crit = q.get("criteria")
            desc = crit.get(label) if isinstance(crit, dict) else None
            answer = {"true": "yes", "false": "no"}.get(label, label) if q["type"] == "noul" else label
            lines.append("- %s -> %s%s" % (q["instructions"], answer, (" (%s)" % desc) if desc else ""))
        return lines

    def generate(
        self,
        n: int,
        teacher: Any,
        *,
        about: str | None = None,
        per_call: int = 10,
        seed: int = 0,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Have an LLM write `n` labelled examples (a golden dataset to review and train on).

        Strategy: every label (every option of every field) gets an equal share; each batch asks for a different
        style; duplicates are dropped; every text is then labelled again by the teacher, blind, and kept only if
        that answer matches the intended label.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        engine = from_string(teacher)
        rng = random.Random(seed)
        plans = {}
        for name, f in self.schema.fields.items():
            cycle: list[str] = []
            while len(cycle) < n:
                block = list(f.labels)
                rng.shuffle(block)
                cycle += block
            plans[name] = cycle[:n]
        counts = Counter(tuple((name, plans[name][i]) for name in self.schema.fields) for i in range(n))
        jobs: list[tuple[dict[str, str], int, str]] = []
        for key, count in counts.items():
            for start in range(0, count, per_call):
                jobs.append((dict(key), min(per_call, count - start), STYLES[len(jobs) % len(STYLES)]))

        def one(job: tuple[dict[str, str], int, str]) -> list[tuple[str, dict[str, str]]]:
            target, k, style = job
            prompt = "\n".join(
                [
                    ("Write %d different realistic texts%s." if k > 1 else "Write %d realistic text%s.")
                    % (k, (" (%s)" % about) if about else ""),
                    "Every text must have these answers:",
                    *self._describe(target),
                    "Style: %s. Vary the wording, length and details. No numbering, no quotes." % style,
                ]
            )
            try:
                return [(t, target) for t in write(engine, prompt, k)]
            except EngineError as e:
                if verbose:
                    print("could not generate a batch: %s" % str(e).splitlines()[0])
                return []

        with ThreadPoolExecutor(max_workers=8) as pool:
            written = [x for batch in pool.map(one, jobs) for x in batch]
        firsts: dict[str, tuple[str, dict[str, str]]] = {}
        for text, target in written:
            firsts.setdefault(" ".join(text.lower().split()), (text, target))
        unique = list(firsts.values())
        checked = {r["text"]: r["answers"] for r in self.label([t for t, _ in unique], engine, verbose=False)}
        rows = [
            {"text": text, "answers": dict(target)}
            for text, target in unique
            if (got := checked.get(text)) and all(top(got[f]) == target[f] for f in target)
        ]
        if verbose:
            print(
                "generated %d texts with %s · %d duplicates dropped · %d kept after re-checking"
                % (len(written), engine.name, len(written) - len(unique), len(rows))
            )
        return rows

    def _rows(self, data: Any) -> Any:
        if isinstance(data, (str, os.PathLike)):
            return data
        rows = []
        for i, x in enumerate(data):
            if isinstance(x, str):
                raise ValueError("row %d is only text; add labels, or pass teacher=... to label it" % i)
            if isinstance(x, (tuple, list)):
                if not self.simple or len(x) != 2:
                    raise ValueError("row %d: (text, label) pairs work with ds.model([labels]); use dicts here" % i)
                x = {"text": x[0], "label": x[1]}
            if not isinstance(x, dict):
                raise ValueError("row %d: expected {'text': ..., 'label': ...}" % i)
            answers = x.get("answers")
            if not isinstance(answers, dict):
                answers = {k: v for k, v in x.items() if k in self.schema.fields}
            rows.append({"id": x.get("id"), "text": x.get("text"), "answers": answers})
        return rows

    def train(
        self,
        data: Any = None,
        *,
        teacher: Any = None,
        generate: int | None = None,
        about: str | None = None,
        out: str | None = None,
        verbose: bool = True,
        **options: Any,
    ) -> Report:
        """Fine-tune Laya.

            model.train("tickets.csv")                                  # your labels (text + label columns)
            model.train(texts, teacher="claude-haiku-4-5")               # only texts: the teacher labels them
            model.train(generate=300, teacher=ds.LLM(...), about="...")  # no data: the teacher writes it

        The model switches to the trained weights unless they came out worse than before.
        """
        from .training.finetuning import finetune

        if not isinstance(self.engine, LayaEngine):
            raise EngineError(
                self.engine.name, "this engine can't be trained", "use the default: ds.model(..., 'laya')"
            )
        synthetic = data is None
        if data is None:
            if not generate or teacher is None:
                raise ValueError(
                    "give training data, or let an LLM write it: "
                    "model.train(generate=300, teacher='claude-haiku-4-5', about='support emails')"
                )
            rows: Any = self.generate(generate, teacher, about=about, verbose=verbose)
        elif teacher is not None:
            if isinstance(data, (str, os.PathLike)):
                raise ValueError("with teacher=..., pass the texts as a list: model.train(texts, teacher=...)")
            rows = self.label([x if isinstance(x, str) else x["text"] for x in data], teacher, verbose=verbose)
        else:
            rows = self._rows(data)
        out = out or next_run(self.schema.name)
        report = finetune(rows, self.schema.model, base=self.engine.spec, out=out, verbose=False, **options)
        base = report.details["base"]["all"].get("accuracy") or 0.0
        tuned = report.details["finetuned"]["all"].get("accuracy") or 0.0
        n = report.details["finetuned"]["all"].get("n", 0)
        better = tuned >= base
        if better:
            self.engine = LayaEngine(out, device=self.device)
            self.trained = out
        if verbose:
            rows_n, using = report.details["provenance"]["rows"]["train"], "now using %s" % out
            print(
                "trained on %d rows · accuracy %.2f -> %.2f on %d held-out decisions · %s"
                % (rows_n, base, tuned, n, using if better else "worse than before, kept the old model")
            )
            if synthetic:
                print(
                    "  note: the held-out rows are LLM-written too, so real accuracy will be lower; test on real texts"
                )
            elif n < 100:
                print("  note: only %d held-out decisions, so these numbers are rough; more data helps" % n)
        return report

    def evaluate(self, data: Any, *, target: float = 0.97) -> Report:
        """Test on labelled data the model never trained on: accuracy, macro-F1, calibration, coverage, worst cases.

            report = model.evaluate("test.csv")   # same formats as .train()
            report.go                              # ready to answer behind the harness?

        Also picks each field's confidence threshold (the lowest at which it reached `target` accuracy here);
        `model.save()` stores it and the report, and the harness uses it for cascade.
        """
        from .evaluation import evaluate

        return evaluate(self, data, target)

    def save(self, path: str | None = None) -> str:
        """Write a versioned model folder and return its path; versions are never overwritten.

            model.save()                  # models/<name>-v1, then -v2, ...
            model.save("models/ticket")   # models/ticket-v1, then models/ticket-v2, ...

        The folder holds the Laya checkpoint (it loads in `laya.load`), `decisionsmith.json` (labels or schema,
        calibration, thresholds, provenance), `report.json` (the latest `evaluate`) and `MODEL_CARD.md`.
        Load it with `ds.load(path)`.
        """
        from .artifact import save

        return save(self, path)


def write_rows(rows: list[dict[str, Any]], path: str, schema: Schema) -> str:
    """Rows -> CSV (`text` + one column per field, easy to review) or JSONL (`{"text", "answers"}`)."""
    if not path.endswith(".csv"):
        write_jsonl(path, ({"text": r["text"], "answers": r["answers"]} for r in rows))
        return path
    with open(parent(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", *schema.fields])
        w.writeheader()
        for r in rows:
            w.writerow(
                {"text": r["text"], **{k: top(v) if isinstance(v, dict) else str(v) for k, v in r["answers"].items()}}
            )
    return path


def load(path: str | os.PathLike[str], schema: Any = None, *, device: str | None = None) -> Model:
    """Load a folder written by `model.save()`, ready to predict or to use in `ds.harness`.

    m = ds.load("models/ticket-v2")           # labels, schema, calibration and thresholds come with it
    m = ds.load("models/ticket-v2", Ticket)   # use your own class; it must match the saved schema
    """
    from .artifact import load as _load

    return _load(path, schema, device=device)


def model(spec: Any, engine: Any = "laya", *, question: str | None = None, device: str | None = None) -> Model:
    """A decision model in one line.

        model = ds.model(["billing", "technical", "sales"])   # or a Pydantic class for several answers
        model.train("tickets.csv")                             # columns: text, label
        model.predict("You charged me twice")                  # -> "billing"

    `engine` is `"laya"` (default, trainable), a saved model folder, `"jev"`, or any LLM.
    """
    return Model(spec, engine, question=question, device=device)
