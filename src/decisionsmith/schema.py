"""Pydantic model <-> Jev `/v1/systemone` questions and answers."""

from __future__ import annotations

import enum
import hashlib
import json
import math
import types
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

MAX_OPTIONS = 32
WARN_OPTIONS = 20

Distribution = dict[str, float]


class Options:
    """Option descriptions for a field: `Annotated[Literal["a", "b"], ds.Options(a="...", b="...")]`."""

    def __init__(self, **descriptions: str) -> None:
        self.descriptions = {k: str(v) for k, v in descriptions.items()}

    def __repr__(self) -> str:
        return "Options(%s)" % ", ".join("%s=%r" % kv for kv in self.descriptions.items())


class Scale:
    """Marks an ordered rating, lowest first: `urgency: Annotated[Literal["low", "medium", "high"], ds.Scale]`.

    The value is the most likely level.
    """


@dataclass(frozen=True)
class Field:
    name: str
    kind: Literal["choice", "bool", "scale"]
    labels: tuple[str, ...]
    values: tuple[Any, ...]
    question: dict[str, Any]


def _human(name: str) -> str:
    return name.replace("_", " ").strip()


def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType) and type(None) in get_args(annotation)


def _field(model: type[BaseModel], name: str, info: Any, context: str) -> Field | None:
    annotation, meta = info.annotation, info.metadata
    options = next((m for m in meta if isinstance(m, Options)), None)
    scale = any(m is Scale or isinstance(m, Scale) for m in meta)
    descriptions = options.descriptions if options else {}
    where = "%s.%s" % (model.__name__, name)
    has_default = not info.is_required()

    if _is_optional(annotation):
        if has_default:
            return None
        raise TypeError("%s: Optional fields can't be decided; remove None or give the field a default" % where)

    if annotation is bool:
        kind: Literal["choice", "bool", "scale"] = "bool"
        labels: tuple[str, ...] = ("false", "true")
        values: tuple[Any, ...] = (False, True)
    elif get_origin(annotation) is Literal:
        values = get_args(annotation)
        if all(isinstance(v, bool) for v in values) and set(values) == {True, False}:
            kind, labels, values = "bool", ("false", "true"), (False, True)
        else:
            kind = "scale" if scale else "choice"
            labels = tuple(str(v) for v in values)
    elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        kind, values = ("scale" if scale else "choice"), tuple(annotation)
        labels = tuple(str(m.value) for m in values)
    else:
        if has_default:
            return None
        raise TypeError(
            "%s: %r can't be decided; use Literal[...], an Enum, bool, or Annotated[Literal[...], ds.Scale], "
            "or give the field a default so it is skipped" % (where, annotation)
        )

    if scale and kind == "bool":
        raise TypeError("%s: ds.Scale marks ordered levels; use it on Literal[...] or an Enum, not bool" % where)
    if len(labels) < 2:
        raise TypeError("%s: a decision needs at least two options" % where)
    if len(set(labels)) != len(labels):
        raise TypeError("%s: option labels must be unique, got %r" % (where, labels))
    if len(labels) > MAX_OPTIONS:
        raise TypeError("%s: %d options is more than %d; split the field" % (where, len(labels), MAX_OPTIONS))
    if len(labels) > WARN_OPTIONS:
        warnings.warn(
            "%s has %d options; decision models lose accuracy above %d" % (where, len(labels), WARN_OPTIONS),
            UserWarning,
            stacklevel=4,
        )
    unknown = set(descriptions) - set(labels)
    if unknown:
        raise TypeError("%s: ds.Options names unknown options %s" % (where, sorted(unknown)))

    human = _human(name)
    default_ins = {
        "choice": "What is the %s?" % human,
        "bool": "Is this true: %s?" % human,
        "scale": "Rate the %s." % human,
    }[kind]
    instructions = (info.description or default_ins).strip()
    if context:
        instructions = "%s %s" % (context, instructions)

    question: dict[str, Any]
    if kind == "choice":
        criteria: Any = {label: descriptions.get(label, "") for label in labels} if descriptions else list(labels)
        question = {"type": "choice", "instructions": instructions, "criteria": criteria}
    elif kind == "bool":
        question = {"type": "noul", "instructions": instructions}
        if descriptions:
            question["criteria"] = dict(descriptions)
    else:
        question = {
            "type": "score",
            "instructions": instructions,
            "criteria": [
                "%s: %s" % (label, descriptions[label]) if label in descriptions else label for label in labels
            ],
        }
    return Field(name=name, kind=kind, labels=labels, values=values, question=question)


class Schema:
    """A compiled decision schema. Build with `compile(Model)`."""

    def __init__(self, model: type[BaseModel]) -> None:
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError("expected a Pydantic model class, got %r" % (model,))
        doc = (model.__doc__ or "").strip()
        context = " ".join(doc.split()) if doc and model.__doc__ is not BaseModel.__doc__ else ""
        fields = [f for name, info in model.model_fields.items() if (f := _field(model, name, info, context))]
        if not fields:
            raise TypeError("%s has no decision fields (Literal, Enum, bool or ds.Scale)" % model.__name__)
        self.model = model
        self.name = model.__name__
        self.fields: dict[str, Field] = {f.name: f for f in fields}

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({n: [f.kind, f.labels] for n, f in self.fields.items()}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def questions(self, names: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        return {n: self.fields[n].question for n in (self.fields if names is None else names)}

    def distributions(self, answers: dict[str, Any], names: Iterable[str] | None = None) -> dict[str, Distribution]:
        return {n: self.distribution(n, answers[n]) for n in (self.fields if names is None else names)}

    def distribution(self, name: str, answer: Any) -> Distribution:
        """One engine answer (Jev wire format) -> probabilities over the field's labels."""
        f = self.fields[name]
        if not isinstance(answer, dict):
            raise ValueError("answer for %r is not an object: %r" % (name, answer))
        probs: list[float]
        if f.kind == "bool":
            p = answer.get("noul")
            if p is None and isinstance(answer.get("probabilities"), dict):
                p = answer["probabilities"].get("true")
            p = _prob(p, name)
            probs = [1.0 - p, p]
        else:
            raw = answer.get("probabilities")
            keys = f.labels if f.kind == "choice" else [str(i) for i in range(len(f.labels))]
            probs = []
            if isinstance(raw, dict) and raw:
                probs = [_prob(raw.get(k, 0.0), name) for k in keys]
            if not probs or sum(probs) <= 0:
                probs = self._one_hot(f, answer)
        total = sum(probs)
        return {label: p / total for label, p in zip(f.labels, probs)}

    def _one_hot(self, f: Field, answer: dict[str, Any]) -> list[float]:
        if f.kind == "choice":
            pick = answer.get("choice")
            if pick not in f.labels:
                raise ValueError("answer for %r picks %r, not one of %s" % (f.name, pick, list(f.labels)))
            idx = f.labels.index(pick)
        else:
            score = answer.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                raise ValueError("answer for %r has no probabilities or score" % f.name)
            idx = min(len(f.labels) - 1, max(0, round(score)))
        return [1.0 if i == idx else 0.0 for i in range(len(f.labels))]

    def label_of(self, name: str, value: Any) -> str:
        """A human or teacher value (label, Python value, enum member, bool-ish) -> the field's label."""
        if name not in self.fields:
            raise KeyError("%s has no decision field %r; fields: %s" % (self.name, name, list(self.fields)))
        f = self.fields[name]
        if f.kind == "bool":
            text = str(value).strip().lower()
            if isinstance(value, bool) or text in ("true", "false"):
                return "true" if text == "true" else "false"
            if text in ("yes", "y", "1"):
                return "true"
            if text in ("no", "n", "0"):
                return "false"
        else:
            if isinstance(value, enum.Enum):
                value = value.value
            text = str(value)
            if text in f.labels:
                return text
            for member in f.values:
                if isinstance(member, enum.Enum) and member.name == text:
                    return str(member.value)
        raise ValueError("%r is not an option of %s.%s: %s" % (value, self.name, name, list(f.labels)))

    def value(self, name: str, dist: Distribution) -> Any:
        f = self.fields[name]
        return f.values[argmax([dist.get(k, 0.0) for k in f.labels])]

    def build(self, dists: dict[str, Distribution]) -> BaseModel:
        return self.model(**{n: self.value(n, d) for n, d in dists.items()})


def _prob(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("answer for %r has a non-numeric probability %r" % (name, value))
    return min(1.0, max(0.0, float(value)))


def confidence(dist: Distribution) -> float:
    return max(dist.values()) if dist else 0.0


def top(dist: Distribution) -> str:
    return max(dist, key=dist.__getitem__)


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def option_keys(q: dict[str, Any]) -> list[str]:
    """The answer keys of a Jev-format question: labels for choice, level indices for score, false/true for noul."""
    if q["type"] == "noul":
        return ["false", "true"]
    crit = q.get("criteria") or []
    return [str(k) for k in crit] if q["type"] == "choice" else [str(i) for i in range(len(crit))]


@lru_cache(maxsize=256)
def compile_schema(model: type[BaseModel]) -> Schema:
    return Schema(model)
