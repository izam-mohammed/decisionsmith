"""`data_check`: what to fix in a data file before training on it (balance, repeats, leaks, lengths)."""

from __future__ import annotations

import os
from collections import Counter
from statistics import median
from typing import Any

from .evaluation import MIN_PER_OPTION
from .schema import Schema
from .training.data import MIN_ROWS, _records, _split_of, same_text

NOT_FIELDS = {"id", "text", "split", "labelled_by", "checked", "group", "answers", "_csv"}
SHORT_WORDS = 3
EXAMPLES = 5


def _values(rec: dict[str, Any], schema: Schema | None) -> dict[str, Any]:
    raw = rec["answers"] if isinstance(rec.get("answers"), dict) else rec
    names = list(schema.fields) if schema else [k for k in raw if k not in NOT_FIELDS]
    return {k: raw[k] for k in names if raw.get(k) not in (None, "")}


def check(data: str | os.PathLike[str], schema: Schema | None = None) -> dict[str, Any]:
    """Counts per option, repeated texts (and ones labelled differently), texts in both train and test, lengths,
    who labelled the rows, and advice. Without a schema, every column other than id/text/split/labelled_by is a field.
    """
    records = [(where, rec) for where, rec in _records(data) if isinstance(rec, dict)]
    texts = [(where, rec, rec.get("text")) for where, rec in records]
    usable = [(w, r, t) for w, r, t in texts if isinstance(t, str) and t.strip()]
    balance: dict[str, Counter[str]] = {}
    invalid: Counter[str] = Counter()
    labelled = 0
    by_text: dict[str, list[tuple[str, dict[str, str], str]]] = {}
    for where, rec, text in usable:
        labels = {}
        for name, value in _values(rec, schema).items():
            try:
                label = schema.label_of(name, value) if schema else str(value)
            except ValueError:
                invalid[name] += 1
                continue
            labels[name] = label
            balance.setdefault(name, Counter())[label] += 1
        labelled += bool(labels)
        by_text.setdefault(same_text(text), []).append((where, labels, _split_of(rec, where)))
    repeats = {k: v for k, v in by_text.items() if len(v) > 1}
    conflicts = [k for k, v in repeats.items() if len({tuple(sorted(x[1].items())) for x in v if x[1]}) > 1]
    leaks = [k for k, v in by_text.items() if _mixed(v)]
    words = sorted(len(t.split()) for _, _, t in usable)
    splits = Counter(_split_of(r, w) or "none" for w, r, _ in usable)
    advice = _advice(balance, schema, invalid, labelled, repeats, conflicts, leaks, splits, len(texts) - len(usable))
    options = {n: dict(sorted(c.items())) for n, c in balance.items()}
    if schema:
        options = {n: {k: balance.get(n, Counter())[k] for k in f.labels} for n, f in schema.fields.items()}
    return {
        "path": os.path.basename(os.fspath(data)),
        "rows": len(records),
        "empty_texts": len(texts) - len(usable),
        "labelled": labelled,
        "balance": options,
        "invalid": dict(invalid),
        "split": dict(splits),
        "labelled_by": dict(Counter(str(r.get("labelled_by") or "none") for _, r, _ in usable)),
        "repeats": {
            "texts": len(repeats),
            "extra_rows": sum(len(v) - 1 for v in repeats.values()),
            "labelled_differently": len(conflicts),
            "examples": [[w for w, _, _ in repeats[k]] for k in (conflicts + list(repeats))[:EXAMPLES]],
        },
        "leaks": {"texts": len(leaks), "examples": [[w for w, _, _ in by_text[k]] for k in leaks[:EXAMPLES]]},
        "lengths": {
            "min_words": words[0] if words else 0,
            "median_words": median(words) if words else 0,
            "max_words": words[-1] if words else 0,
            "under_%d_words" % SHORT_WORDS: sum(w < SHORT_WORDS for w in words),
        },
        "advice": advice or ["looks fine: no repeats, no leaks, every option has rows"],
        "ok": not advice,
    }


def _mixed(rows: list[tuple[str, dict[str, str], str]]) -> bool:
    kinds = {x[2] or "train" for x in rows}
    return "test" in kinds and len(kinds) > 1


def _advice(
    balance: dict[str, Counter[str]],
    schema: Schema | None,
    invalid: Counter[str],
    labelled: int,
    repeats: dict[str, Any],
    conflicts: list[str],
    leaks: list[str],
    splits: Counter[str],
    empty: int,
) -> list[str]:
    out = []
    if empty:
        out.append("%d rows have no text; remove them" % empty)
    for name, n in sorted(invalid.items()):
        out.append("%s: %d values are not options of the field; fix or blank them" % (name, n))
    if labelled and labelled < MIN_ROWS:
        out.append("only %d labelled rows; fine-tuning needs at least %d" % (labelled, MIN_ROWS))
    names = list(schema.fields) if schema else list(balance)
    for name in names:
        counts = balance.get(name, Counter())
        options = list(schema.fields[name].labels) if schema else list(counts)
        low = [k for k in options if counts[k] < MIN_PER_OPTION]
        if labelled and low:
            few = ", ".join("%s (%d)" % (k, counts[k]) for k in low)
            out.append(
                "%s: %s have fewer than %d rows; label or write more of them (evaluate wants %d per option in the "
                "test split)" % (name, few, MIN_PER_OPTION, MIN_PER_OPTION)
            )
    if repeats:
        out.append("%d texts appear more than once; keep one copy of each" % len(repeats))
    if conflicts:
        out.append("%d repeated texts are labelled differently; decide which label is right" % len(conflicts))
    if leaks:
        out.append(
            "%d texts are in the test split and in training too, so evaluation would be optimistic; keep one copy"
            % len(leaks)
        )
    if labelled and "test" not in splits:
        out.append(
            "no split=test rows; hold some out (ds.golden marks a test split) or evaluate on a separate file, so "
            "evaluate never tests on rows the model trained on"
        )
    return out
