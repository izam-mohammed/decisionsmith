"""Labelling sessions: `ds.golden(..., teacher="agent")` picks the rows, a coding agent labels them over MCP,
`decisionsmith golden --finish` writes golden.csv.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from typing import Any

from .schema import Schema, compile_schema
from .training.data import near_copy, same_text, text_hash, words_of

FORMAT = "decisionsmith.golden-session/1"
RECHECK = 0.2
DEFAULT_AGENT = "coding-agent"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
_LOCK = threading.Lock()

RULES = (
    "Answer from the text alone, one text at a time. If no option fits or you are unsure, send "
    '{"id": ..., "skip": "why"} instead of guessing. Never change a text. '
)
INSTRUCTIONS = RULES + (
    "Each id takes one answer; send them with golden_submit. A share of the texts comes back later under a new id "
    "for an independent second answer: label every text as if you had never seen it."
)


def agent_of(teacher: Any) -> str | None:
    """`"agent"` or `"agent:<name>"` -> the agent's name ("" when unnamed); anything else -> None."""
    if not isinstance(teacher, str) or not (teacher == "agent" or teacher.startswith("agent:")):
        return None
    name = teacher.partition(":")[2]
    return _checked_name(name) if name else ""


def _checked_name(name: str) -> str:
    if not _NAME.match(name):
        raise ValueError("agent name must be letters, digits, '.', '_' or '-' (e.g. claude-code), got %r" % name)
    return name


def session_path(out: str) -> str:
    """`golden.csv` -> `golden.session.json`."""
    return os.path.splitext(out)[0] + ".session.json"


def _save(path: str, session: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no labelling session at %s; start one with `decisionsmith golden texts.csv --labels a,b --teacher agent`"
            % path
        )
    try:
        with open(path, encoding="utf-8") as f:
            session = json.load(f)
    except ValueError as e:
        raise ValueError("%s is not valid JSON (%s)" % (path, e)) from None
    if not isinstance(session, dict) or session.get("format") != FORMAT:
        raise ValueError("%s is not a decisionsmith labelling session (%s)" % (path, FORMAT))
    return session


def spec_of(session: dict[str, Any]) -> Any:
    """What `ds.model` takes for this session: its labels, or a class that asks the same questions."""
    from .artifact import rebuild

    return list(session["labels"]) if session["labels"] else rebuild(session["schema"])


def schema_of(session: dict[str, Any]) -> Schema:
    from .predictor import labels_model

    spec = spec_of(session)
    return compile_schema(labels_model(spec, session["question"]) if isinstance(spec, list) else spec)


def start(
    items: list[tuple[str, dict[str, str], str]],
    schema: Schema,
    out: str,
    *,
    agent: str | None,
    strategy: str,
    overwrite: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Write the session for `items` (text, human labels, split) and return the rows, unlabelled."""
    from .artifact import describe
    from .golden_set import _position

    path = session_path(out)
    simple = list(schema.fields) == ["label"] and schema.name == "Label"
    rows = []
    for text, human, split in items:
        human = {k: v for k, v in human.items() if k in schema.fields}
        full = bool(human) and set(human) == set(schema.fields)
        rows.append(
            {
                "id": "g" + text_hash(text)[:12],
                "text": text,
                "split": split,
                "human": human,
                "answers": dict(human) if full else None,
                "labelled_by": "human" if full else None,
                "skipped": None,
                "synthetic": False,
                "recheck": False,
                "check_id": None,
                "check": None,
            }
        )
    _pick_rechecks(rows, _position)
    session = {
        "format": FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": list(schema.fields["label"].labels) if simple else None,
        "question": schema.fields["label"].question["instructions"] if simple else None,
        "schema": describe(schema),
        "agent": agent or None,
        "strategy": strategy,
        "out": os.path.basename(out),
        "finished": None,
        "recheck": RECHECK,
        "rows": rows,
    }
    _save(path, session)
    todo = sum(r["answers"] is None for r in rows)
    if verbose:
        print(
            "golden: %d rows (%s) for a coding agent to label · %d marked split=test · wrote %s\n"
            "  label them with the MCP tools golden_batch / golden_submit, then: decisionsmith golden --finish %s"
            % (todo, strategy, sum(r["split"] == "test" for r in rows), path, path)
        )
    return [
        {"id": r["id"], "text": r["text"], "answers": {}, "split": r["split"], "labelled_by": r["labelled_by"]}
        for r in rows
    ]


def _pick_rechecks(rows: list[dict[str, Any]], position: Any) -> None:
    """A stable share of the rows the agent labels gets an independent second answer (at least one)."""
    open_rows = [r for r in rows if r["labelled_by"] != "human"]
    picked = [r for r in open_rows if position("recheck " + r["text"]) < RECHECK]
    if open_rows and not picked:
        picked = [min(open_rows, key=lambda r: position("recheck " + r["text"]))]
    for r in picked:
        r["recheck"] = True


def _fields(schema: Schema) -> dict[str, Any]:
    from .artifact import describe

    return {
        f["name"]: {"question": f["instructions"], "options": f["labels"], "descriptions": f["descriptions"]}
        for f in describe(schema)["fields"]
    }


def _stage(r: dict[str, Any]) -> int | None:
    if r["answers"] is None and r["skipped"] is None:
        return 1
    if r["recheck"] and r["check"] is None and r["answers"] is not None:
        return 2
    return None


def _open(schema: Schema, r: dict[str, Any]) -> list[str]:
    return [f for f in schema.fields if f not in r["human"]]


def _not_finished(path: str, session: dict[str, Any]) -> None:
    if session.get("finished"):
        raise ValueError(
            "%s is finished (%s was written from it); edit that file, or start a new session with overwrite=True "
            "(CLI: --overwrite) to label more" % (path, session["out"])
        )


def _check_id(taken: set[str]) -> str:
    """An id for a row's second answer: random, so it can't be worked out from the text or the row's id."""
    while True:
        cid = "g" + secrets.token_hex(6)
        if cid not in taken:
            taken.add(cid)
            return cid


def batch(path: str, size: int = 20) -> dict[str, Any]:
    """The next texts to label, with the options and instructions. Texts waiting for their second answer are mixed
    in under ids that only this function hands out."""
    from .golden_set import _position

    if size < 1:
        raise ValueError("size must be at least 1")
    with _LOCK:
        session = load(path)
        _not_finished(path, session)
        schema = schema_of(session)
        rows = session["rows"]
        taken = {r["id"] for r in rows} | {r["check_id"] for r in rows if r.get("check_id")}
        todo, new = [], False
        for r in rows:
            stage = _stage(r)
            if stage == 2 and not r.get("check_id"):
                r["check_id"], new = _check_id(taken), True
            if stage:
                todo.append((r["id"] if stage == 1 else r["check_id"], r))
        if new:
            _save(path, session)
    todo.sort(key=lambda x: _position("order " + x[0]))
    items = []
    for rid, r in todo[:size]:
        fields = _open(schema, r)
        only = {"fields": fields} if fields != list(schema.fields) else {}
        items.append({"id": rid, "text": r["text"], **only})
    return {
        "session": path,
        "items": items,
        "remaining": len(todo),
        "fields": _fields(schema),
        "instructions": INSTRUCTIONS if items else "",
        "answer_format": [{"id": "<id>", "answers": {n: "<option>" for n in schema.fields}}],
        "next": _next(path, session),
    }


def _labels(schema: Schema, fields: list[str], given: Any) -> dict[str, str]:
    if not isinstance(given, dict):
        raise ValueError("'answers' must be an object: {field: option}")
    unknown = sorted(set(given) - set(schema.fields))
    if unknown:
        raise ValueError("unknown fields %s; the fields are %s" % (unknown, list(schema.fields)))
    missing = [f for f in fields if given.get(f) is None]
    if missing:
        raise ValueError("missing %s; answer every field, or skip the text" % missing)
    return {f: schema.label_of(f, given[f]) for f in fields}


def _why(e: Exception) -> str:
    return str(e.args[0]) if e.args else type(e).__name__


def _name(session: dict[str, Any], agent: str | None) -> str:
    return "agent:" + _checked_name(agent or session["agent"] or DEFAULT_AGENT)


def _target(rows: dict[str, Any], checks: dict[str, Any], rid: Any) -> tuple[dict[str, Any] | None, int | None]:
    if not isinstance(rid, str):
        return None, None
    if rid in checks:
        r = checks[rid]
        return r, 2 if r["check"] is None else None
    r = rows.get(rid)
    return r, 1 if r is not None and r["answers"] is None and r["skipped"] is None else None


def _reason(given: Any) -> str:
    if not isinstance(given, str) or not given.strip():
        raise ValueError('a skip needs a reason: {"id": ..., "skip": "no option fits"}')
    return given.strip()


def submit(path: str, answers: list[dict[str, Any]], agent: str | None = None) -> dict[str, Any]:
    """Store the agent's answers after checking them against the schema; each bad item is rejected with why.
    Every id takes one answer: a second one is rejected, whichever row it is."""
    if not isinstance(answers, list):
        raise ValueError('answers must be a list: [{"id": "...", "answers": {"field": "option"}}]')
    with _LOCK:
        session = load(path)
        _not_finished(path, session)
        schema = schema_of(session)
        by = _name(session, agent)
        rows = {r["id"]: r for r in session["rows"]}
        checks = {r["check_id"]: r for r in session["rows"] if r.get("check_id")}
        accepted, rejected = 0, []
        for i, item in enumerate(answers):
            rid = item.get("id") if isinstance(item, dict) else None
            r, stage = _target(rows, checks, rid)
            try:
                if r is None:
                    raise ValueError("item %d: no row with id %r in this session" % (i, rid))
                if stage is None:
                    raise ValueError("already answered; to change a label, edit golden.csv after --finish")
                if isinstance(item, dict) and "skip" in item:
                    reason = _reason(item["skip"])
                    if stage == 1:
                        r["skipped"], r["labelled_by"] = reason, by
                    else:
                        r["check"] = {"skipped": reason, "labelled_by": by}
                else:
                    labels = _labels(schema, _open(schema, r), item.get("answers"))
                    if stage == 1:
                        r["answers"] = {**labels, **r["human"]}
                        r["labelled_by"] = by + ("+human" if r["human"] else "")
                    else:
                        r["check"] = {"answers": labels, "labelled_by": by}
                accepted += 1
            except (ValueError, KeyError) as e:
                rejected.append({"id": rid, "error": _why(e)})
        _save(path, session)
    return {"accepted": accepted, "rejected": rejected, **_counts(session), "next": _next(path, session)}


def add(path: str, examples: list[dict[str, Any]], agent: str | None = None) -> dict[str, Any]:
    """Add examples the agent wrote (marked synthetic, train split only, each needs an agreeing second answer
    before it counts). Texts that repeat a session text, or share most words with a test text, are rejected."""
    if not isinstance(examples, list):
        raise ValueError('examples must be a list: [{"text": "...", "answers": {"field": "option"}}]')
    with _LOCK:
        session = load(path)
        _not_finished(path, session)
        schema = schema_of(session)
        by = _name(session, agent) + ":synthetic"
        seen = {same_text(r["text"]) for r in session["rows"]}
        tests = [words_of(r["text"]) for r in session["rows"] if r["split"] == "test"]
        accepted, rejected = 0, []
        for i, ex in enumerate(examples):
            text = ex.get("text") if isinstance(ex, dict) else None
            try:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("item %d: needs a non-empty 'text'" % i)
                if same_text(text) in seen:
                    raise ValueError("item %d: this text is already in the session" % i)
                if any(near_copy(words_of(text), t) for t in tests):
                    raise ValueError("item %d: shares most of its words with a held-out test text; write a new one" % i)
                labels = _labels(schema, list(schema.fields), ex.get("answers"))
            except (ValueError, KeyError) as e:
                rejected.append({"item": i, "error": _why(e)})
                continue
            seen.add(same_text(text))
            session["rows"].append(
                {
                    "id": "g" + text_hash(text)[:12],
                    "text": text,
                    "split": "train",
                    "human": {},
                    "answers": labels,
                    "labelled_by": by,
                    "skipped": None,
                    "synthetic": True,
                    "recheck": True,
                    "check_id": None,
                    "check": None,
                }
            )
            accepted += 1
        _save(path, session)
    return {"accepted": accepted, "rejected": rejected, **_counts(session), "next": _next(path, session)}


def _disputed(r: dict[str, Any]) -> list[str]:
    check = r["check"]
    if not check or r["answers"] is None:
        return []
    if "skipped" in check:
        return [f for f in r["answers"] if f not in r["human"]]
    return [f for f, v in check["answers"].items() if r["answers"].get(f) != v]


def _counts(session: dict[str, Any]) -> dict[str, Any]:
    rows = session["rows"]
    real = [r for r in rows if not r["synthetic"]]
    checked = [r for r in rows if r["check"] is not None]
    disagreed = [r for r in checked if _disputed(r)]
    return {
        "rows": len(real),
        "labelled": sum(r["answers"] is not None for r in real),
        "skipped": sum(r["skipped"] is not None for r in real),
        "to_label": sum(_stage(r) == 1 for r in rows),
        "to_recheck": sum(_stage(r) == 2 for r in rows),
        "rechecked": len(checked),
        "agreed": len(checked) - len(disagreed),
        "disagreed": len(disagreed),
        "synthetic": len(rows) - len(real),
    }


def _next(path: str, session: dict[str, Any]) -> str:
    if session.get("finished"):
        return "finished: %s is written; edit it there" % session["out"]
    c = _counts(session)
    if c["to_label"] or c["to_recheck"]:
        return "golden_batch(%r) for the next texts" % path
    return "done: golden_finish(%r) or `decisionsmith golden --finish %s` writes the CSV" % (path, path)


def status(path: str) -> dict[str, Any]:
    """Progress, label balance, agreement between the two passes, and every row to show the user."""
    session = load(path)
    c = _counts(session)
    rows = session["rows"]
    balance: dict[str, dict[str, int]] = {}
    for r in rows:
        for f, v in (r["answers"] or {}).items():
            balance.setdefault(f, {}).setdefault(v, 0)
            balance[f][v] += 1
    disagreements = [
        {
            "id": r["id"],
            "text": r["text"][:200],
            "field": f,
            "first": r["answers"][f],
            "second": r["check"]["answers"][f] if "answers" in r["check"] else "skipped: " + r["check"]["skipped"],
            "synthetic": r["synthetic"],
        }
        for r in rows
        for f in _disputed(r)
    ]
    return {
        "session": path,
        **c,
        "agreement": c["agreed"] / c["rechecked"] if c["rechecked"] else None,
        "finished": bool(session.get("finished")),
        "split": {s: sum(r["split"] == s for r in rows) for s in ("train", "test")},
        "balance": balance,
        "labelled_by": _tally(r["labelled_by"] for r in rows if r["labelled_by"]),
        "disagreements": disagreements,
        "skipped_rows": [{"id": r["id"], "text": r["text"][:200], "why": r["skipped"]} for r in rows if r["skipped"]],
        "next": _next(path, session),
    }


def _tally(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _bare(path: str, name: Any) -> str:
    if not isinstance(name, str) or name in ("", ".", "..") or os.path.basename(name) != name or "\\" in name:
        raise ValueError("%s: 'out' must be a file name next to the session file, got %r" % (path, name))
    return name


def finish(path: str, out: str | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Write golden.csv from the session: the same columns as an LLM-labelled one, plus `checked`.

    Fields the two passes disagree on are left blank for the user; synthetic rows count only once re-checked and
    agreed; skipped and unlabelled rows are left out.
    """
    from .golden_set import _check_out, _write

    session = load(path)
    schema = schema_of(session)
    out = out or os.path.join(os.path.dirname(path), _bare(path, session["out"]))
    _check_out(out, overwrite)
    rows, dropped = [], 0
    for r in session["rows"]:
        if r["answers"] is None or r["skipped"] is not None:
            continue
        disputed = _disputed(r)
        if r["synthetic"] and (r["check"] is None or disputed):
            dropped += 1
            continue
        answers = {
            f: {k: float(k == v) for k in schema.fields[f].labels} for f, v in r["answers"].items() if f not in disputed
        }
        checked = "" if r["check"] is None else "disagreed: " + ", ".join(disputed) if disputed else "agreed"
        rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "answers": answers,
                "split": r["split"],
                "labelled_by": r["labelled_by"],
                "checked": checked,
            }
        )
    if not rows:
        raise ValueError("nothing labelled yet in %s; label with golden_batch / golden_submit first" % path)
    _write(rows, out, schema)
    with _LOCK:
        latest = load(path)
        latest["finished"] = latest.get("finished") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(path, latest)
    c = _counts(session)
    test = sum(r["split"] == "test" for r in rows)
    message = "golden: wrote %s: %d rows (%d split=test)%s%s%s%s" % (
        out,
        len(rows),
        test,
        " · %d re-checked, %d agreed, %d disagreed%s"
        % (c["rechecked"], c["agreed"], c["disagreed"], " (left blank for you to fill)" if c["disagreed"] else "")
        if c["rechecked"]
        else "",
        " · %d skipped" % c["skipped"] if c["skipped"] else "",
        " · %d not labelled yet" % c["to_label"] if c["to_label"] else "",
        " · %d synthetic rows left out (not re-checked or disagreed)" % dropped if dropped else "",
    )
    return {
        "path": os.path.abspath(out),
        "rows": len(rows),
        "test": test,
        "dropped_synthetic": dropped,
        **{k: c[k] for k in ("skipped", "to_label", "rechecked", "agreed", "disagreed")},
        "message": message,
    }
