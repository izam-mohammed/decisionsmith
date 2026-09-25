"""`DS_OFFLINE=1`: every LLM and hosted engine answers locally with a keyword stand-in, so examples, notebooks and
demos run without keys or network. `DS_LAYA=<dir>` swaps the named Laya checkpoints for a local one.
"""

from __future__ import annotations

import hashlib
import os
import re
import typing
from typing import Any

from pydantic import BaseModel

_TEXT = re.compile(r"<text>\n(.*)\n</text>", re.S)
_TARGET = re.compile(r"-> (.+?)(?: \(|$)", re.M)


def enabled() -> bool:
    return os.environ.get("DS_OFFLINE", "").strip().lower() not in ("", "0", "false", "no")


def laya(spec: str) -> str | None:
    """The `DS_LAYA` checkpoint that stands in for a named Laya model when offline."""
    path = os.environ.get("DS_LAYA")
    return path if enabled() and path and os.path.isdir(path) else None


def _hash(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _word(text: str, word: str) -> bool:
    return re.search(r"(?<!\w)%s(?!\w)" % re.escape(word.lower()), text.lower()) is not None


def _pick(text: str, name: str, annotation: Any) -> Any:
    if annotation is bool:
        if _word(text, "yes") or _word(text, "true"):
            return True
        if _word(text, "no") or _word(text, "false"):
            return False
        return _hash(text, name) % 2 == 0
    opts = [str(o) for o in typing.get_args(annotation)]
    return next((o for o in opts if _word(text, o)), opts[_hash(text, name) % len(opts)])


def fill(user: str, model: type[BaseModel]) -> dict[str, Any]:
    """A valid reply for `model`: labels named in the text win, otherwise a stable hash picks one."""
    if "texts" in model.model_fields:
        n = int((re.search(r"exactly (\d+) texts", user) or re.search(r"(\d+)", user) or ["", "3"])[1])
        targets = " ".join(t.strip() for t in _TARGET.findall(user))
        return {"texts": ["offline sample %d %d: %s" % (_hash(user) % 1000, i, targets) for i in range(n)]}
    m = _TEXT.search(user)
    text = m.group(1) if m else user
    return {name: _pick(text, name, f.annotation) for name, f in model.model_fields.items()}


def answer(text: str, questions: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    """A Jev-format response for `questions`, as an LLM teacher would give it offline."""
    from .engines.structured import answers_model, one_hot

    model = answers_model(questions)
    return one_hot(model(**fill("<text>\n%s\n</text>" % text, model)), questions, {}, name)
