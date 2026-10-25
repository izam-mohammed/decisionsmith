"""File helpers shared by export, generate, training logs and reports."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import Any


def parent(path: str) -> str:
    """Create the folder `path` goes in; returns `path`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def write_jsonl(path: str, records: Iterable[Any]) -> int:
    n = 0
    with open(parent(path), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def next_run(name: str, base: str = "") -> str:
    """The first free `runs/<name>-vN` folder under `base`."""
    stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
    n = 1
    while os.path.exists(os.path.join(base, "runs", "%s-v%d" % (stem, n))):
        n += 1
    return os.path.join(base, "runs", "%s-v%d" % (stem, n))


def folder_name(path: str) -> str:
    """The last part of a path, split on both `/` and `\\` so a Windows path reduces on any OS."""
    parts = [p for p in re.split(r"[\\/]+", path) if p and p != "."]
    return parts[-1] if parts else path


def path_like(value: str) -> bool:
    """A value that is a local path, whatever folder we run in: absolute, relative (`./`, `../`), `~`, or Windows."""
    return (
        os.path.isabs(value)
        or value.startswith((".", "~"))
        or "\\" in value
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
    )
