"""The decision log: one SQLite file with every decision, human label and adaptation."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from typing import Any

_TABLES = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY, ts REAL, schema TEXT, text TEXT, value TEXT, source TEXT,
    teacher TEXT, student TEXT, teacher_dists TEXT, student_dists TEXT, latency_ms REAL
);
CREATE INDEX IF NOT EXISTS decisions_schema ON decisions (schema, ts);
CREATE TABLE IF NOT EXISTS labels (id TEXT, field TEXT, label TEXT, ts REAL, PRIMARY KEY (id, field));
CREATE TABLE IF NOT EXISTS trained (id TEXT, run TEXT, PRIMARY KEY (id, run));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
_COLUMNS = ("id", "ts", "text", "value", "source", "teacher", "student", "teacher_dists", "student_dists", "latency_ms")
_JSON = ("value", "source", "teacher_dists", "student_dists")
_INSERT = "INSERT INTO decisions (schema, %s) VALUES (?%s)" % (", ".join(_COLUMNS), ",?" * len(_COLUMNS))


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class Log:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA secure_delete=ON")
            self._db.executescript(_TABLES)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> Log:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, row: dict[str, Any]) -> None:
        row = {"ts": time.time(), **row}
        values = [row["schema"], *(_dump(row.get(k)) if k in _JSON else row.get(k) for k in _COLUMNS)]
        with self._lock:
            self._db.execute(_INSERT, values)
            self._db.commit()

    def label(self, decision_id: str, labels: dict[str, str]) -> None:
        with self._lock:
            found = self._db.execute("SELECT 1 FROM decisions WHERE id = ?", (decision_id,)).fetchone()
            if not found:
                raise KeyError("no decision with id %r in %s" % (decision_id, self.path))
            now = time.time()
            self._db.executemany(
                "INSERT OR REPLACE INTO labels VALUES (?,?,?,?)",
                [(decision_id, f, lab, now) for f, lab in labels.items()],
            )
            self._db.commit()

    def rows(self, schema: str) -> list[dict[str, Any]]:
        with self._lock:
            decisions = self._db.execute(
                "SELECT %s FROM decisions WHERE schema = ? ORDER BY ts, id" % ", ".join(_COLUMNS), (schema,)
            ).fetchall()
            labels: dict[str, dict[str, str]] = {}
            for did, field, lab in self._db.execute(
                "SELECT l.id, l.field, l.label FROM labels l JOIN decisions d ON d.id = l.id WHERE d.schema = ?",
                (schema,),
            ):
                labels.setdefault(did, {})[field] = lab
        out = []
        for rec in decisions:
            row = dict(zip(_COLUMNS, rec))
            for k in _JSON:
                row[k] = _load(row[k])
            row["labels"] = labels.get(row["id"], {})
            out.append(row)
        return out

    def forget(self, decision_id: str | None = None, before: float | None = None) -> int:
        """Delete one decision (and its labels), or every decision logged before a timestamp. Returns rows deleted.

        Deleted bytes are overwritten (`secure_delete`) and the write-ahead log is checkpointed and truncated, so the
        text is gone from the database files, not only hidden.
        """
        where, args = ("id = ?", (decision_id,)) if decision_id is not None else ("ts < ?", (before,))
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM decisions WHERE %s" % where, args)]
            for table in ("labels", "trained", "decisions"):
                self._db.executemany("DELETE FROM %s WHERE id = ?" % table, [(i,) for i in ids])
            self._db.commit()
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return len(ids)

    def mark_trained(self, ids: Iterable[str], run: str) -> None:
        with self._lock:
            self._db.executemany("INSERT OR IGNORE INTO trained VALUES (?,?)", [(i, run) for i in ids])
            self._db.commit()

    def trained(self) -> set[str]:
        with self._lock:
            return {r[0] for r in self._db.execute("SELECT id FROM trained")}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            r = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if r is None else _load(r[0])

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, _dump(value)))
            self._db.commit()
