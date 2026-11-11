"""DuckDB (`uv add "decisionsmith[duckdb]"`): decisions as SQL functions, batched (vectorised Arrow UDFs).

register(con, h)
con.sql("SELECT text, ds_decide(text, 'team') AS team, ds_confidence(text, 'team') FROM tickets")
con.sql("SELECT * FROM tickets WHERE ds_is_true(text)")         # the schema's yes/no field (or field=...)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ._base import decide_columns, pick_fields, value_type

CACHE = 4096


def _sql(value: Any) -> str | None:
    return None if value is None else str(value).lower() if isinstance(value, bool) else str(value)


def register(con: Any, x: Any, *, field: str | None = None, prefix: str = "ds") -> Any:
    """Add `<prefix>_decide(text, field) -> VARCHAR`, `<prefix>_confidence(text, field) -> DOUBLE` and, when there
    is a yes/no field (`field`, or the only `bool` field), `<prefix>_is_true(text) -> BOOLEAN` to `con`. A text
    decided once is reused by every function in the same session (the last 4096 texts), so asking for several
    fields decides each text once. NULL texts give NULL."""
    import pyarrow as pa

    names = pick_fields(x)
    flags = [f for f in names if value_type(x, f) is bool]
    if field is not None and field not in flags:
        raise ValueError("field %r is not a yes/no (bool) field; bool fields: %s" % (field, flags))
    flag = field or (flags[0] if len(flags) == 1 else None)
    seen: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def rows(texts: list[str]) -> list[dict[str, Any]]:
        new = list(dict.fromkeys(t for t in texts if t not in seen))
        got = decide_columns(x, new, names)
        for i, t in enumerate(new):
            seen[t] = {c: v[i] for c, v in got.items()}
        for t in texts:
            seen.move_to_end(t)
        out = [seen[t] for t in texts]
        while len(seen) > CACHE:
            seen.popitem(last=False)
        return out

    def column(texts: Any, fields: Any, end: str) -> list[Any]:
        asked = fields.to_pylist()
        unknown = sorted(set(asked) - set(names))
        if unknown:
            raise ValueError("unknown fields %s; this model decides %s" % (unknown, names))
        return [r[f + end] for r, f in zip(rows(texts.to_pylist()), asked)]

    def decide(texts: Any, fields: Any) -> Any:
        return pa.array([_sql(v) for v in column(texts, fields, "")], pa.string())

    def conf(texts: Any, fields: Any) -> Any:
        return pa.array(column(texts, fields, "_confidence"), pa.float64())

    arrow: dict[str, Any] = {"type": "arrow", "null_handling": "special"}
    con.create_function(prefix + "_decide", decide, ["VARCHAR", "VARCHAR"], "VARCHAR", **arrow)
    con.create_function(prefix + "_confidence", conf, ["VARCHAR", "VARCHAR"], "DOUBLE", **arrow)
    if flag is not None:

        def is_true(texts: Any) -> Any:
            return pa.array([r[flag] for r in rows(texts.to_pylist())], pa.bool_())

        con.create_function(prefix + "_is_true", is_true, ["VARCHAR"], "BOOLEAN", **arrow)
    return con
