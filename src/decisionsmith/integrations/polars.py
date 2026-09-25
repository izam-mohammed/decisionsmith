"""Polars (`uv add "decisionsmith[polars]"`): a decision as a Polars expression, batched through `map_batches`.

df.with_columns(decide_expr(h, "text", "team"))                   # one field: a `team` column
df.with_columns(decide_expr(h, "text")).unnest("decision")         # every field + <field>_confidence, _source
pl.scan_csv("tickets.csv").with_columns(decide_expr(model, "text")).collect()   # lazy frames too
"""

from __future__ import annotations

from typing import Any

from ._base import columns, decide_columns, pick_fields, value_type


def _dtypes(x: Any, names: list[str], prefix: str) -> dict[str, Any]:
    import polars as pl

    kinds = {bool: pl.Boolean, int: pl.Int64, float: pl.Float64, str: pl.String}
    types = [kinds[value_type(x, f)] for f in names]
    return dict(zip(columns(names, prefix), (t for dtype in types for t in (dtype, pl.Float64, pl.String))))


def decide_expr(
    x: Any, column: Any = "text", field: str | None = None, *, prefix: str = "", batch_size: int = 256
) -> Any:
    """An expression deciding the texts in `column` (a name or an expression). With `field`, it is that field's
    values, named `<prefix><field>`; without, a struct named `decision` with a column per field plus
    `<field>_confidence` and `<field>_source`. Missing or empty texts give nulls."""
    import polars as pl

    names = pick_fields(x, field)
    dtypes = _dtypes(x, names, prefix)
    wanted = prefix + field if field else None

    def run(texts: Any) -> Any:
        got = decide_columns(x, texts.to_list(), names, prefix=prefix, batch_size=batch_size)
        cols = {c: [v if v is None or dtypes[c] != pl.String else str(v) for v in got[c]] for c in dtypes}
        if wanted:
            return pl.Series(wanted, cols[wanted], dtype=dtypes[wanted])
        return pl.DataFrame(cols, schema=dtypes).to_struct("decision")

    expr = pl.col(column) if isinstance(column, str) else column
    out = pl.Struct(dtypes) if wanted is None else dtypes[wanted]
    return expr.map_batches(run, return_dtype=out, is_elementwise=True).alias(wanted or "decision")
