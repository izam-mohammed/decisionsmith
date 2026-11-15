"""Dask (`uv add "decisionsmith[dask]"`): decide a text column of a Dask DataFrame, one batch per partition.

out = decide(ddf, "text", h)          # lazy: + team, team_confidence, team_source, ...; then out.compute()
decide(ddf, "text", build)            # build() returns your model or harness (for process-based schedulers)
"""

from __future__ import annotations

from typing import Any

from ._base import Lazy, columns, decide_columns, pick_fields, value_type

_DTYPES = {bool: "boolean", int: "Int64", float: "Float64", str: "string"}


def decide(ddf: Any, column: str, x: Any, fields: Any = None, *, prefix: str = "", batch_size: int = 256) -> Any:
    """A lazy Dask DataFrame with the decision for each row's `column` text: a column per field, plus
    `<field>_confidence` and `<field>_source` (nullable pandas dtypes). `x` is a model, a harness, or a function that
    builds one: it runs once here (for the schema) and, with a process-based scheduler or `dask.distributed`, once in
    each worker process, since nothing built is pickled. Missing or empty texts get empty values."""
    if column not in ddf.columns:
        raise KeyError("no column %r; the DataFrame has %s" % (column, list(ddf.columns)))
    lazy = Lazy(x)
    names = pick_fields(lazy.get(), fields)
    kinds = [_DTYPES[value_type(lazy.get(), f)] for f in names]
    dtypes = dict(zip(columns(names, prefix), (t for k in kinds for t in (k, "Float64", "string"))))
    meta = {**ddf.dtypes.to_dict(), **dtypes}
    return ddf.map_partitions(_partition, lazy, column, names, prefix, batch_size, dtypes, meta=meta)


def _partition(
    df: Any, lazy: Lazy, column: str, names: list[str], prefix: str, batch_size: int, dtypes: dict[str, str]
) -> Any:
    got = decide_columns(lazy.get(), df[column].tolist(), names, prefix=prefix, batch_size=batch_size)
    return df.assign(**got).astype(dtypes)
