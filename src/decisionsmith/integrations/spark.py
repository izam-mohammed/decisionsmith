"""Spark (`uv add "decisionsmith[spark]"`): a `pandas_udf` that decides a text column, batched per Arrow batch.

team = udf(build, "team")                        # build() returns your ds.model(...) or ds.harness(...)
df.withColumn("team", team("text"))
df.withColumn("d", udf(build)("text")).select("text", "d.*")   # every field + <field>_confidence, _source

`build` runs once on the driver (to read the schema) and once in each Python worker, so nothing heavy is pickled.
A `ds.model(...)` whose engine can be pickled can be passed directly instead.
"""

from __future__ import annotations

from typing import Any

from ._base import Lazy, columns, decide_columns, pick_fields, value_type


def udf(x: Any, field: str | None = None, *, prefix: str = "", batch_size: int = 256) -> Any:
    """A Spark `pandas_udf` over a string column. With `field`, it returns that field (a string, boolean or number
    column); without, a struct with a column per field plus `<field>_confidence` and `<field>_source`. NULL or
    empty texts give NULL."""
    import pandas as pd
    from pyspark.sql import types as T
    from pyspark.sql.functions import pandas_udf

    lazy = Lazy(x)
    names = pick_fields(lazy.get(), field)
    kinds = {bool: T.BooleanType(), int: T.LongType(), float: T.DoubleType(), str: T.StringType()}
    types = [kinds[value_type(lazy.get(), f)] for f in names]
    spark_types = [t for v in types for t in (v, T.DoubleType(), T.StringType())]
    cols = columns(names, prefix)

    def decide(texts: pd.Series) -> Any:
        got = decide_columns(lazy.get(), texts.tolist(), names, prefix=prefix, batch_size=batch_size)
        if field is not None:
            return pd.Series(got[prefix + field], dtype=object)
        return pd.DataFrame({c: pd.Series(got[c], dtype=object) for c in cols})

    if field is not None:
        decide.__annotations__ = {"texts": pd.Series, "return": pd.Series}
        return pandas_udf(decide, types[0])
    decide.__annotations__ = {"texts": pd.Series, "return": pd.DataFrame}
    return pandas_udf(decide, T.StructType([T.StructField(c, t) for c, t in zip(cols, spark_types)]))
