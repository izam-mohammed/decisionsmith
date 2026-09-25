"""Ray Data (`uv add "decisionsmith[ray]"`): decide a text column with `Dataset.map_batches`, batched.

ds_data.map_batches(Decide(model))                                              # tasks: the model is pickled
ds_data.map_batches(Decide, fn_constructor_args=(build,), concurrency=2)       # actors: build() runs once in each

`build` is a function that returns your `ds.model(...)` or `ds.harness(...)`: use it for a harness (its log can't be
pickled) or a Laya model you want loaded once in each actor rather than pickled with every task.
"""

from __future__ import annotations

from typing import Any

from ._base import Lazy, decide_columns


class Decide:
    """A `map_batches` function (an instance) or callable class (with `fn_constructor_args`) that adds the decision
    of each row's `column` text: a column per field, plus `<field>_confidence` and `<field>_source`. Works with every
    `batch_format` that is a dict of columns or a pandas DataFrame."""

    def __init__(
        self, x: Any, column: str = "text", fields: Any = None, *, prefix: str = "", batch_size: int = 256
    ) -> None:
        self.x, self.column, self.fields, self.prefix, self.batch_size = Lazy(x), column, fields, prefix, batch_size

    def __call__(self, batch: Any) -> Any:
        import numpy as np

        texts = batch[self.column]
        got = decide_columns(self.x.get(), list(texts), self.fields, prefix=self.prefix, batch_size=self.batch_size)
        if isinstance(batch, dict):
            conf = {c: np.array([np.nan if v is None else v for v in got[c]]) for c in got if c.endswith("_confidence")}
            return {**batch, **{c: np.array(v, dtype=object) for c, v in got.items()}, **conf}
        return batch.assign(**got)
