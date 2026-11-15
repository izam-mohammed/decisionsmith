"""pandas (`uv add "decisionsmith[pandas]"`): decide a text column of a DataFrame into new columns, batched.

import decisionsmith.integrations.pandas            # adds the `.ds` accessor to every DataFrame
df = df.ds.decide("text", h)                        # + team, team_confidence, team_source, wants_refund, ...
df = decide(df, "text", ds.model(["spam", "ham"]), prefix="pred_")   # + pred_label, pred_label_confidence, ...

Importing this module imports pandas (when it is installed) to register the accessor.
"""

from __future__ import annotations

from typing import Any

from ._base import decide_columns


def decide(df: Any, column: str, x: Any, fields: Any = None, *, prefix: str = "", batch_size: int = 256) -> Any:
    """A copy of `df` with the decision for each row's `column` text: a column per field, plus `<field>_confidence`
    and `<field>_source` (`"student"` or `"teacher"`). Rows with a missing or empty text get empty values.
    `fields` picks some fields; `prefix` keeps the new columns apart from yours (e.g. your own labels)."""
    if column not in df.columns:
        raise KeyError("no column %r; the DataFrame has %s" % (column, list(df.columns)))
    return df.assign(**decide_columns(x, df[column].tolist(), fields, prefix=prefix, batch_size=batch_size))


class DecisionAccessor:
    """`df.ds.decide(column, x, fields=None, prefix="", batch_size=256)`: see `decide`."""

    def __init__(self, df: Any) -> None:
        self._df = df

    def decide(self, column: str, x: Any, fields: Any = None, *, prefix: str = "", batch_size: int = 256) -> Any:
        return decide(self._df, column, x, fields, prefix=prefix, batch_size=batch_size)


def register() -> None:
    """Add `df.ds` to pandas DataFrames (done on import when pandas is installed)."""
    import pandas as pd

    if getattr(pd.DataFrame, "ds", None) is not DecisionAccessor:
        pd.api.extensions.register_dataframe_accessor("ds")(DecisionAccessor)


try:
    register()
except ImportError:
    pass
