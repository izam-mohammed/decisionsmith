"""Hugging Face datasets (`uv add "decisionsmith[datasets]"`): decide a dataset's texts with `map`, and train on
a dataset by name.

dataset = dataset.map(ds_map(h), batched=True)            # + team, team_confidence, team_source, ...
model.train("hf:my-org/tickets")                           # or ds.finetune("hf:my-org/tickets:train", Ticket)
model.evaluate("hf:my-org/tickets:test")                   # a split, or a slice like "train[:10%]"
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

from ._base import decide_columns

_SPEC = re.compile(r"^(?P<name>[^:]+?)(?::(?P<split>[\w.-]+(?:\[[^\]]*\])?))?$")


def ds_map(
    x: Any, column: str = "text", fields: Any = None, *, prefix: str = "", batch_size: int = 256
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """A function for `dataset.map(..., batched=True)` that adds the decision of each row's `column` text: a column
    per field, plus `<field>_confidence` and `<field>_source`. It also works without `batched=True`."""

    def decide(batch: dict[str, Any]) -> dict[str, Any]:
        texts = batch[column]
        if isinstance(texts, str) or texts is None:
            return {k: v[0] for k, v in decide_columns(x, [texts], fields, prefix=prefix).items()}
        return decide_columns(x, list(texts), fields, prefix=prefix, batch_size=batch_size)

    return decide


def records(spec: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """The rows of `hf:<name>` or `hf:<name>:<split>` (default split `train`) for training and evaluation: a hub
    dataset id or a local folder of CSV / JSON / Parquet files, loaded with `datasets.load_dataset`. `ClassLabel`
    columns become their label names (-1, no label, becomes blank)."""
    import datasets

    m = _SPEC.match(spec)
    if not m:
        raise ValueError(
            "expected hf:<dataset> or hf:<dataset>:<split>, e.g. hf:my-org/tickets:train, got hf:%s" % spec
        )
    name, split = m["name"], m["split"] or "train"
    data = datasets.load_dataset(name, split=split)
    named = {k: f for k, f in data.features.items() if isinstance(f, datasets.ClassLabel)}
    for i, row in enumerate(data):
        for k, f in named.items():
            row[k] = f.int2str(row[k]) if row[k] is not None and row[k] >= 0 else None
        yield "hf:%s:%s row %d" % (name, split, i), {"_csv": True, **row}
