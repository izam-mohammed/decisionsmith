"""Qdrant (`uv add "decisionsmith[qdrant]"`): tag points with decisions at ingest, and filter retrieved points.

client.upsert("tickets", points=points(h, ids, vectors, texts))           # payload: text, team, team_confidence, ...
client.query_points("tickets", query=q, query_filter=Filter(must=[FieldCondition(key="team", match=...)]))
filter_points(judge, client.query_points("docs", query=q), query=question)  # drop off-topic hits
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._base import decide_columns, resolve


def tag(
    x: Any, texts: list[Any], payloads: list[Any] | None = None, *, fields: Any = None, prefix: str = ""
) -> list[dict[str, Any]]:
    """A payload for each text: yours (if given) plus a key per decision field and `<field>_confidence` /
    `<field>_source` (null for an empty text)."""
    got = decide_columns(x, list(texts), fields, prefix=prefix)
    base = list(payloads) if payloads is not None else [None] * len(texts)
    return [{**(p or {}), **{c: v[i] for c, v in got.items()}} for i, p in enumerate(base)]


def points(
    x: Any,
    ids: list[Any],
    vectors: list[Any],
    texts: list[Any],
    payloads: list[Any] | None = None,
    *,
    text_key: str = "text",
    fields: Any = None,
    prefix: str = "",
) -> list[Any]:
    """`PointStruct`s for `client.upsert`: each payload holds the text (under `text_key`) and the decision tags."""
    from qdrant_client import models

    tags = tag(x, texts, payloads, fields=fields, prefix=prefix)
    return [
        models.PointStruct(id=i, vector=v, payload={text_key: t, **p}) for i, v, t, p in zip(ids, vectors, texts, tags)
    ]


def filter_points(
    x: Any,
    hits: Any,
    field: str | None = None,
    keep: Iterable[Any] = ("relevant",),
    *,
    query: str | None = None,
    text_key: str = "text",
) -> list[Any]:
    """The retrieved points (a `QueryResponse` or a list of points) whose decision `field` is one of `keep`. The
    model reads `Query: ...` and `Document: ...` when `query` is given, else the text in `payload[text_key]`.
    `field` can be left out when the model has one field."""
    d, kept = resolve(x), list(keep)
    if field is None and len(d.fields) != 1:
        raise ValueError("this model decides %s; pass field=... to filter on one of them" % d.fields)
    found = list(getattr(hits, "points", hits))
    docs = [(p.payload or {}).get(text_key) for p in found]
    texts = [doc if doc is None or not query else "Query: %s\n\nDocument: %s" % (query, doc) for doc in docs]
    values = decide_columns(x, texts, field)[field or d.fields[0]]
    return [p for p, v in zip(found, values) if v in kept]
