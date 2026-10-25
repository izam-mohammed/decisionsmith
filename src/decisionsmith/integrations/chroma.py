"""Chroma (`uv add "decisionsmith[chroma]"`): tag documents with decisions at ingest, and filter what a query
returns. Nothing here imports chromadb; it works on any Chroma collection and result.

add(collection, h, ids=ids, documents=docs, embeddings=vectors)          # metadata: team, team_confidence, ...
collection.query(query_embeddings=[q], where={"team": "billing"})         # then filter on the tags as usual
filter_results(judge, collection.query(query_texts=[question]), query=question)   # drop off-topic hits
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._base import decide_columns, resolve


def tag(
    x: Any, documents: list[Any], metadatas: list[Any] | None = None, *, fields: Any = None, prefix: str = ""
) -> list[dict[str, Any] | None]:
    """Metadata for each document: yours (if given) plus a key per decision field and `<field>_confidence` /
    `<field>_source`. Empty documents get no decision keys, and `None` when nothing is left (Chroma metadata can't
    hold nulls or be empty)."""
    got = decide_columns(x, list(documents), fields, prefix=prefix)
    base = list(metadatas) if metadatas is not None else [None] * len(documents)
    return [{**(m or {}), **{c: v[i] for c, v in got.items() if v[i] is not None}} or None for i, m in enumerate(base)]


def add(
    collection: Any,
    x: Any,
    *,
    ids: Any,
    documents: list[Any],
    metadatas: list[Any] | None = None,
    fields: Any = None,
    prefix: str = "",
    **kwargs: Any,
) -> None:
    """`collection.add(...)` with `metadatas=tag(x, documents, metadatas)`; other arguments pass through."""
    tags = tag(x, documents, metadatas, fields=fields, prefix=prefix)
    collection.add(ids=ids, documents=documents, metadatas=tags, **kwargs)


def filter_results(
    x: Any, results: Any, field: str | None = None, keep: Iterable[Any] = ("relevant",), *, query: Any = None
) -> dict[str, Any]:
    """`results` (from `collection.query` or `collection.get`, with documents included) without the hits whose
    decision `field` is not one of `keep`. The model reads `Query: ...` and `Document: ...` when `query` is given
    (one text, or one per query), else the document alone. `field` can be left out when the model has one field."""
    d, kept = resolve(x), list(keep)
    if field is None and len(d.fields) != 1:
        raise ValueError("this model decides %s; pass field=... to filter on one of them" % d.fields)
    nested = bool(results["ids"]) and isinstance(results["ids"][0], list)
    groups = results["documents"] if nested else [results["documents"]]
    queries = [query] * len(groups) if query is None or isinstance(query, str) else list(query)
    texts = [
        doc if doc is None or not q else "Query: %s\n\nDocument: %s" % (q, doc)
        for q, docs in zip(queries, groups)
        for doc in docs
    ]
    ok = iter([v in kept for v in decide_columns(x, texts, field)[field or d.fields[0]]])
    masks = [[next(ok) for _ in docs] for docs in groups]
    out = dict(results)
    for key, value in results.items():
        if key != "included" and value is not None:
            rows = value if nested else [value]
            filtered = [[v for v, m in zip(row, mask) if m] for row, mask in zip(rows, masks)]
            out[key] = filtered if nested else filtered[0]
    return out
