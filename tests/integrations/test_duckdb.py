from typing import Literal

import pytest

pytest.importorskip("duckdb")

import duckdb
from pydantic import BaseModel

import decisionsmith as ds
from decisionsmith.integrations import duckdb as dsdb
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth


class Flags(BaseModel):
    spam: bool
    rude: bool


class Stars(BaseModel):
    stars: Literal[1, 2, 3]


def tickets(con):
    con.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES ('refund my card charge'), (NULL), ('sync is broken')) v(text)"
    )
    return con


def test_decide_confidence_and_is_true_decide_each_text_once():
    teacher = FakeEngine(truth, confidence=1.0)
    con = dsdb.register(tickets(duckdb.connect()), ds.harness(Ticket, teacher=teacher, log=None))
    rows = con.sql(
        "SELECT ds_decide(text, 'team'), ds_decide(text, 'wants_refund'), ds_confidence(text, 'team') FROM t"
    ).fetchall()
    assert rows == [("billing", "true", 1.0), (None, None, None), ("technical", "false", 1.0)]
    assert con.sql("SELECT text FROM t WHERE ds_is_true(text)").fetchall() == [("refund my card charge",)]
    assert len(teacher.calls) == 2
    with pytest.raises(duckdb.Error, match="unknown fields"):
        con.sql("SELECT ds_decide(text, 'nope') FROM t").fetchall()


def test_cache_is_bounded_and_empty_texts_give_null(monkeypatch):
    monkeypatch.setattr(dsdb, "CACHE", 2)
    engine = FakeEngine(lambda t: {"label": "b"})
    con = dsdb.register(duckdb.connect(), ds.model(["a", "b"], engine), prefix="m")
    con.execute("CREATE TABLE u AS SELECT * FROM (VALUES ('x'), ('y'), ('z'), ('')) v(text)")
    assert con.sql("SELECT m_decide(text, 'label') FROM u").fetchall() == [("b",), ("b",), ("b",), (None,)]
    assert con.sql("SELECT m_decide(text, 'label') FROM u").fetchall()[0] == ("b",)
    assert len(engine.calls) == 5
    with pytest.raises(duckdb.CatalogException):
        con.sql("SELECT m_is_true(text) FROM u").fetchall()


def test_is_true_needs_one_bool_field_or_field():
    m = ds.model(Flags, FakeEngine(lambda t: {"spam": "win" in t, "rude": False}))
    con = dsdb.register(duckdb.connect(), m, field="spam")
    assert con.sql("SELECT ds_is_true('win a prize'), ds_is_true('hello')").fetchall() == [(True, False)]
    con = dsdb.register(duckdb.connect(), m)
    with pytest.raises(duckdb.CatalogException):
        con.sql("SELECT ds_is_true('x')").fetchall()
    with pytest.raises(ValueError, match="not a yes/no"):
        dsdb.register(duckdb.connect(), ds.model(Stars, FakeEngine()), field="stars")
    con = dsdb.register(duckdb.connect(), ds.model(Stars, FakeEngine()))
    assert con.sql("SELECT ds_decide('x', 'stars')").fetchone()[0] in ("1", "2", "3")
