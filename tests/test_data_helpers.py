"""The shared helpers behind the data integrations (pandas, Polars, datasets, DuckDB, Spark, Dask, Ray, Chroma,
Qdrant): batched decisions into columns, field types, lazy builders, and the `hf:` training input."""

import enum
import pickle
from typing import Literal

import pytest
from pydantic import BaseModel

import decisionsmith as ds
from decisionsmith.integrations import _base
from decisionsmith.testing import FakeEngine
from decisionsmith.training import data
from tests.conftest import Ticket, truth


class Colour(enum.Enum):
    red = "red"
    blue = "blue"


class Mixed(BaseModel):
    colour: Colour
    stars: Literal[1, 2, 3]
    weight: Literal[0.5, 1.5]
    odd: Literal["a", 1]
    ok: bool


class Counting(FakeEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.batches = []

    def ask_many(self, texts, questions):
        self.batches.append(len(texts))
        return [self.ask(t, questions) for t in texts]


def test_plain_value_types_and_fields():
    m = ds.model(Mixed, FakeEngine())
    assert _base.plain(Colour.red) == "red" and _base.plain(3) == 3
    assert [_base.value_type(m, f) for f in ["colour", "stars", "weight", "odd", "ok"]] == [str, int, float, str, bool]
    assert _base.pick_fields(m) == ["colour", "stars", "weight", "odd", "ok"]
    assert _base.pick_fields(m, "ok") == ["ok"] and _base.pick_fields(m, ["ok", "colour"]) == ["ok", "colour"]
    for bad in ("nope", []):
        with pytest.raises(ValueError, match="unknown fields"):
            _base.pick_fields(m, bad)
    assert _base.columns(["a"], "p_") == ["p_a", "p_a_confidence", "p_a_source"]


def test_decide_columns_with_a_model_batches_and_skips_empty_texts():
    engine = Counting(lambda t: {"label": "spam" if "win" in t else "ham"}, confidence=0.8)
    m = ds.model(["spam", "ham"], engine)
    texts = ["win money", None, "  ", "lunch?", "win again", 3]
    got = _base.decide_columns(m, texts, batch_size=2)
    assert got["label"] == ["spam", None, None, "ham", "spam", None]
    assert got["label_confidence"] == [0.8, None, None, 0.8, 0.8, None]
    assert got["label_source"] == ["student", None, None, "student", "student", None]
    assert engine.batches == [2, 1]
    assert _base.decide_columns(m, [], prefix="p_") == {"p_label": [], "p_label_confidence": [], "p_label_source": []}


def test_decide_columns_applies_a_saved_calibration_and_plain_values():
    m = ds.model(Mixed, FakeEngine(confidence=0.9))
    m.calibration = {"ok": {"temperature": 3.0}}
    got = _base.decide_columns(m, ["x"], ["ok", "colour"])
    assert got["ok_confidence"][0] < 0.9 and got["colour_confidence"][0] == 0.9
    assert got["colour"][0] in ("red", "blue") and isinstance(got["ok"][0], bool)


def test_decide_columns_with_a_harness_keeps_sources(db):
    student = FakeEngine(truth, confidence=lambda text, q: 0.95 if "invoice" in text else 0.5, name="s")
    h = ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0, name="t"), student=student, log=db, audit=0)
    got = _base.decide_columns(h, ["my invoice is wrong", "sync is broken", ""], "team", prefix="p_")
    assert got == {
        "p_team": ["billing", "technical", None],
        "p_team_confidence": [0.95, 1.0, None],
        "p_team_source": ["student", "teacher", None],
    }
    assert len(h.log.rows("Ticket")) == 2


CALLS = []


def build():
    CALLS.append(1)
    return ds.model(Ticket, FakeEngine(truth))


def test_lazy_builds_once_and_never_pickles_what_it_built():
    calls = CALLS
    calls.clear()
    lazy = _base.Lazy(build)
    assert lazy.get() is lazy.get() and calls == [1]
    again = pickle.loads(pickle.dumps(lazy))
    assert again.x is None and again.get() is lazy.get() and calls == [1]
    _base._BUILT.clear()  # as in a new worker process
    fresh = pickle.loads(pickle.dumps(lazy))
    assert fresh.get() is not lazy.get() and calls == [1, 1]
    assert fresh.__dask_tokenize__() == lazy.__dask_tokenize__() != _base.Lazy(build).__dask_tokenize__()
    kept = pickle.loads(pickle.dumps(_base.Lazy(ds.model(Ticket, FakeEngine(truth)))))
    assert kept.make is None and kept.get().predict("sync is broken").team == "technical"
    with pytest.raises(TypeError, match="function that returns one"):
        _base.Lazy("nope")
    with pytest.raises(TypeError, match="expected ds.model"):
        _base.Lazy(lambda: "nope").get()


def test_hf_input_reads_rows_from_the_datasets_integration(monkeypatch):
    import decisionsmith.integrations.datasets as hf

    def records(spec):
        assert spec == "my-org/tickets:test"
        yield "hf row 0", {"_csv": True, "text": "you charged me twice", "team": "billing", "wants_refund": None}
        yield "hf row 1", {"_csv": True, "text": "sync is broken", "team": "technical", "wants_refund": False}

    monkeypatch.setattr(hf, "records", records)
    rows = data.load("hf:my-org/tickets:test", Ticket)
    assert [sorted(r.targets) for r in rows] == [["team"], ["team", "wants_refund"]]

    def no_text(spec):
        yield "hf row 0", {"_csv": True, "team": "billing"}

    monkeypatch.setattr(hf, "records", no_text)
    with pytest.raises(data.DataError, match="needs a 'text' column"):
        data.load("hf:x", Ticket)
