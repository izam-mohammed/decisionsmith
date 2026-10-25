from typing import Literal

import pytest

pytest.importorskip("polars")

import polars as pl
from pydantic import BaseModel

import decisionsmith as ds
from decisionsmith.integrations.polars import decide_expr
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth


class Mixed(BaseModel):
    stars: Literal[1, 2, 3]
    odd: Literal["a", 1]


def test_one_field_as_a_column():
    h = ds.harness(Ticket, teacher=FakeEngine(truth), log=None)
    df = pl.DataFrame({"text": ["refund my card charge", None, "sync is broken", ""]})
    out = df.with_columns(decide_expr(h, "text", "team")).with_columns(
        decide_expr(h, pl.col("text"), "wants_refund", prefix="p_")
    )
    assert out.schema["team"] == pl.String and out.schema["p_wants_refund"] == pl.Boolean
    assert out["team"].to_list() == ["billing", None, "technical", None]
    assert out["p_wants_refund"].to_list() == [True, None, False, None]


def test_every_field_as_a_struct_in_a_lazy_frame(tmp_path):
    path = tmp_path / "t.csv"
    pl.DataFrame({"text": ["you charged me twice", "how much is the plan"]}).write_csv(path)
    m = ds.model(["billing", "technical", "sales"], FakeEngine(lambda t: {"label": truth(t)["team"]}, confidence=0.7))
    out = pl.scan_csv(path).with_columns(decide_expr(m)).collect().unnest("decision")
    assert out.columns == ["text", "label", "label_confidence", "label_source"]
    assert out["label"].to_list() == ["billing", "sales"] and out["label_confidence"].to_list() == [0.7, 0.7]
    assert out["label_source"].to_list() == ["student", "student"]


def test_number_and_mixed_option_types():
    m = ds.model(Mixed, FakeEngine())
    out = pl.DataFrame({"text": ["x", "y"]}).with_columns(decide_expr(m, prefix="d_")).unnest("decision")
    assert out.schema["d_stars"] == pl.Int64 and out.schema["d_odd"] == pl.String
    assert set(out["d_odd"].to_list()) <= {"a", "1"}
    with pytest.raises(ValueError, match="unknown fields"):
        decide_expr(m, "text", "nope")
