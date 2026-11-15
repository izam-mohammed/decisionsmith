import importlib
import sys

import pytest

pytest.importorskip("pandas")

import pandas as pd

import decisionsmith as ds
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth


def test_accessor_adds_columns_for_every_field(db):
    import decisionsmith.integrations.pandas  # noqa: F401

    h = ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0), log=db)
    df = pd.DataFrame({"text": ["refund my card charge", None, "sync is broken"], "id": [1, 2, 3]})
    out = df.ds.decide("text", h)
    assert list(out.columns) == [
        "text",
        "id",
        "team",
        "team_confidence",
        "team_source",
        "wants_refund",
        "wants_refund_confidence",
        "wants_refund_source",
    ]
    assert out["team"].tolist()[::2] == ["billing", "technical"] and pd.isna(out["team"][1])
    assert out["wants_refund"].tolist()[::2] == [True, False]
    assert out["team_source"][0] == "teacher" and out["team_confidence"][0] == 1.0
    assert list(df.columns) == ["text", "id"]


def test_decide_function_with_prefix_fields_and_a_model():
    from decisionsmith.integrations.pandas import decide

    m = ds.model(["billing", "technical", "sales"], FakeEngine(lambda t: {"label": truth(t)["team"]}))
    df = pd.DataFrame({"body": ["you charged me twice", "can I get a quote"], "label": ["billing", "sales"]})
    out = decide(df, "body", m, prefix="pred_", batch_size=1)
    assert (out["pred_label"] == out["label"]).all() and set(out["pred_label_source"]) == {"student"}
    h = ds.harness(Ticket, teacher=FakeEngine(truth), log=None)
    assert list(df.ds.decide("body", h, "team").columns) == ["body", "label", "team", "team_confidence", "team_source"]
    with pytest.raises(KeyError, match="no column 'text'"):
        decide(df, "text", m)


def test_register_is_idempotent_and_import_works_without_pandas(monkeypatch):
    import decisionsmith.integrations.pandas as mod

    mod.register()
    assert pd.DataFrame.ds is mod.DecisionAccessor
    monkeypatch.setitem(sys.modules, "pandas", None)
    importlib.reload(mod)
    with pytest.raises(ImportError):
        mod.register()
    monkeypatch.undo()
    importlib.reload(mod)
    assert pd.DataFrame.ds is mod.DecisionAccessor
