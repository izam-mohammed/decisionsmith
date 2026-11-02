import pytest

pytest.importorskip("dask.dataframe")

import dask.dataframe as dd
import pandas as pd

import decisionsmith as ds
from decisionsmith.integrations.dask import decide
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth


def build():
    return ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0), log=None)


def frame():
    texts = ["refund my card charge", None, "sync is broken", "how much is the plan"]
    return dd.from_pandas(pd.DataFrame({"text": texts, "id": range(4)}), npartitions=2)


def test_lazy_columns_with_a_harness(db):
    h = ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0), log=db)
    out = decide(frame(), "text", h)
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
    assert out.dtypes["team_confidence"] == "Float64" and out.dtypes["wants_refund"] == "boolean"
    got = out.compute()
    assert got["team"].tolist()[2:] == ["technical", "sales"] and got["team"].tolist()[0] == "billing"
    assert pd.isna(got["team"].tolist()[1]) and got["wants_refund"].tolist()[0] is True


def test_a_builder_on_the_process_scheduler():
    out = decide(frame(), "text", build, "team", prefix="p_", batch_size=1)
    got = out.compute(scheduler="processes", num_workers=2)
    assert got["p_team"].tolist()[2:] == ["technical", "sales"] and set(got["p_team_source"].dropna()) == {"teacher"}
    with pytest.raises(KeyError, match="no column 'body'"):
        decide(frame(), "body", build)
