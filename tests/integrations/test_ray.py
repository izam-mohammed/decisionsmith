import os
from pathlib import Path

import pytest

pytest.importorskip("ray")

import numpy as np
import pandas as pd
import ray
import ray.data

import decisionsmith as ds
from decisionsmith.integrations.ray import Decide
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, truth

ROOT = str(Path(__file__).resolve().parents[2])


def build():
    return ds.harness(Ticket, teacher=FakeEngine(truth, confidence=1.0), log=None)


@pytest.fixture(scope="module")
def cluster():
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
        _node_ip_address="127.0.0.1",
        runtime_env={"env_vars": {"PYTHONPATH": ROOT}},
    )
    yield
    ray.shutdown()


def items():
    return ray.data.from_items([{"text": "refund my card charge"}, {"text": None}, {"text": "sync is broken"}])


def by_text(rows, column):
    return {r["text"] if isinstance(r["text"], str) else None: r[column] for r in rows}


EXPECTED = {"refund my card charge": "billing", None: None, "sync is broken": "technical"}


def test_tasks_with_a_model(cluster):
    m = ds.model(["billing", "technical", "sales"], FakeEngine(lambda t: {"label": truth(t)["team"]}))
    rows = items().map_batches(Decide(m, prefix="p_")).take_all()
    assert by_text(rows, "p_label") == EXPECTED
    assert by_text(rows, "p_label_source")["sync is broken"] == "student"
    assert np.isnan(by_text(rows, "p_label_confidence")[None])


def test_actors_build_a_harness_once_each(cluster):
    out = items().map_batches(
        Decide,
        fn_constructor_args=(build,),
        fn_constructor_kwargs={"fields": "team"},
        concurrency=1,
        batch_format="pandas",
    )
    rows = out.take_all()
    assert by_text(rows, "team") == EXPECTED and "wants_refund" not in rows[0]
    assert by_text(rows, "team_source")["sync is broken"] == "teacher"


def test_the_batch_function_directly():
    d = Decide(build, batch_size=1)
    got = d({"text": np.array(["you charged me twice", None], dtype=object)})
    assert got["team"].tolist() == ["billing", None] and got["team_confidence"][0] == 1.0
    frame = d(pd.DataFrame({"text": ["sync is broken"]}))
    assert frame["wants_refund"].tolist() == [False]
