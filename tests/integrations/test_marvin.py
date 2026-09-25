import ast
import asyncio
import enum
import importlib.util
import inspect
from pathlib import Path

import pytest

import decisionsmith as ds
from decisionsmith.integrations import marvin as integration
from decisionsmith.testing import FakeEngine

LABELS = ["billing", "technical", "sales"]


def fake(text):
    return {"label": next((x for x in LABELS if x in text), "sales")}


class Team(enum.Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"


def test_classify_with_a_model():
    m = ds.model(LABELS, FakeEngine(fake))
    assert integration.classify("a billing question", LABELS, model=m) == "billing"
    assert integration.classify("a technical one", Team, model=m) is Team.TECHNICAL
    h = ds.harness(m, log=None)
    assert integration.classify(42, LABELS, model=h) == "sales"
    assert asyncio.run(integration.classify_async("billing", Team, model=m)) is Team.BILLING
    with pytest.raises(ValueError, match="not one of the labels"):
        integration.classify("technical", ["billing", "sales"], model=m)


def test_unsupported_options_raise():
    m = ds.model(LABELS, FakeEngine(fake))
    for kw in ({"multi_label": True}, {"agent": object()}, {"thread": "t"}, {"context": {}}, {"prompt": "p"}):
        with pytest.raises(TypeError, match="does not support"):
            integration.classify("x", LABELS, model=m, **kw)


def test_default_model_is_cached_laya(monkeypatch):
    made, real = [], ds.model

    def model(labels, question=None):
        made.append((labels, question))
        return real(labels, FakeEngine(fake))

    monkeypatch.setattr(ds, "model", model)
    integration._MODELS.clear()
    assert integration.classify("billing", LABELS, instructions="Which team?") == "billing"
    assert integration.classify("sales", LABELS, instructions="Which team?") == "sales"
    assert made == [(LABELS, "Which team?")]


def test_signature_matches_marvin():
    spec = importlib.util.find_spec("marvin")
    if spec is None:
        pytest.skip("marvin is not installed")
    source = Path(spec.submodule_search_locations[0], "fns", "classify.py").read_text()
    theirs = {
        f.name: [a.arg for a in f.args.args + f.args.kwonlyargs] for f in ast.parse(source).body if hasattr(f, "args")
    }
    for ours in (integration.classify, integration.classify_async):
        names = theirs[ours.__name__]
        assert list(inspect.signature(ours).parameters)[: len(names)] == names
