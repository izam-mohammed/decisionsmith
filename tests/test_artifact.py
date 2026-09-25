import enum
import json
import os
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

import decisionsmith as ds
from decisionsmith import artifact
from decisionsmith.schema import compile_schema
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, corpus, truth

LABELS = ["billing", "technical", "sales"]


def pairs(n):
    return [(t, truth(t)["team"]) for t in corpus(n)]


def ticket_rows(n):
    return [{"text": t, **truth(t)} for t in corpus(n)]


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Rich(BaseModel):
    """An order."""

    colour: Colour
    urgency: Annotated[Literal["low", "medium", "high"], ds.Scale, ds.Options(high="today")]
    fragile: Annotated[bool, ds.Options(true="breaks", false="sturdy")]
    size: Literal["s", "m"] = Field(description="Which size?")
    note: str = ""


def test_describe_and_rebuild_ask_the_same_questions():
    for cls in (Rich, Ticket):
        schema = compile_schema(cls)
        again = compile_schema(artifact.rebuild(artifact.describe(schema)))
        assert again.questions() == schema.questions()
        assert again.fingerprint == schema.fingerprint and again.name == schema.name


def test_version_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert artifact.version_path(None, "Ticket Model") == os.path.join("models", "ticket-model-v1")
    assert artifact.version_path(None, "!!") == os.path.join("models", "model-v1")
    os.makedirs("out/t-v1")
    assert artifact.version_path("out/t", "x") == os.path.join("out", "t-v2")
    assert artifact.version_path("out/t-v7", "x") == os.path.join("out", "t-v7")
    with pytest.raises(FileExistsError, match="never overwritten"):
        artifact.version_path("out/t-v1", "x")


def test_evaluate_go_and_thresholds():
    m = ds.model(Ticket, FakeEngine(truth, confidence=1.0))
    report = m.evaluate(ticket_rows(120))
    assert report.kind == "evaluate" and report.go and m.report is report
    assert report.reasons == ["ready for the harness"]
    row = report.rows[0]
    assert row["field"] == "team" and row["decisions"] == 120 and row["accuracy"] == 1.0 and row["coverage"] == 1.0
    assert m.calibration == {"team": {"threshold": 1.0}, "wants_refund": {"threshold": 1.0}}
    d = report.details
    assert d["worst"] == [] and d["rows"] == 120 and d["ms_per_text"] >= 0 and d["fields"]["team"]["confusions"] == {}
    assert "evaluate" in str(report)


def test_evaluate_no_go_reasons():
    m = ds.model(LABELS, FakeEngine(lambda t: {"label": truth(t)["team"]}, accuracy=0.5, confidence=0.9))
    report = m.evaluate(pairs(24))
    assert not report.go
    text = " ".join(report.reasons)
    assert "only 24 test decisions" in text and "fewer than 10" in text and "calibration error" in text
    assert "no confidence level reaches 97%" in text and m.calibration["label"]["threshold"] is None
    worst = report.details["worst"]
    assert worst and set(worst[0]) == {"field", "text", "gold", "pred", "confidence"}
    assert all(" -> " in k for k in report.details["fields"]["label"]["confusions"])


def test_evaluate_missing_field_and_empty():
    m = ds.model(Ticket, FakeEngine(truth, confidence=1.0))
    report = m.evaluate([{"text": t, "team": truth(t)["team"]} for t in corpus(120)])
    assert "wants_refund: no labelled rows for this field" in report.reasons
    assert report.rows[1] == {"field": "wants_refund", "decisions": 0}
    with pytest.raises(ValueError, match="no labelled rows"):
        m.evaluate([{"text": "hi"}])


def test_save_needs_a_trained_model():
    with pytest.raises(ValueError, match="nothing to save yet"):
        ds.model(LABELS, "fake").save()


def test_save_and_load_labels(tiny, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = ds.model(LABELS, str(tiny), question="Which team?")
    m.trained = str(tiny)
    first = m.save()
    assert first == os.path.join("models", "label-v1")
    assert not os.path.exists(os.path.join(first, "report.json"))
    m.evaluate(pairs(30))
    path = m.save("models/team")
    assert path == os.path.join("models", "team-v1") and m.save("models/team").endswith("team-v2")
    files = set(os.listdir(path))
    assert {"decisionsmith.json", "report.json", "MODEL_CARD.md", "model.safetensors", "rl_agent_config.json"} <= files
    meta = json.loads(open(os.path.join(path, "decisionsmith.json")).read())
    assert meta["format"] == "decisionsmith.model/1" and meta["kind"] == "labels" and meta["version"] == 1
    assert meta["labels"] == LABELS and meta["question"] == "Which team?" and meta["name"] == "team-v1"
    assert meta["thresholds"] == {"label": m.calibration["label"]["threshold"]} and meta["calibration"] == {}
    card = open(os.path.join(path, "MODEL_CARD.md")).read()
    assert "labels: billing, technical, sales" in card and "no-go" in card and "Laya" in card

    loaded = ds.load(path)
    assert loaded.simple and loaded.path == path and loaded.info["name"] == "team-v1"
    assert loaded.schema.questions() == m.schema.questions()
    assert loaded.calibration == {"label": {"threshold": m.calibration["label"]["threshold"]}}
    assert loaded.report is not None and loaded.report.kind == "evaluate" and loaded.report.go is False
    assert loaded.predict("you charged me twice") in LABELS
    assert ds.load(path, LABELS).simple
    with pytest.raises(ValueError, match="does not match"):
        ds.load(path, ["spam", "ham"])
    again = loaded.save()
    assert again == os.path.join("models", "label-v2")


def test_save_and_load_class(tiny, tmp_path):
    m = ds.model(Ticket, str(tiny))
    m.trained = str(tiny)
    m.calibration = {"team": {"temperature": 1.5, "threshold": 0.7}}
    path = m.save(str(tmp_path / "ticket"))
    meta = json.loads(open(os.path.join(path, "decisionsmith.json")).read())
    assert meta["kind"] == "class" and meta["labels"] is None and meta["calibration"] == {"team": 1.5}
    assert (
        "fields: team (billing/technical/sales), wants_refund (false/true)"
        in open(os.path.join(path, "MODEL_CARD.md")).read()
    )
    loaded = ds.load(path)
    assert not loaded.simple and loaded.schema.model is not Ticket and loaded.schema.name == "Ticket"
    assert loaded.schema.questions() == compile_schema(Ticket).questions()
    assert loaded.calibration == {"team": {"temperature": 1.5, "threshold": 0.7}}
    assert loaded.report is None
    mine = ds.load(path, Ticket)
    assert isinstance(mine.predict("you charged me twice"), Ticket)
    with pytest.raises(ValueError, match="does not match"):
        ds.load(path, Rich)


def test_load_errors(tiny, tmp_path):
    with pytest.raises(ValueError, match="plain Laya checkpoint"):
        ds.load(tiny)
    with pytest.raises(FileNotFoundError, match="no saved model"):
        ds.load(tmp_path / "nope")
    (tmp_path / "odd").mkdir()
    (tmp_path / "odd" / "decisionsmith.json").write_text(json.dumps({"format": "other/9"}))
    with pytest.raises(ValueError, match="this decisionsmith reads"):
        ds.load(tmp_path / "odd")


def test_harness_uses_saved_thresholds(tiny, tmp_path, monkeypatch):
    from decisionsmith.training import adapt as adapting

    m = ds.model(LABELS, str(tiny))
    m.calibration = {"label": {"threshold": 0.4}}
    teacher = FakeEngine(lambda t: {"label": truth(t)["team"]}, name="teacher")
    h = ds.harness(m, teacher=teacher, log=None)
    assert h._threshold("label") == 0.4
    h = ds.harness(m, teacher=teacher, log=str(tmp_path / "d.db"))
    assert h._threshold("label") == 0.4
    fitted = {"label": {"temperature": 1.2, "threshold": 0.9}}
    monkeypatch.setattr(adapting, "fit", lambda *a: (fitted, []))
    h.adapt()
    assert m.calibration == fitted and h._threshold("label") == 0.9
    assert ds.harness(m, teacher=teacher, log=str(tmp_path / "d.db"))._threshold("label") == 0.9
    h.close()


def test_harness_finetune_forgets_the_old_calibration(tiny, tmp_path, monkeypatch):
    import decisionsmith.training.finetuning as fmod

    m = ds.model(LABELS, str(tiny))
    m.calibration = {"label": {"threshold": 0.4}}
    teacher = FakeEngine(lambda t: {"label": truth(t)["team"]}, name="teacher")
    with ds.harness(m, teacher=teacher, log=str(tmp_path / "d.db"), mode="shadow") as h:
        h.many(corpus(60))
        monkeypatch.setattr(fmod, "finetune", lambda *a, **k: ds.Report("finetune", "x", [], go=True))
        h.finetune(out=str(tiny))
        assert h._model is None and h._threshold("label") == 0.8
