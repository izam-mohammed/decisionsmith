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
    assert artifact.version_path(None, "Ticket Model") == (os.path.join("models", "ticket-model-v1"), True)
    assert artifact.version_path(None, "!!")[0] == os.path.join("models", "model-v1")
    os.makedirs("out/t-v1")
    os.makedirs("out/t-v5")
    os.makedirs("out/t-vx")
    assert artifact.version_path("out/t", "x") == (os.path.join("out", "t-v6"), True)
    assert artifact.version_path("out/t-v7", "x") == (os.path.join("out", "t-v7"), False)
    with pytest.raises(FileExistsError, match="never overwritten"):
        artifact.version_path("out/t-v1", "x")
    home = os.path.expanduser("~")
    assert artifact.version_path("~/nowhere-decisionsmith/t", "x")[0].startswith(home)


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
    assert first == os.path.join("models", "billing-technical-sales-v1")
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
    assert "license:" not in card and 'ds.load("models/team-v1")' in card
    saved_report = json.loads(open(os.path.join(path, "report.json")).read())
    assert "worst" not in saved_report["details"] and saved_report["path"] is None
    assert str(tiny) not in json.dumps(saved_report) and "you charged" not in json.dumps(saved_report)

    loaded = ds.load(path)
    assert loaded.simple and loaded.path == path and loaded.meta["name"] == "team-v1"
    assert loaded.schema.questions() == m.schema.questions()
    assert loaded.calibration == {"label": {"threshold": m.calibration["label"]["threshold"]}}
    assert loaded.report is not None and loaded.report.kind == "evaluate" and loaded.report.go is False
    assert loaded.predict("you charged me twice") in LABELS
    assert ds.load(path, LABELS).simple
    with pytest.raises(ValueError, match="does not ask the same questions"):
        ds.load(path, ["spam", "ham"])
    again = loaded.save()
    assert again == os.path.join("models", "team-v3")


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
    with pytest.raises(ValueError, match="does not ask the same questions"):
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


def test_train_after_evaluate_forgets_the_old_report_and_thresholds(tiny, tmp_path, monkeypatch):
    import decisionsmith.training.finetuning as fmod

    monkeypatch.chdir(tmp_path)

    def fake(rows, model, base, out, **kw):
        import shutil

        shutil.copytree(tiny, out, dirs_exist_ok=True)
        return ds.Report(
            "finetune",
            "x",
            [],
            details={
                "base": {"all": {"accuracy": 0.1}},
                "finetuned": {"all": {"accuracy": 0.9, "n": 200}},
                "provenance": {"rows": {"train": 10}},
            },
        )

    monkeypatch.setattr(fmod, "finetune", fake)
    m = ds.model(LABELS, str(tiny))
    m.evaluate(pairs(30))
    assert m.report is not None and "label" in m.calibration
    m.train(pairs(30), out="runs/one", verbose=False)
    assert m.report is None and m.calibration == {}
    v1 = m.save("models/team", verbose=False)
    meta = json.loads(open(os.path.join(v1, "decisionsmith.json")).read())
    assert meta["thresholds"] == {} and not os.path.exists(os.path.join(v1, "report.json"))
    loaded = ds.load(v1)
    loaded.evaluate(pairs(30))
    loaded.train(pairs(30), out="runs/two", verbose=False)
    v2 = loaded.save(verbose=False)
    assert v2 == os.path.join("models", "team-v2") and loaded.report is None
    assert json.loads(open(os.path.join(v2, "decisionsmith.json")).read())["thresholds"] == {}


class Described(BaseModel):
    """A support ticket."""

    team: Annotated[Literal["billing", "technical", "sales"], ds.Options(billing="money things", sales="pricing")] = (
        Field(description="Which team should handle this?")
    )
    wants_refund: bool = Field(description="Does the customer ask for a refund?")


class NoDoc(BaseModel):
    team: Annotated[
        Literal["billing", "technical", "sales"], ds.Options(billing="payments and refunds", sales="pricing")
    ] = Field(description="Which team should handle this?")
    wants_refund: bool = Field(description="Does the customer ask for a refund?")


def test_load_with_a_class_checks_questions_not_only_labels(tiny, tmp_path):
    m = ds.model(Ticket, str(tiny))
    path = m.save(str(tmp_path / "ticket"), verbose=False)
    with pytest.raises(ValueError, match=r"team\.criteria: saved .*payments and refunds.*money things"):
        ds.load(path, Described)
    with pytest.raises(ValueError, match=r"instructions: saved 'A support ticket\."):
        ds.load(path, NoDoc)
    with pytest.raises(ValueError, match="wants_refund: only in the saved model"):
        ds.load(path, ["billing", "technical", "sales"])


class Levels(BaseModel):
    stars: Literal[1, 2, 3]
    colour: Colour


def test_int_labels_round_trip_and_enums_come_back_as_strings(tiny, tmp_path):
    path = ds.model(Levels, str(tiny)).save(str(tmp_path / "levels"), verbose=False)
    loaded = ds.load(path)
    assert loaded.predict("a b").stars in (1, 2, 3)
    assert loaded.predict("a b").colour in ("red", "blue")
    assert isinstance(ds.load(path, Levels).predict("a b").colour, Colour)


def test_save_plain_checkpoint_dir_and_saved_folder_through_ds_model(tiny, tmp_path, capsys):
    path = ds.model(LABELS, str(tiny)).save(str(tmp_path / "plain"))
    assert capsys.readouterr().out.strip() == "saved to %s" % path
    m = ds.model(LABELS, path)
    assert m.path == path and m.meta["name"] == "plain-v1"
    with pytest.raises(ValueError, match="does not ask the same questions"):
        ds.model(["a", "b"], path)


def test_save_moves_on_when_the_version_appears_meanwhile(tiny, tmp_path, monkeypatch):
    real = artifact.shutil.copytree

    def racing(src, dst, *args, **kw):
        real(src, dst, *args, **kw)
        os.makedirs(str(tmp_path / "team-v1"), exist_ok=True)

    monkeypatch.setattr(artifact.shutil, "copytree", racing)
    m = ds.model(LABELS, str(tiny))
    assert m.save(str(tmp_path / "team"), verbose=False) == str(tmp_path / "team-v2")
    os.makedirs(str(tmp_path / "exact"), exist_ok=True)

    def racing_exact(src, dst, *args, **kw):
        real(src, dst, *args, **kw)
        os.makedirs(str(tmp_path / "exact-v4"), exist_ok=True)

    monkeypatch.setattr(artifact.shutil, "copytree", racing_exact)
    with pytest.raises(FileExistsError):
        m.save(str(tmp_path / "exact-v4"), verbose=False)
    assert not [p for p in os.listdir(tmp_path) if ".tmp-" in p]


def test_meta_validation(tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "decisionsmith.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        ds.load(folder)
    (folder / "decisionsmith.json").write_text(json.dumps({"format": "decisionsmith.model/1", "name": "x"}))
    with pytest.raises(ValueError, match="is missing kind"):
        ds.load(folder)
    (folder / "decisionsmith.json").write_text("[1]")
    with pytest.raises(ValueError, match="has format None"):
        ds.load(folder)


def test_threshold_precedence(tiny, tmp_path):
    m = ds.model(LABELS, str(tiny))
    m.calibration = {"label": {"threshold": 0.4, "temperature": 1.3}}
    teacher = FakeEngine(lambda t: {"label": truth(t)["team"]}, name="teacher")
    assert ds.harness(m, teacher=teacher, log=None)._threshold("label") == 0.4
    h = ds.harness(m, teacher=teacher, log=None, threshold=0.9)
    assert h._threshold("label") == 0.9 and h._calib["label"] == {"temperature": 1.3}
    assert ds.harness(ds.model(LABELS, str(tiny)), teacher=teacher, log=None)._threshold("label") == 0.8


def test_cascade_uses_the_loaded_threshold_end_to_end(tmp_path):
    student = FakeEngine(lambda t: {"label": truth(t)["team"]}, confidence=0.6, name="student")
    teacher = FakeEngine(lambda t: {"label": truth(t)["team"]}, name="teacher")
    m = ds.model(LABELS, student)
    m.calibration = {"label": {"threshold": 0.5}}
    with ds.harness(m, teacher=teacher, log=None, audit=0) as h:
        assert h.decide("you charged me twice").source == {"label": "student"}
    m.calibration = {"label": {"threshold": 0.7}}
    with ds.harness(m, teacher=teacher, log=None, audit=0) as h:
        assert h.decide("you charged me twice").source == {"label": "teacher"}


def test_metrics_by_hand():
    from decisionsmith.report import metrics

    pred = [[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7]]
    gold = [[1, 0], [1, 0], [1, 0], [0, 1]]
    m = metrics(pred, gold)
    assert m["accuracy"] == 0.75
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert m["ece"] == pytest.approx((0.1 + 0.2 + 0.6 + 0.3) / 4)
    assert m["brier"] == pytest.approx((0.02 + 0.08 + 0.72 + 0.18) / 4)


def test_threshold_is_chosen_on_one_half_and_reported_on_the_other():
    from decisionsmith.training.data import text_hash

    texts = corpus(80)
    right = FakeEngine(lambda t: {"label": truth(t)["team"]}, confidence=0.95)
    report = ds.model(LABELS, right).evaluate([(t, truth(t)["team"]) for t in texts])
    f = report.details["fields"]["label"]
    even = sum(int(text_hash(t)[:8], 16) % 2 == 0 for t in texts)
    assert f["threshold_rows"] == even and f["coverage_rows"] == 80 - even
    assert f["threshold"] == 0.95 and f["coverage"] == 1.0 and f["accuracy_when_sure"] == 1.0
    one = ds.model(LABELS, right).evaluate([(texts[0], truth(texts[0])["team"])])
    assert one.details["fields"]["label"]["threshold_rows"] == 1 == one.details["fields"]["label"]["coverage_rows"]
