import json
from typing import Annotated, Literal

import pytest

import decisionsmith as ds
from decisionsmith.engines import EngineError, LayaEngine
from decisionsmith.report import Report
from decisionsmith.testing import FakeEngine
from decisionsmith.training.export import gold_entry as _gold_entry
from tests.conftest import Ticket, corpus, truth

TEXT = "you charged me twice, refund please"


def student(conf=0.9, accuracy=1.0, **kw):
    return FakeEngine(truth, confidence=conf, accuracy=accuracy, name="student", **kw)


def test_needs_an_engine_and_valid_settings(teacher, db):
    with pytest.raises(ValueError, match="teacher, a student"):
        ds.harness(Ticket, log=db)
    with pytest.raises(ValueError, match="threshold"):
        ds.harness(Ticket, teacher=teacher, threshold=0, log=db)
    with pytest.raises(ValueError, match="audit"):
        ds.harness(Ticket, teacher=teacher, audit=2, log=db)
    with pytest.raises(ValueError, match="needs a student"):
        ds.harness(Ticket, teacher=teacher, mode="shadow", log=db)
    with pytest.raises(ValueError, match="needs a teacher"):
        ds.harness(Ticket, student=student(), mode="cascade", log=db)
    with pytest.raises(ValueError, match="use one of"):
        ds.harness(Ticket, teacher=teacher, mode="yolo", log=db)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown fields"):
        ds.harness(Ticket, teacher=teacher, mode={"nope": "teacher"}, log=db)


def test_default_modes(teacher, db):
    assert set(ds.harness(Ticket, teacher=teacher, student=student(), log=db).modes.values()) == {"cascade"}
    assert set(ds.harness(Ticket, teacher=teacher, log=db).modes.values()) == {"teacher"}
    assert set(ds.harness(Ticket, student=student(), log=db).modes.values()) == {"student"}
    h = ds.harness(Ticket, teacher=teacher, student=student(), mode={"team": "shadow"}, log=db)
    assert h.modes == {"team": "shadow", "wants_refund": "cascade"}


def test_teacher_mode_logs(teacher, db):
    h = ds.harness(Ticket, teacher=teacher, log=db)
    t = h(TEXT)
    assert t == Ticket(team="billing", wants_refund=True)
    row = h.log.rows("Ticket")[0]
    assert row["source"] == {"team": "teacher", "wants_refund": "teacher"}
    assert row["student_dists"] is None and row["teacher_dists"]["team"]["billing"] == 1.0
    assert row["value"] == {"team": "billing", "wants_refund": "true"} and row["text"] == TEXT


def test_shadow_measures_student(teacher, db):
    s = student(accuracy=0.0)
    h = ds.harness(Ticket, teacher=teacher, student=s, mode="shadow", log=db)
    r = h.decide(TEXT)
    assert r.value == Ticket(team="billing", wants_refund=True) and r.sure
    assert r.source == {"team": "teacher", "wants_refund": "teacher"}
    row = h.log.rows("Ticket")[0]
    assert row["student_dists"]["team"]["billing"] < 0.5 and len(s.calls) == 1


def test_cascade_routes_on_confidence(teacher, db):
    sure = ds.harness(Ticket, teacher=teacher, student=student(conf=0.95), log=db, audit=0)
    r = sure.decide(TEXT)
    assert r.source == {"team": "student", "wants_refund": "student"} and r.sure and teacher.calls == []
    unsure = ds.harness(Ticket, teacher=teacher, student=student(conf=0.6), log=db, audit=0)
    r = unsure.decide(TEXT)
    assert r.source == {"team": "teacher", "wants_refund": "teacher"}
    assert teacher.calls[-1][1] == ["team", "wants_refund"]
    mixed = FakeEngine(truth, confidence=lambda t, q: 0.95 if q == "team" else 0.5, name="student")
    r = ds.harness(Ticket, teacher=teacher, student=mixed, log=db, audit=0).decide(TEXT)
    assert r.source == {"team": "student", "wants_refund": "teacher"} and teacher.calls[-1][1] == ["wants_refund"]


def test_cascade_audit_asks_teacher_on_sure_answers(teacher, db):
    h = ds.harness(Ticket, teacher=teacher, student=student(conf=0.95), log=db, audit=1.0)
    r = h.decide(TEXT)
    assert r.source["team"] == "student" and teacher.calls
    assert h.log.rows("Ticket")[0]["teacher_dists"] is not None


def test_student_mode_and_fallbacks(teacher, db):
    r = ds.harness(Ticket, student=student(conf=0.6), log=db).decide(TEXT)
    assert r.source["team"] == "student" and not r.sure
    down = student(error="down")
    r = ds.harness(Ticket, teacher=teacher, student=down, mode="student", log=db).decide(TEXT)
    assert r.source == {"team": "teacher", "wants_refund": "teacher"}
    with pytest.raises(EngineError, match="no engine could answer"):
        ds.harness(Ticket, student=down, log=db).decide(TEXT)


def test_teacher_failures(db):
    bad_teacher = FakeEngine(error=RuntimeError("boom"), name="teacher")
    r = ds.harness(Ticket, teacher=bad_teacher, student=student(), mode="shadow", log=db).decide(TEXT)
    assert r.source["team"] == "student" and not r.sure
    r = ds.harness(Ticket, teacher=bad_teacher, student=student(conf=0.5), log=db).decide(TEXT)
    assert r.source["team"] == "student" and not r.sure
    with pytest.raises(EngineError):
        ds.harness(Ticket, teacher=bad_teacher, log=db).decide(TEXT)


def test_bad_student_answers_count_as_failure(teacher, db):
    class Broken:
        name = "broken"

        def ask(self, text, questions):
            return {"answers": {"team": {"choice": "nope"}, "wants_refund": {"noul": 0.5}}}

    r = ds.harness(Ticket, teacher=teacher, student=Broken(), log=db).decide(TEXT)
    assert r.source == {"team": "teacher", "wants_refund": "teacher"}


def test_many_keeps_order_and_batches(teacher, db):
    s = student(conf=0.95)
    h = ds.harness(Ticket, teacher=teacher, student=s, mode="shadow", log=db)
    texts = corpus(9)
    assert h.many(texts) == [Ticket(**truth(t)) for t in texts]
    assert len(h.log.rows("Ticket")) == 9 and h.many([]) == []

    class Batch(FakeEngine):
        def ask_many(self, texts, questions):
            raise EngineError("batch", "down")

    h = ds.harness(Ticket, teacher=teacher, student=Batch(truth, name="student"), log=db)
    assert h.many(texts[:3]) == [Ticket(**truth(t)) for t in texts[:3]]
    assert ds.harness(Ticket, teacher=teacher, log=db).many(texts[:2])[0] == Ticket(**truth(texts[0]))


def test_bad_text(teacher, db):
    h = ds.harness(Ticket, teacher=teacher, log=db)
    for bad in ("", "  ", None, 3):
        with pytest.raises(ValueError, match="non-empty"):
            h(bad)  # type: ignore[arg-type]


def test_label_and_no_log(teacher, db):
    h = ds.harness(Ticket, teacher=teacher, log=db)
    r = h.decide(TEXT)
    h.label(r.id, team="sales", wants_refund="no")
    assert h.log.rows("Ticket")[0]["labels"] == {"team": "sales", "wants_refund": "false"}
    with pytest.raises(ValueError, match="at least one"):
        h.label(r.id)
    with pytest.raises(ValueError, match="not an option"):
        h.label(r.id, team="legal")
    with pytest.raises(KeyError):
        h.label("missing", team="sales")
    off = ds.harness(Ticket, teacher=teacher, log=None)
    assert off.decide(TEXT).id is None
    for call in (lambda: off.label("x", team="sales"), off.status, off.adapt, lambda: off.export("x.jsonl")):
        with pytest.raises(ValueError, match="log=None"):
            call()


def test_export(teacher, db, tmp_path):
    h = ds.harness(Ticket, teacher=teacher, student=student(), mode="shadow", log=db)
    ids = [h.decide(t).id for t in corpus(3)]
    h.label(ids[0], team="sales")
    n = h.export(str(tmp_path / "a.jsonl"))
    rows = [json.loads(x) for x in open(tmp_path / "a.jsonl")]
    assert n == 3 and rows[0]["answers"]["team"] == {"billing": 0.0, "technical": 0.0, "sales": 1.0}
    assert rows[1]["answers"]["wants_refund"]["false"] == 1.0
    assert h.export(str(tmp_path / "t.jsonl"), format="typed-decisions") == 3
    t = json.loads(open(tmp_path / "t.jsonl").readline())
    assert json.loads(t["gold"])["team"] == {
        "probabilities": {"billing": 0.0, "technical": 0.0, "sales": 1.0},
        "label": "sales",
    }
    assert json.loads(t["questions"])["team"]["type"] == "choice"
    with pytest.raises(ValueError, match="format"):
        h.export(str(tmp_path / "x.jsonl"), format="csv")  # type: ignore[arg-type]
    ds.harness(Ticket, teacher=FakeEngine(name="t2"), log=db)
    from decisionsmith.schema import compile_schema

    class Scaled(Ticket):
        level: Annotated[Literal["low", "high"], ds.Scale]

    s = compile_schema(Scaled)
    assert _gold_entry(s, "level", {"low": 0.3, "high": 0.7}) == {"probabilities": {"0": 0.3, "1": 0.7}, "label": "1"}


def test_adapt_calibrates_and_changes_routing(teacher, db):
    s = student(conf=0.99, accuracy=0.75)
    h = ds.harness(Ticket, teacher=teacher, student=s, mode="shadow", log=db)
    h.many(corpus(300))
    before = h.status().fields["team"].sure_rate
    rep = h.adapt(target=0.97)
    row = next(r for r in rep.rows if r["field"] == "team")
    assert row["rows"] == 300 and row["temperature"] > 1.0 and row["ece_after"] < row["ece_before"]
    assert row["threshold"] is None and "no threshold" in rep.reasons[0] and "/" in row["confused"]
    assert h._threshold("team") == 1.01 and before == 1.0
    cascade = ds.harness(Ticket, teacher=teacher, student=s, log=db, audit=0)
    assert cascade.decide(corpus(1)[0]).source["team"] == "teacher"
    assert "team" in cascade._calib and str(rep).startswith("adapt: Ticket")


def test_adapt_keeps_answers_and_finds_threshold(teacher, db):
    s = FakeEngine(truth, confidence=lambda t, q: 0.9 if int(t.split()[-1]) % 4 else 0.6, accuracy=0.9, name="student")
    h = ds.harness(Ticket, teacher=teacher, student=s, mode="shadow", log=db)
    h.many(corpus(400))
    rep = h.adapt(target=0.8)
    team = next(r for r in rep.rows if r["field"] == "team")
    assert team["threshold"] is not None and 0 < team["coverage"] <= 1
    r = h.decide(corpus(1)[0])
    assert r.value.team == "billing"


def test_adapt_needs_rows_and_student(teacher, db):
    h = ds.harness(Ticket, teacher=teacher, student=student(), mode="shadow", log=db)
    h.many(corpus(10))
    rep = h.adapt()
    assert all("needs" in r["note"] for r in rep.rows)
    h._calib = {"team": {"temperature": 2.0, "threshold": 0.5}}
    assert h.adapt().rows and h._calib["team"]["temperature"] == 2.0
    with pytest.raises(ValueError, match="calibrates the student"):
        ds.harness(Ticket, teacher=teacher, log=db).adapt()


def test_calibrated_identity():
    h = ds.harness(Ticket, teacher=FakeEngine(), log=None)
    d = {"a": 0.7, "b": 0.3}
    assert h._calibrated("team", d) is d
    h._calib = {"team": {"temperature": 1.0}}
    assert h._calibrated("team", d) is d


def _fake_finetune(go):
    calls = {}

    def fake(rows, model, base, out, **kw):
        calls.update(rows=rows, base=base, out=out, kw=kw)
        __import__("os").makedirs(out, exist_ok=True)
        return Report("finetune", "x", [], go=go, path=out, details={"train_ids": [rows[0]["id"]]})

    return fake, calls


def test_finetune_swaps_only_when_better(teacher, db, monkeypatch, tiny):
    import decisionsmith.training.finetuning as fmod

    h = ds.harness(Ticket, teacher=teacher, log=db)
    with pytest.raises(ValueError, match="at least 50"):
        h.finetune()
    h.many(corpus(60))
    fake, calls = _fake_finetune(go=True)
    monkeypatch.setattr(fmod, "finetune", fake)
    (db_dir := __import__("pathlib").Path(db).parent / "runs" / "ticket-v1").mkdir(parents=True)
    tiny_out = str(tiny)
    rep = h.finetune(epochs=1)
    assert calls["base"] == "laya" and calls["out"].endswith("ticket-v2") and calls["kw"] == {"epochs": 1}
    assert isinstance(h.student, LayaEngine) and "teacher mode" in rep.reasons[-1]
    assert h.log.trained() == {calls["rows"][0]["id"]} and db_dir.exists()
    h2 = ds.harness(Ticket, teacher=teacher, student="laya:%s" % tiny_out, log=db)
    fake2, calls2 = _fake_finetune(go=False)
    monkeypatch.setattr(fmod, "finetune", fake2)
    rep = h2.finetune(out=str(tiny))
    assert calls2["base"] == tiny_out and rep.reasons[-1] == "kept the current student"
    assert h2.student.name == "laya:%s" % tiny_out
    fake3, _ = _fake_finetune(go=True)
    monkeypatch.setattr(fmod, "finetune", fake3)
    rep = h2.finetune(out=tiny_out)
    assert "now uses" in rep.reasons[-1]


def test_finetune_refuses_non_laya_student(teacher, db):
    with pytest.raises(EngineError, match="only Laya"):
        ds.harness(Ticket, teacher=teacher, student=student(), log=db).finetune()


def test_next_run_without_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = ds.harness(Ticket, teacher=FakeEngine(), log=None)
    assert h._next_run() == str(tmp_path / "runs" / "ticket-v1")


def test_context_manager(teacher, db):
    with ds.harness(Ticket, teacher=teacher, student=student(), mode="shadow", log=db) as h:
        h(TEXT)
        assert h._pool is not None
    assert h._pool is None
    h.close()


def test_result_is_serialisable(teacher, db):
    r = ds.harness(Ticket, teacher=teacher, log=db).decide(TEXT)
    data = r.model_dump(mode="json")
    assert data["value"] == {"team": "billing", "wants_refund": True} and data["latency_ms"] >= 0


def test_adapt_skips_fields_without_student_answers_and_reports_no_confusion(teacher, db):
    s = FakeEngine(truth, confidence=0.9, name="student")
    h = ds.harness(Ticket, teacher=teacher, student=s, mode={"team": "shadow", "wants_refund": "teacher"}, log=db)
    h.many(corpus(120))
    rows = {r["field"]: r for r in h.adapt().rows}
    assert rows["team"]["rows"] == 120 and rows["team"]["confused"] is None
    assert rows["wants_refund"]["rows"] == 0 and "needs" in rows["wants_refund"]["note"]


def test_close_without_log(teacher):
    h = ds.harness(Ticket, teacher=teacher, log=None)
    h("you charged me twice")
    h.close()
    assert h._pool is None
