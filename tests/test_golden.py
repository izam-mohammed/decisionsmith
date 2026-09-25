import csv
import json
import os

import pytest

import decisionsmith as ds
from decisionsmith import cli
from decisionsmith.log import Log
from decisionsmith.testing import FakeEngine
from decisionsmith.training import data as data_mod
from tests.conftest import Ticket, corpus, truth

LABELS = ["billing", "technical", "sales"]


def team(text):
    return {"label": truth(text)["team"]}


def unsure_on_odd(text, qid):
    return 0.4 if int(text.split()[-1]) % 2 else 0.95


def shadow_log(db, n=30, collect=1.0):
    student = FakeEngine(truth, confidence=unsure_on_odd, name="student")
    with ds.harness(
        Ticket, teacher=FakeEngine(truth, name="teacher"), student=student, mode="shadow", log=db, collect=collect
    ) as h:
        h.many(corpus(n))


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# collect= and forget


def test_collect_keeps_text_only_for_sampled_unsure_or_teacher_rows(db):
    student = FakeEngine(truth, confidence=unsure_on_odd, name="student")
    teacher = FakeEngine(truth, name="teacher")
    with ds.harness(Ticket, teacher=teacher, student=student, mode="cascade", log=db, collect=0.0001, audit=0) as h:
        h._random.seed(1)
        h.many(corpus(40))
        rows = h.log.rows("Ticket")
    kept = [r for r in rows if r["text"]]
    assert len(rows) == 40 and kept and len(kept) < 40
    assert all("teacher" in r["source"].values() for r in kept)
    assert all(r["text"] is None for r in rows if "teacher" not in r["source"].values())
    assert all(r["value"] for r in rows)


def test_collect_all_and_none(db, tmp_path):
    shadow_log(db, 6)
    with Log(db) as log:
        assert all(r["text"] for r in log.rows("Ticket"))
    other = str(tmp_path / "none.db")
    shadow_log(other, 6, collect=0)
    with Log(other) as log:
        rows = log.rows("Ticket")
    assert rows and all(r["text"] is None for r in rows)
    h = ds.harness(Ticket, teacher=FakeEngine(truth), log=other)
    assert h.export(str(tmp_path / "x.jsonl")) == 0
    h.close()
    with pytest.raises(ValueError, match="collect must be in"):
        ds.harness(Ticket, teacher="fake", log=None, collect=2)


def test_forget_one_and_old(db, monkeypatch):
    import decisionsmith.core as core

    with ds.harness(Ticket, teacher=FakeEngine(truth), log=db) as h:
        r = h.decide("you charged me twice")
        h.label(r.id, team="billing")
        h.decide("the app crashes on login")
        assert h.forget(r.id) == 1
        assert [x["text"] for x in h.log.rows("Ticket")] == ["the app crashes on login"]
        with pytest.raises(KeyError, match="no decision"):
            h.forget(r.id)
        assert h.forget(older_than_days=1) == 0
        monkeypatch.setattr(core.time, "time", lambda: 10**12)
        assert h.forget(older_than_days=1) == 1 and h.log.rows("Ticket") == []
        for bad in ({}, {"decision_id": "x", "older_than_days": 1}):
            with pytest.raises(ValueError, match="give a decision id"):
                h.forget(**bad)
    with pytest.raises(ValueError, match="log=None"):
        ds.harness(Ticket, teacher="fake", log=None).forget("x")


# ds.golden


def test_golden_uncertain_from_a_log(db, tmp_path, capsys):
    shadow_log(db, 30)
    out = str(tmp_path / "golden.csv")
    rows = ds.golden(db, FakeEngine(truth, name="big-llm"), n=10, schema=Ticket, out=out)
    assert len(rows) == 10 and all(int(r["text"].split()[-1]) % 2 == 1 for r in rows)
    assert sum(r["split"] == "test" for r in rows) == 2 and {r["labelled_by"] for r in rows} == {"big-llm"}
    csv_rows = read(out)
    assert list(csv_rows[0]) == ["id", "text", "team", "wants_refund", "split", "labelled_by"]
    assert csv_rows[0]["team"] == truth(csv_rows[0]["text"])["team"]
    assert "golden: 10 rows (uncertain) labelled by big-llm · 2 marked split=test · wrote" in capsys.readouterr().out


def test_golden_from_a_harness_and_disagree(db):
    student = FakeEngine(truth, accuracy=0.5, name="student")
    with ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), student=student, mode="shadow", log=db) as h:
        h.many(corpus(30))
        rows = h.log.rows("Ticket")
        wrong = {
            r["text"]
            for r in rows
            if r["student_dists"]["team"] != r["teacher_dists"]["team"]
            or r["student_dists"]["wants_refund"] != r["teacher_dists"]["wants_refund"]
        }
        got = ds.golden(h, FakeEngine(truth), n=len(wrong), strategy="disagree", out=None, verbose=False)
    assert wrong and {r["text"] for r in got} == wrong


def test_golden_diverse_random_and_texts(tmp_path, capsys):
    texts = [*corpus(30), "you charged me twice please 0"]
    model = ds.model(LABELS, FakeEngine(team, confidence=0.7))
    diverse = ds.golden(texts, FakeEngine(team), n=9, strategy="diverse", schema=model, out=None, verbose=False)
    assert sorted(
        r["answers"]["label"] and max(r["answers"]["label"], key=r["answers"]["label"].get) for r in diverse
    ) == sorted(LABELS * 3)
    assert (
        len(ds.golden(texts, FakeEngine(team), n=100, strategy="diverse", schema=model, out=None, verbose=False)) == 30
    )
    plain = ds.golden(texts, FakeEngine(team), n=5, strategy="diverse", schema=LABELS, out=None, verbose=False)
    assert len(plain) == 5
    rnd = ds.golden(
        [{"text": t} for t in texts],
        FakeEngine(team),
        n=5,
        strategy="random",
        schema=LABELS,
        test=0,
        out=None,
        verbose=False,
    )
    assert len(rnd) == 5 and all(r["split"] == "train" for r in rnd)
    path = tmp_path / "texts.txt"
    path.write_text("\n".join(texts))
    assert (
        len(ds.golden(str(path), FakeEngine(team), n=4, strategy="random", schema=LABELS, out=None, verbose=False)) == 4
    )
    ds.golden(texts, FakeEngine(team, error="down"), n=3, strategy="random", schema=LABELS, out=None)
    assert "golden: 0 rows (random) labelled by fake · 3 could not be labelled · 0 marked split=test" in (
        capsys.readouterr().out
    )


def test_golden_diverse_uneven_buckets_and_disagree_without_teacher():
    texts = ["the app crashes on login %d" % i for i in range(10)] + ["you charged me twice 0"]
    model = ds.model(LABELS, FakeEngine(team, confidence=0.7))
    got = ds.golden(texts, FakeEngine(team), n=6, strategy="diverse", schema=model, out=None, verbose=False)
    assert len(got) == 6 and texts[-1] in {r["text"] for r in got}
    unsure = ds.model(LABELS, FakeEngine(team, confidence=lambda t, q: 0.4 if t.endswith("3") else 0.9))
    got = ds.golden(texts, FakeEngine(team), n=1, strategy="disagree", schema=unsure, out=None, verbose=False)
    assert [r["text"] for r in got] == ["the app crashes on login 3"]


def test_golden_errors(db, tmp_path):
    teacher = FakeEngine(truth)
    cases = [
        (dict(source=corpus(3), teacher=teacher, strategy="nope", schema=Ticket), ValueError, "strategy must be"),
        (dict(source=corpus(3), teacher=teacher, n=0, schema=Ticket), ValueError, "at least 1"),
        (dict(source=corpus(3), teacher=teacher, test=1, schema=Ticket), ValueError, "held out"),
        (dict(source=corpus(3), teacher=teacher), ValueError, "needs to know the answers"),
        (dict(source=corpus(3), teacher=teacher, schema=Ticket), ValueError, "strategy='random'"),
        (dict(source=[" "], teacher=teacher, schema=Ticket), ValueError, "no texts"),
        (dict(source=str(tmp_path / "x.db"), teacher=teacher, schema=Ticket), FileNotFoundError, "no log"),
        (dict(source=ds.harness(Ticket, teacher="fake", log=None), teacher=teacher), ValueError, "log=None"),
    ]
    for kw, err, match in cases:
        with pytest.raises(err, match=match):
            ds.golden(out=None, **kw)
    shadow_log(db, 4, collect=0)
    with pytest.raises(ValueError, match="no Ticket decisions with text"):
        ds.golden(db, teacher, schema=Ticket, out=None)


def test_golden_split_is_kept_out_of_training_and_used_for_evaluation(tmp_path):
    out = str(tmp_path / "golden.csv")
    model = ds.model(LABELS, FakeEngine(team, confidence=0.6))
    ds.golden(corpus(50), FakeEngine(team), n=50, schema=model, out=out, verbose=False)
    train = data_mod.load(out, model.schema)
    test = data_mod.load(out, model.schema, split="test")
    assert len(train) == 40 and len(test) == 10 and not {r.id for r in train} & {r.id for r in test}
    report = ds.model(LABELS, FakeEngine(team, confidence=1.0)).evaluate(out)
    assert report.details["rows"] == 10
    both = [{"text": "a", "label": "sales", "split": "test"}, {"text": "b", "label": "sales"}]
    assert [r.text for r in data_mod.load(model._rows(both), model.schema)] == ["b"]
    assert [r.text for r in data_mod.load(model._rows(both), model.schema, split="test")] == ["a", "b"]


# CLI


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_cli_golden(db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_teacher", lambda args: FakeEngine(truth) if args.teacher else None)
    shadow_log(db, 20)
    out = str(tmp_path / "g.csv")
    code, text = run(
        capsys,
        "golden",
        "--log",
        db,
        "--schema",
        "tests/conftest.py:Ticket",
        "--teacher",
        "x",
        "-n",
        "5",
        "--out",
        out,
        "--json",
    )
    assert code == 0 and json.loads(text)["rows"] == 5 and len(read(out)) == 5
    texts = tmp_path / "t.txt"
    texts.write_text("\n".join(corpus(12)))
    monkeypatch.setattr(cli, "_teacher", lambda args: FakeEngine(team) if args.teacher else None)
    monkeypatch.setattr(cli, "_model", lambda args: ds.model(LABELS, FakeEngine(team, confidence=0.6)))
    code, text = run(
        capsys,
        "golden",
        str(texts),
        "--labels",
        "billing,technical,sales",
        "--teacher",
        "x",
        "--score",
        "--strategy",
        "diverse",
        "-n",
        "6",
        "--out",
        out,
    )
    assert code == 0 and "golden: 6 rows (diverse)" in text
    code, _ = run(
        capsys,
        "golden",
        str(texts),
        "--labels",
        "billing,technical,sales",
        "--teacher",
        "x",
        "--strategy",
        "random",
        "-n",
        "3",
        "--out",
        out,
    )
    assert code == 0
    for argv, match in (
        (["golden", "--labels", "a,b", "--teacher", "x"], "not both"),
        (["golden", str(texts), "--labels", "a,b"], "needs --teacher"),
        (["golden", str(texts), "--teacher", "x"], "either --labels"),
    ):
        code, text = run(capsys, *argv, "--json")
        assert code == cli.INVALID and match in json.loads(text)["error"]["message"]


def test_cli_golden_no_rows(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_teacher", lambda args: FakeEngine(team, error="down"))
    texts = tmp_path / "t.txt"
    texts.write_text("\n".join(corpus(3)))
    code, _ = run(
        capsys,
        "golden",
        str(texts),
        "--labels",
        "billing,technical,sales",
        "--teacher",
        "x",
        "--strategy",
        "random",
        "--out",
        str(tmp_path / "g.csv"),
        "--json",
    )
    assert code == cli.NOT_READY


def test_cli_eval(tiny, tmp_path, capsys):
    m = ds.model(LABELS, str(tiny))
    m.trained = str(tiny)
    path = m.save(str(tmp_path / "m"))
    data = tmp_path / "test.csv"
    with open(data, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows([t, truth(t)["team"]] for t in corpus(12))
    report = str(tmp_path / "r.json")
    code, text = run(capsys, "eval", path, str(data), "--out", report, "--device", "cpu")
    assert code == cli.NOT_READY and "evaluate:" in text and os.path.exists(report)
    code, text = run(capsys, "eval", path, str(data), "--schema", "tests/conftest.py:Ticket", "--json")
    assert code == cli.INVALID and "does not match" in json.loads(text)["error"]["message"]


def test_cli_eval_go(tmp_path, capsys, monkeypatch):
    import decisionsmith.predictor as predictor

    model = ds.model(LABELS, FakeEngine(team, confidence=1.0))
    monkeypatch.setattr(predictor, "load", lambda *a, **k: model)
    data = tmp_path / "test.csv"
    with open(data, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows([t, truth(t)["team"]] for t in corpus(120))
    code, text = run(capsys, "eval", "models/x-v1", str(data), "--json")
    assert code == cli.OK and json.loads(text)["go"] is True
