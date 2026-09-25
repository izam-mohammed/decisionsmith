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


def test_collect_keeps_unsure_disagreeing_and_sampled_texts(db):
    from decisionsmith.core import sampled

    student = FakeEngine(truth, confidence=unsure_on_odd, name="student")
    teacher = FakeEngine(truth, name="teacher")
    with ds.harness(Ticket, teacher=teacher, student=student, mode="cascade", log=db, collect=0.3, audit=0) as h:
        h.many(corpus(60))
        rows = h.log.rows("Ticket")
    kept = {r["id"] for r in rows if r["text"]}
    by_chance = {r["id"] for r in rows if sampled(r["id"], 0.3)}
    unsure_ids = {r["id"] for r in rows if "teacher" in r["source"].values()}
    assert kept == unsure_ids | by_chance and unsure_ids and by_chance - unsure_ids and len(kept) < 60
    assert all(r["value"] for r in rows)


def test_collect_samples_in_shadow_and_teacher_modes(db, tmp_path):
    from decisionsmith.core import sampled

    student = FakeEngine(truth, confidence=0.95, name="student")
    with ds.harness(Ticket, teacher=FakeEngine(truth), student=student, mode="shadow", log=db, collect=0.2) as h:
        h.many(corpus(50))
        rows = h.log.rows("Ticket")
    assert {r["id"] for r in rows if r["text"]} == {r["id"] for r in rows if sampled(r["id"], 0.2)}
    wrong = FakeEngine(truth, accuracy=0.0, confidence=0.95, name="wrong")
    with ds.harness(Ticket, teacher=FakeEngine(truth), student=wrong, mode="shadow", log=db, collect=0.01) as h:
        h.many(corpus(10))
        assert all(r["text"] for r in h.log.rows("Ticket") if r["student"] == "wrong")
    only = str(tmp_path / "teacher.db")
    with ds.harness(Ticket, teacher=FakeEngine(truth), log=only, collect=0.2) as h:
        h.many(corpus(50))
        rows = h.log.rows("Ticket")
    assert 0 < sum(bool(r["text"]) for r in rows) < 50


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
    with pytest.warns(UserWarning, match="no text"):
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
        with pytest.raises(ValueError, match="0 or more"):
            h.forget(older_than_days=-1)
    with pytest.raises(ValueError, match="log=None"):
        ds.harness(Ticket, teacher="fake", log=None).forget("x")


def test_forget_leaves_no_bytes_in_the_database_files(db):
    secret = "my card number is 4111 secret-marker-xyz"
    with ds.harness(Ticket, teacher=FakeEngine(truth), log=db) as h:
        r = h.decide("you charged me twice " + secret)
        h.label(r.id, team="billing")
        for t in corpus(20):
            h.decide(t)
        h.forget(r.id)
        files = [db, db + "-wal"]
        found = sum(open(f, "rb").read().count(b"secret-marker-xyz") for f in files if os.path.exists(f))
    assert found == 0


# ds.golden


def test_golden_uncertain_from_a_log(db, tmp_path, capsys):
    shadow_log(db, 30)
    out = str(tmp_path / "golden.csv")
    rows = ds.golden(db, FakeEngine(truth, name="big-llm"), n=10, schema=Ticket, out=out)
    assert len(rows) == 10 and all(int(r["text"].split()[-1]) % 2 == 1 for r in rows)
    assert sum(r["split"] == "test" for r in rows) == 2 and {r["labelled_by"] for r in rows} == {"llm:big-llm"}
    csv_rows = read(out)
    assert list(csv_rows[0]) == ["id", "text", "team", "wants_refund", "split", "labelled_by"]
    assert csv_rows[0]["team"] == truth(csv_rows[0]["text"])["team"]
    assert all(r["id"].startswith("g") and len(r["id"]) == 13 for r in csv_rows)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        ds.golden(db, FakeEngine(truth), n=3, schema=Ticket, out=out)
    assert len(read(out)) == 10
    ds.golden(db, FakeEngine(truth), n=3, schema=Ticket, out=out, overwrite=True, verbose=False)
    assert len(read(out)) == 3 and not os.path.exists(out + ".tmp")
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
    with pytest.raises(ds.EngineError, match="could not label any of the 3 texts tried; first error: .*down"):
        ds.golden(texts, FakeEngine(team, error="down"), n=3, strategy="random", schema=LABELS, out=None)


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
    assert (
        len(train) == 40
        and len(test) == 10
        and {r.split for r in train} == {"train"}
        and not {r.id for r in train} & {r.id for r in test}
    )
    report = ds.model(LABELS, FakeEngine(team, confidence=1.0)).evaluate(out)
    assert report.details["rows"] == 10
    both = [{"text": "a", "label": "sales", "split": "test"}, {"text": "b", "label": "sales"}]
    assert [r.text for r in data_mod.load(model._rows(both), model.schema)] == ["b"]
    assert [r.text for r in data_mod.load(model._rows(both), model.schema, split="test")] == ["a"]
    plain = [{"text": "a", "label": "sales"}, {"text": "b", "label": "sales"}]
    assert len(data_mod.load(model._rows(plain), model.schema, split="test")) == 2
    assert len(data_mod.load(model._rows(both), model.schema, split="all")) == 2
    with pytest.raises(ValueError, match="split must be 'train'"):
        data_mod.load(model._rows(both), model.schema, split="dev")


def test_blank_split_rows_are_train_only_when_any_row_has_a_split():
    model = ds.model(LABELS, "fake")
    rows = [{"text": "t%d" % i, "label": "sales", "split": ["test"] * 5 + ["train"] * 10 + [""] * 5} for i in range(20)]
    rows = [{**r, "split": r["split"][i]} for i, r in enumerate(rows)]
    train = data_mod.load(model._rows(rows), model.schema)
    test = data_mod.load(model._rows(rows), model.schema, split="test")
    assert len(train) == 15 and len(test) == 5 and not {r.text for r in train} & {r.text for r in test}
    assert {r.split for r in train} == {"train"}


def test_bench_uses_only_the_test_split(tmp_path):
    rows = [{"text": t, "answers": truth(t), "split": s} for t, s in zip(corpus(12), ["test"] * 4 + ["train"] * 8)]
    assert ds.bench(Ticket, rows, [FakeEngine(truth)]).details["rows"] == 4
    assert (
        ds.bench(Ticket, [{k: v for k, v in r.items() if k != "split"} for r in rows], [FakeEngine(truth)]).details[
            "rows"
        ]
        == 12
    )


def test_golden_jsonl_round_trip_and_bad_extension(tmp_path):
    out = str(tmp_path / "golden.jsonl")
    model = ds.model(LABELS, FakeEngine(team, confidence=0.6))
    rows = ds.golden(corpus(20), FakeEngine(team), n=20, schema=model, out=out, verbose=False)
    lines = [json.loads(x) for x in open(out, encoding="utf-8")]
    assert len(lines) == 20 and lines[0]["answers"]["label"] in LABELS and "split" in lines[0]
    assert len(data_mod.load(out, model.schema, split="test")) == sum(r["split"] == "test" for r in rows) == 4
    with pytest.raises(ValueError, match="out must end in .csv"):
        ds.golden(corpus(3), FakeEngine(team), schema=LABELS, strategy="random", out=str(tmp_path / "g.txt"))


def test_golden_csv_cells_are_safe_from_formulas(tmp_path):
    out = str(tmp_path / "g.csv")
    texts = ["=HYPERLINK(1) you charged me twice", "@SUM refund my card charge", "+1 the app crashes on login"]
    ds.golden(texts, FakeEngine(team), n=3, schema=LABELS, strategy="random", out=out, verbose=False)
    raw = open(out, encoding="utf-8").read()
    assert "'=HYPERLINK" in raw and "'@SUM" in raw and "'+1" in raw
    assert sorted(r.text for r in data_mod.load(out, ds.model(LABELS, "fake").schema, split="all")) == sorted(texts)
    assert sorted(cli.read_texts(out)) == sorted(texts)


def test_golden_teacher_failures(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    class Flaky(FakeEngine):
        def ask(self, text, questions):
            calls.append(text)
            if text.endswith(" 0"):
                raise ValueError("bad reply")
            return super().ask(text, questions)

    rows = ds.golden(corpus(9), Flaky(team), n=9, schema=LABELS, strategy="random")
    assert len(rows) == 6
    assert "3 could not be labelled (first error: ValueError: bad reply)" in capsys.readouterr().out
    calls.clear()
    with pytest.raises(ds.EngineError, match="first error"):
        ds.golden(corpus(40), FakeEngine(team, error="quota"), n=40, schema=LABELS, strategy="random", out=None)
    assert len(FakeEngine(team, error="quota").calls) == 0


def test_golden_uses_human_labels_and_keeps_trained_rows_out_of_test(db):
    with ds.harness(
        Ticket, teacher=FakeEngine(truth), student=FakeEngine(truth, confidence=0.5), mode="shadow", log=db
    ) as h:
        results = [h.decide(t) for t in corpus(10)]
        h.label(results[0].id, team="sales", wants_refund=True)
        h.label(results[1].id, team="sales")
        h.log.mark_trained([r.id for r in results[2:]], "runs/x")
    llm = FakeEngine(truth, name="big")
    rows = {r["text"]: r for r in ds.golden(db, llm, n=10, schema=Ticket, test=0.5, out=None, verbose=False)}
    first, second = rows[corpus(10)[0]], rows[corpus(10)[1]]
    assert first["labelled_by"] == "human" and first["answers"]["team"]["sales"] == 1.0
    assert second["labelled_by"] == "llm:big+human" and second["answers"]["team"]["sales"] == 1.0
    assert corpus(10)[0] not in {c[0] for c in llm.calls}
    assert {t for t, r in rows.items() if r["split"] == "test"} <= set(corpus(10)[:2])


def test_golden_small_n_still_holds_out_a_row():
    rows = ds.golden(corpus(2), FakeEngine(team), n=2, schema=LABELS, strategy="random", out=None, verbose=False)
    assert sorted(r["split"] for r in rows) == ["test", "train"]
    one = ds.golden(corpus(1), FakeEngine(team), n=1, schema=LABELS, strategy="random", out=None, verbose=False)
    assert [r["split"] for r in one] == ["train"]


def test_golden_ids_are_stable_across_runs():
    a = ds.golden(corpus(6), FakeEngine(team), n=3, schema=LABELS, strategy="random", seed=1, out=None, verbose=False)
    b = ds.golden(corpus(6), FakeEngine(team), n=6, schema=LABELS, strategy="random", seed=2, out=None, verbose=False)
    ids = {r["text"]: r["id"] for r in b}
    assert all(ids[r["text"]] == r["id"] for r in a) and len(set(ids.values())) == 6


def test_same_text():
    assert data_mod.same_text("Ｒefund, please!!  NOW") == data_mod.same_text("refund please now")
    assert data_mod.text_hash("Straße") == data_mod.text_hash("STRASSE")
    assert data_mod.text_hash("refund") != data_mod.text_hash("refunds")


def test_train_with_teacher_keeps_test_rows_out(tiny, monkeypatch):
    import decisionsmith.training.finetuning as fmod

    seen = {}

    def fake(rows, model, **kw):
        seen["rows"] = rows
        return ds.Report(
            "finetune",
            "x",
            [],
            details={"base": {"all": {}}, "finetuned": {"all": {"n": 0}}, "provenance": {"rows": {"train": 0}}},
        )

    monkeypatch.setattr(fmod, "finetune", fake)
    m = ds.model(LABELS, str(tiny))
    data = [{"text": t, "split": "test" if i < 3 else "train"} for i, t in enumerate(corpus(10))] + [
        "sync is broken today"
    ]
    m.train(data, teacher=FakeEngine(team), out=str(tiny), verbose=False)
    texts = {r["text"] for r in seen["rows"]}
    assert not texts & set(corpus(10)[:3]) and "sync is broken today" in texts
    assert {r["split"] for r in seen["rows"]} == {"train", None}


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
        "--overwrite",
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
    assert code == cli.INVALID
    for argv, match in (
        (["golden", "--labels", "a,b", "--teacher", "x"], "not both"),
        (["golden", str(texts), "--labels", "a,b"], "needs --teacher"),
        (["golden", str(texts), "--teacher", "x"], "either --labels"),
        (["golden", "--log", db, "--labels", "a,b", "--teacher", "x", "--score"], "--score is for a texts file"),
    ):
        code, text = run(capsys, *argv, "--json")
        assert code == cli.INVALID and match in json.loads(text)["error"]["message"]


def test_cli_golden_no_rows(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_teacher", lambda args: FakeEngine(team, error="down"))
    texts = tmp_path / "t.txt"
    texts.write_text("\n".join(corpus(3)))
    code, text = run(
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
    assert code == cli.ENGINE and not os.path.exists(tmp_path / "g.csv")
    error = json.loads(text)["error"]
    assert error["code"] == "engine" and "first error" in error["message"] and "doctor" in error["fix"]


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


def test_split_values_calib_and_errors(tmp_path):
    rows = [{"text": "t%d" % i, "label": "sales", "split": "calib" if i < 5 else ""} for i in range(40)]
    model = ds.model(LABELS, "fake")
    loaded = data_mod.load(model._rows(rows), model.schema)
    tr, ca, te = data_mod.split(loaded)
    assert {r.id for r in ca} == {r.id for r in loaded if r.split == "calib"} and len(tr) + len(te) == 35
    assert not {r.id for r in ca} & {r.id for r in tr + te}
    with pytest.raises(data_mod.DataError, match="split must be one of"):
        data_mod.load(model._rows([{"text": "a", "label": "sales", "split": "holdout"}]), model.schema)
    assert data_mod.load(model._rows([{"text": "a", "label": "sales", "split": "DEV"}]), model.schema)[0].split == (
        "calib"
    )


def test_evaluate_flags_texts_it_trained_on(tiny, tmp_path):
    import shutil

    folder = tmp_path / "ckpt"
    shutil.copytree(tiny, folder)
    cfg = json.loads((folder / "rl_agent_config.json").read_text())
    texts = corpus(12)
    cfg["decisionsmith"] = {"text_hashes": [data_mod.text_hash(t.upper() + "  ") for t in texts[:4]]}
    (folder / "rl_agent_config.json").write_text(json.dumps(cfg))
    model = ds.model(LABELS, str(folder))
    report = model.evaluate([(t, truth(t)["team"]) for t in texts])
    assert report.details["overlap"] == 4 and not report.go
    assert report.reasons[0].startswith("4 of 12 test texts were in the training data")
    assert ds.model(LABELS, str(tiny)).evaluate([(t, truth(t)["team"]) for t in texts]).details["overlap"] == 0
    assert ds.model(LABELS, FakeEngine(team)).evaluate([(t, truth(t)["team"]) for t in texts]).details["overlap"] == 0


def test_training_hashes_unknown_checkpoint():
    from decisionsmith.evaluation import training_hashes

    assert training_hashes(ds.model(LABELS, "laya")) == set()


def test_hash_chain_train_retrain_save_load_evaluate(tiny, tmp_path):
    from decisionsmith.training.finetuning import _base_hashes

    first = [(t, truth(t)["team"]) for t in corpus(30)]
    second = [(t + " again", truth(t)["team"]) for t in corpus(30)]
    to_rows = ds.model(LABELS, "fake")._rows
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    ds.finetune(to_rows(first), ds.model(LABELS, "fake").schema.model, base=str(tiny), out=a, epochs=1, verbose=False)
    ds.finetune(to_rows(second), ds.model(LABELS, "fake").schema.model, base=a, out=b, epochs=1, verbose=False)
    assert _base_hashes(a) <= _base_hashes(b) and len(_base_hashes(b)) == 60
    assert _base_hashes(str(tmp_path / "missing")) == set()
    m = ds.model(LABELS, b)
    m.trained = b
    loaded = ds.load(m.save(str(tmp_path / "models" / "team")))
    report = loaded.evaluate(first[:12])
    assert report.details["overlap"] == 12 and not report.go
