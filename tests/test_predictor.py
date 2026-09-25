import json
import os
import re
import types

import pytest

import decisionsmith as ds
from decisionsmith import cli
from decisionsmith.engines import EngineError
from decisionsmith.predictor import labels_model, write_rows
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, corpus, truth

LABELS = ["billing", "technical", "sales"]


def team_of(text):
    return truth(text)["team"]


class Writer(FakeEngine):
    """A fake LLM that writes texts for the requested label and labels texts it wrote correctly."""

    def __init__(self, bad_every=0, **kw):
        super().__init__(self._truth, confidence=1.0, name="writer", **kw)
        self.prompts, self.bad_every, self.count = [], bad_every, 0

    @staticmethod
    def _truth(text):
        team = re.search(r"\[(\w+)\]", text).group(1)
        return {"label": team, "team": team, "wants_refund": "refund" in text}

    def write(self, prompt, n):
        self.prompts.append(prompt)
        team = re.search(r"-> (billing|technical|sales)", prompt).group(1)
        refund = "-> yes" in prompt
        out = []
        for _ in range(n):
            self.count += 1
            tag = "sales" if self.bad_every and self.count % self.bad_every == 0 and team != "sales" else team
            out.append("case %d [%s]%s" % (self.count, tag, " refund" if refund else ""))
        return [*out, "", out[0]]


def test_labels_model_and_repr():
    m = ds.model(LABELS)
    assert m.simple and list(m.schema.fields) == ["label"] and m.name == "laya"
    assert repr(m) == "ds.model(['billing', 'technical', 'sales'], engine='laya')"
    assert "Ticket" in repr(ds.model(Ticket, "fake"))
    assert m.schema.fields["label"].question["instructions"] == "Which label fits this text?"
    assert ds.model(LABELS, question="Which team?").schema.fields["label"].question["instructions"] == "Which team?"
    for bad in (["a"], ["a", "a"]):
        with pytest.raises(ValueError, match="two different"):
            labels_model(bad)


def test_train_predict_save_load(tiny, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = ds.model(LABELS, "laya:%s" % tiny, device="cpu")
    rows = [(t, team_of(t)) for t in corpus(60)]
    with pytest.raises(ValueError, match="train"):
        ds.model(LABELS, "laya").save("x")
    rep = m.train(rows, epochs=1)
    out = capsys.readouterr().out
    assert "trained on 45 rows" in out and "rough" in out and rep.kind == "finetune"
    if m.trained:
        run = os.path.join("runs", "label-v1")
        assert m.trained == run and m.name == "laya:%s" % run
        saved = m.save(str(tmp_path / "saved"))
        assert saved == str(tmp_path / "saved-v1")
        assert (tmp_path / "saved-v1" / "model.safetensors").exists()
        assert not (tmp_path / "saved-v1" / "checkpoint_latest").exists()
        assert (tmp_path / "saved-v1" / "train_report.json").exists()
        again = ds.model(LABELS, saved)
        assert again.predict("you charged me twice") in LABELS
        assert ds.load(saved).predict("you charged me twice") in LABELS
    else:
        assert "kept the old model" in out
    assert m.predict("you charged me twice") in LABELS
    assert all(x in LABELS for x in m.predict(["a b", "c d"]))
    for bad in ("", ["ok", " "], [3]):
        with pytest.raises(ValueError, match="non-empty"):
            m.predict(bad)


def _fake_train(tuned):
    def fake(rows, model, base, out, **kw):
        acc = {"base": 0.9, "tuned": tuned}
        return ds.Report(
            "finetune",
            "x",
            [],
            details={
                "base": {"all": {"accuracy": acc["base"]}},
                "finetuned": {"all": {"accuracy": acc["tuned"], "n": 200}},
                "provenance": {"rows": {"train": 10}},
            },
        )

    return fake


def test_train_keeps_old_model_when_worse(tiny, tmp_path, monkeypatch, capsys):
    import decisionsmith.training.finetuning as fmod

    monkeypatch.setattr(fmod, "finetune", _fake_train(0.5))
    m = ds.model(LABELS, str(tiny))
    rep = m.train([("x", "sales")], out=str(tmp_path / "o"))
    out = capsys.readouterr().out
    assert "kept the old model" in out and "rough" not in out and m.trained is None
    assert rep.switched is False and rep.to_dict()["switched"] is False
    assert m.save(str(tmp_path / "saved"), verbose=False) == str(tmp_path / "saved-v1")
    monkeypatch.setattr(fmod, "finetune", _fake_train(0.95))
    rep = m.train([("x", "sales")], out=str(tiny))
    assert rep.switched is True and m.trained == str(tiny)
    monkeypatch.setattr(fmod, "finetune", _fake_train(0.5))
    assert m.train([("x", "sales")], out=str(tmp_path / "o2"), verbose=False).switched is False
    assert m.trained == str(tiny)
    assert m.save(str(tmp_path / "saved"), verbose=False) == str(tmp_path / "saved-v2")


def _thresholds(path):
    with open(os.path.join(path, "decisionsmith.json")) as f:
        return json.load(f)["thresholds"]


def test_a_kept_train_leaves_earlier_work_saveable(tiny, tmp_path, monkeypatch):
    import decisionsmith.training.finetuning as fmod

    monkeypatch.setattr(fmod, "finetune", _fake_train(0.5))
    test = [{"text": t, "label": team_of(t)} for t in corpus(40)]
    base = ds.model(LABELS, str(tiny)).save(str(tmp_path / "base"), verbose=False)
    m = ds.load(base)
    m.evaluate(test)
    m.train([("x", "sales")], out=str(tmp_path / "o"), verbose=False)
    assert "label" in _thresholds(m.save(verbose=False))
    m = ds.load(base)
    m.train([("x", "sales")], out=str(tmp_path / "o"), verbose=False)
    m.calibration = {"label": {"temperature": 1.3, "threshold": 0.8}}
    assert _thresholds(m.save(verbose=False)) == {"label": 0.8}
    m = ds.load(base)
    m.train([("x", "sales")], out=str(tmp_path / "o"), verbose=False)
    m.evaluate(test)
    assert "label" in _thresholds(m.save(verbose=False))


def test_save_after_a_kept_train_with_nothing_on_disk(tiny, tmp_path, monkeypatch):
    import decisionsmith.training.finetuning as fmod

    monkeypatch.setattr(fmod, "finetune", _fake_train(0.5))
    monkeypatch.setenv("DS_OFFLINE", "1")
    monkeypatch.setenv("DS_LAYA", str(tiny))
    fresh = ds.model(LABELS)
    fresh.train([("x", "sales")], out=str(tmp_path / "o"), verbose=False)
    fresh.evaluate([{"text": t, "label": team_of(t)} for t in corpus(40)])
    with pytest.raises(ValueError, match="nothing to save: training ran but the new model scored worse"):
        fresh.save(str(tmp_path / "fresh"), verbose=False)
    with pytest.raises(ValueError, match="nothing to save yet: train it first"):
        ds.model(LABELS).save(str(tmp_path / "fresh"), verbose=False)


def test_report_switched_only_after_training():
    rep = ds.Report("bench", "x", [])
    assert rep.switched is None and "switched" not in rep.to_dict()


def test_rows_shapes(tiny):
    m = ds.model(Ticket, str(tiny))
    rows = m._rows([{"text": "a", "team": "sales", "other": 1}, {"text": "b", "answers": {"team": "billing"}}])
    assert rows[0]["answers"] == {"team": "sales"} and rows[1]["answers"] == {"team": "billing"}
    assert m._rows("file.csv") == "file.csv"
    for bad, match in (("just text", "only text"), (("a", "b"), "pairs"), (3, "expected")):
        with pytest.raises(ValueError, match=match):
            m._rows([bad])


def test_train_argument_errors(tiny):
    m = ds.model(LABELS, str(tiny))
    with pytest.raises(ValueError, match="let an LLM write it"):
        m.train()
    with pytest.raises(ValueError, match="as a list"):
        m.train("texts.csv", teacher="fake")
    with pytest.raises(EngineError, match="can't be trained"):
        ds.model(LABELS, "fake").train([("a", "sales")])


def test_label_with_teacher(tiny, monkeypatch, capsys):
    import decisionsmith.training.finetuning as fmod

    seen = {}

    def fake(rows, *a, out, **k):
        os.makedirs(out, exist_ok=True)
        seen.setdefault("rows", rows)
        details = {"base": {"all": {}}, "finetuned": {"all": {}}, "provenance": {"rows": {"train": 1}}}
        return ds.Report("finetune", "x", [], details=details)

    monkeypatch.setattr(fmod, "finetune", fake)
    m = ds.model(LABELS, str(tiny))

    class Flaky(FakeEngine):
        def ask(self, text, questions):
            if "bad" in text:
                raise EngineError("t", "down")
            return super().ask(text, questions)

    texts = [*corpus(6), "bad one"]
    m.train([*texts[:-1], {"text": texts[-1]}], teacher=Flaky(lambda t: {"label": team_of(t)}, name="t"))
    assert len(seen["rows"]) == 6 and "could not label 1 of 7" in capsys.readouterr().out
    assert seen["rows"][0]["answers"]["label"][team_of(texts[0])] > 0.5


def test_generate_balances_and_rechecks():
    m = ds.model(Ticket, "fake")
    w = Writer(bad_every=4)
    rows = m.generate(24, w, about="support emails", per_call=5, verbose=False)
    assert rows and all(r["answers"]["team"] in LABELS for r in rows)
    assert all(re.search(r"\[(\w+)\]", r["text"]).group(1) == r["answers"]["team"] for r in rows)
    assert len(rows) < 24 and any("(support emails)" in p for p in w.prompts)
    assert any("Style: frustrated" in p for p in w.prompts) and any("-> yes" in p for p in w.prompts)
    teams = [r["answers"]["team"] for r in rows]
    assert all(teams.count(t) >= 3 for t in LABELS)
    with pytest.raises(ValueError, match="at least 1"):
        m.generate(0, w)


def test_generate_describes_options_and_reports(capsys):
    m = ds.model(LABELS, "fake")
    w = Writer()
    rows = m.generate(3, w)
    assert len(rows) == 3 and "3 kept after re-checking" in capsys.readouterr().out
    t = ds.model(Ticket, "fake")
    t.generate(3, w, verbose=False)
    assert any("(payments and refunds)" in p for p in w.prompts)

    class Broken(Writer):
        def write(self, prompt, n):
            raise EngineError("w", "quota")

    assert m.generate(3, Broken()) == [] and "could not generate a batch" in capsys.readouterr().out
    assert m.generate(3, Broken(), verbose=False) == []


def test_train_on_generated(tiny, monkeypatch, capsys, tmp_path):
    import decisionsmith.training.finetuning as fmod

    got = {}

    def fake(rows, *a, **k):
        got["rows"] = rows
        return ds.Report(
            "finetune",
            "x",
            [],
            details={
                "base": {"all": {"accuracy": 0.3}},
                "finetuned": {"all": {"accuracy": 0.9, "n": 20}},
                "provenance": {"rows": {"train": 10}},
            },
        )

    monkeypatch.setattr(fmod, "finetune", fake)
    (tmp_path / "o").mkdir()
    m = ds.model(LABELS, str(tiny))
    m.train(generate=6, teacher=Writer(), out=str(tmp_path / "o"))
    out = capsys.readouterr().out
    assert got["rows"] and "LLM-written too" in out and m.trained == str(tmp_path / "o")


def test_write_rows(tmp_path):
    s = ds.model(Ticket, "fake").schema
    rows = [{"text": "hi", "answers": {"team": {"billing": 0.2, "sales": 0.8}, "wants_refund": "true"}}]
    write_rows(rows, str(tmp_path / "g.csv"), s)
    assert (tmp_path / "g.csv").read_text().splitlines() == ["text,team,wants_refund", "hi,sales,true"]
    write_rows(rows, str(tmp_path / "g.jsonl"), s)
    assert json.loads((tmp_path / "g.jsonl").read_text())["answers"]["wants_refund"] == "true"


def test_harness_takes_a_model(db):
    m = ds.model(LABELS, FakeEngine(lambda t: {"label": team_of(t)}, confidence=0.95, name="student"))
    h = ds.harness(m, teacher=FakeEngine(lambda t: {"label": team_of(t)}, name="teacher"), log=db, audit=0)
    assert h.simple and h.student.name == "student"
    assert h("you charged me twice") == "billing" and h.many(corpus(3)) == ["billing", "technical", "sales"]
    h2 = ds.harness(Ticket, teacher=FakeEngine(truth, name="t"), student=ds.model(Ticket, "fake"), log=db)
    assert h2.student.name == "fake" and not h2.simple


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_cli_generate_and_train(capsys, tmp_path, monkeypatch, tiny):
    monkeypatch.setattr(cli, "_teacher", lambda args: Writer() if args.teacher else None)
    out = str(tmp_path / "golden.csv")
    code, text = run(
        capsys, "generate", "--labels", "billing,technical,sales", "--teacher", "w", "-n", "6", "--out", out
    )
    assert code == cli.OK and "wrote 6 examples" in text and "--labels billing,technical,sales" in text
    assert open(out).readline().strip() == "text,label"
    code, text = run(
        capsys,
        "generate",
        "--schema",
        "tests/conftest.py:Ticket",
        "--teacher",
        "w",
        "-n",
        "3",
        "--out",
        str(tmp_path / "g.jsonl"),
        "--json",
    )
    assert json.loads(text)["rows"] == 3
    code, text = run(capsys, "generate", "--labels", "a,b", "--json")
    assert code == cli.INVALID and "needs --teacher" in text
    code, text = run(capsys, "generate", "--labels", "a,b", "--schema", "x.py:M", "--teacher", "w")
    assert code == cli.INVALID

    class Nothing(Writer):
        def write(self, prompt, n):
            return []

    monkeypatch.setattr(cli, "_teacher", lambda args: Nothing())
    code, _ = run(capsys, "generate", "--labels", "a,b", "--teacher", "w", "--out", str(tmp_path / "e.csv"))
    assert code == cli.NOT_READY

    monkeypatch.setattr(cli, "_teacher", lambda args: None)
    rows = "\n".join("%s,%s" % (t, team_of(t)) for t in corpus(60))
    (tmp_path / "labelled.csv").write_text("text,label\n" + rows + "\n")
    code, text = run(
        capsys,
        "train",
        str(tmp_path / "labelled.csv"),
        "--labels",
        "billing,technical,sales",
        "--base",
        str(tiny),
        "--out",
        str(tmp_path / "run"),
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--json",
    )
    data = json.loads(text)
    assert data.get("kind") == "finetune", data
    assert code in (cli.OK, cli.NOT_READY)


def test_cli_train_with_teacher_texts(capsys, tmp_path, monkeypatch):
    got = {}

    def fake_train(self, data, **kw):
        got.update(data=data, **kw)
        self.trained = "o"
        return ds.Report("finetune", "x", [])

    monkeypatch.setattr("decisionsmith.predictor.Model.train", fake_train)
    (tmp_path / "t.txt").write_text("one\n\ntwo\n")
    code, _ = run(
        capsys,
        "train",
        str(tmp_path / "t.txt"),
        "--labels",
        "a,b",
        "--teacher",
        "claude-haiku-4-5",
        "--teacher-url",
        "http://h/v1",
        "--epochs",
        "2",
    )
    assert code == cli.OK and got["data"] == ["one", "two"] and got["epochs"] == 2
    assert got["teacher"].url == "http://h/v1"
    code, _ = run(capsys, "train", "--labels", "a,b", "--generate", "5", "--teacher", "claude-haiku-4-5")
    assert got["data"] is None and got["generate"] == 5 and got["teacher"] == "claude-haiku-4-5"


def test_read_texts(tmp_path):
    (tmp_path / "a.csv").write_text("text,x\nhi,1\n,2\n")
    (tmp_path / "b.csv").write_text("body\nhi\n")
    (tmp_path / "c.jsonl").write_text('{"text": "hey"}\n\n')
    assert cli.read_texts(str(tmp_path / "a.csv")) == ["hi"]
    assert cli.read_texts(str(tmp_path / "c.jsonl")) == ["hey"]
    with pytest.raises(ValueError, match="'text' column"):
        cli.read_texts(str(tmp_path / "b.csv"))
    with pytest.raises(ValueError, match="no such file"):
        cli.read_texts(str(tmp_path / "zz.txt"))
    assert cli._teacher(types.SimpleNamespace(teacher=None)) is None
    assert cli._teacher(types.SimpleNamespace(teacher="gpt-5", teacher_url=None)) == "gpt-5"


def test_generate_is_exactly_balanced():
    w = Writer()
    ds.model(LABELS, "fake").generate(7, w, per_call=1, verbose=False)
    picked = [re.search(r"-> (\w+)", p).group(1) for p in w.prompts]
    assert sorted(picked.count(x) for x in LABELS) == [2, 2, 3] and "Write 1 realistic text." in w.prompts[0]


def test_model_is_an_engine():
    m = ds.model(["a", "b"], FakeEngine())
    q = m.schema.questions()
    assert m.name == "fake" and set(m.ask("x", q)["answers"]) == {"label"}
