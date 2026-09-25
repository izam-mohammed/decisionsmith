import json
import textwrap

import pytest

import decisionsmith as ds
from decisionsmith import benchmark, cli
from decisionsmith import mcp as mcp_tools
from decisionsmith.engines import JevEngine, LayaEngine
from decisionsmith.testing import FakeEngine
from tests.conftest import Ticket, corpus, truth

SCHEMA = "tests/conftest.py:Ticket"


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_bench_report(toy_csv, tiny, tmp_path, monkeypatch):
    perfect = FakeEngine(truth, confidence=0.95, name="oracle")
    rep = ds.bench(Ticket, toy_csv, [perfect, "fake", "laya:%s" % tiny, "jev"], out=str(tmp_path / "b.json"))
    rows = {(r["engine"], r["field"]): r for r in rep.rows}
    assert rows[("oracle", "team")]["accuracy"] == 1.0 and rows[("oracle", "team")]["coverage"] == 1.0
    assert rows[("fake", "team")]["cost_per_1k"] == 0.0
    assert rows[("laya:%s" % tiny, "wants_refund")]["n"] == 60
    assert "no API key" in rows[("jev", "-")]["error"]
    assert json.loads((tmp_path / "b.json").read_text())["kind"] == "bench"
    assert ds.bench(Ticket, toy_csv, "fake", limit=5).rows[0]["n"] == 5
    with pytest.raises(ValueError, match="no labelled rows"):
        ds.bench(Ticket, [], ["fake"])


def test_bench_errors_and_costs(toy_csv, monkeypatch):
    class Batch(FakeEngine):
        def ask_many(self, texts, questions):
            raise RuntimeError("gpu gone")

    rep = ds.bench(Ticket, toy_csv, [Batch(name="b"), FakeEngine(error="down", name="d")])
    assert all(r["errors"] == 60 for r in rep.rows)
    assert rep.details["engines"]["d"]["first_error"].startswith("engine 'd'")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    jev = JevEngine()
    assert benchmark._cost(jev, {"input_tokens": 1_000_000}) == pytest.approx(0.042)
    assert benchmark._cost(jev, {"cost_usd": 0.5}) == 0.5
    assert benchmark._cost(LayaEngine("laya"), {}) == 0.0
    assert benchmark._cost(ds.harness(Ticket, teacher="gpt-5", log=None).teacher, {}) is None


def test_cli_finetune_and_status(capsys, toy_csv, tiny, tmp_path):
    code, out, _ = run(
        capsys,
        "finetune",
        toy_csv,
        "--schema",
        SCHEMA,
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
    data = json.loads(out)
    assert code == cli.NOT_READY and data["go"] is False and data["kind"] == "finetune"
    db = str(tmp_path / "d.db")
    h = ds.harness(
        Ticket, teacher=FakeEngine(truth, name="t"), student=FakeEngine(truth, name="s"), mode="shadow", log=db
    )
    h.many(corpus(5))
    code, out, _ = run(capsys, "status", "--schema", SCHEMA, "--log", db)
    assert code == cli.NOT_READY and "5 decisions · student s" in out and "needs more data" in out
    code, out, _ = run(capsys, "status", "--schema", SCHEMA, "--log", db, "--json")
    assert json.loads(out)["decisions"] == 5
    h.log.set(
        "adapt:Ticket:s",
        {"team": {"temperature": 1.0, "threshold": None}, "wants_refund": {"temperature": 1.0, "threshold": 0.5}},
    )
    code, out, _ = run(capsys, "status", "--schema", SCHEMA, "--log", db, "--json")
    assert json.loads(out)["fields"]["team"]["sure_rate"] == 0.0
    code, out, _ = run(capsys, "export", "--schema", SCHEMA, "--log", db, "--out", str(tmp_path / "t.jsonl"))
    assert code == cli.OK and "wrote 5 rows" in out
    assert json.loads(open(tmp_path / "t.jsonl").readline())["answers"]
    code, out, _ = run(capsys, "status", "--schema", SCHEMA, "--log", str(tmp_path / "missing.db"), "--json")
    assert code == cli.INVALID and json.loads(out)["error"]["code"] == "invalid"


def test_cli_status_ready(capsys, tmp_path):
    db = str(tmp_path / "d.db")
    h = ds.harness(
        Ticket,
        teacher=FakeEngine(truth, name="t"),
        student=FakeEngine(truth, confidence=0.99, name="s"),
        mode="shadow",
        log=db,
    )
    h.many(corpus(500))
    code, out, _ = run(capsys, "status", "--schema", SCHEMA, "--log", db)
    assert code == cli.OK and "ready for student" in out


def test_cli_bench_and_doctor(capsys, toy_csv, tmp_path, monkeypatch):
    code, out, _ = run(capsys, "bench", toy_csv, "--schema", SCHEMA, "--engines", "fake", "--limit", "4")
    assert code == cli.OK and "bench: Ticket" in out
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    code, out, _ = run(capsys, "doctor", "--engines", "fake,jev", "--json")
    checks = {c["check"]: c for c in json.loads(out)["checks"]}
    assert code == cli.ENGINE and checks["engine fake"]["ok"] and not checks["engine jev"]["ok"]
    assert "TYPESAFE_API_KEY" in checks["engine jev"]["fix"] and checks["decisionsmith"]["ok"]
    code, out, _ = run(capsys, "doctor", "--engines", "systemone:")
    assert code == cli.ENGINE and "engine systemone:    systemone needs a URL" in out
    code, out, _ = run(capsys, "doctor")
    assert code == cli.OK and "device" in out


def test_doctor_missing_package(capsys, monkeypatch):
    import importlib.metadata as md

    real = md.version

    def fake_version(name):
        if name == "mcp":
            raise md.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(md, "version", fake_version)
    code, out, _ = run(capsys, "doctor")
    assert code == cli.ERROR and "decisionsmith[mcp]" in out


def test_load_schema(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""
        from pydantic import BaseModel
        class Spam(BaseModel):
            spam: bool
    """)
    )
    assert cli.load_schema(str(tmp_path / "app.py") + ":Spam").__name__ == "Spam"
    monkeypatch.chdir(tmp_path)
    assert cli.load_schema("app:Spam").__name__ == "Spam"
    for bad, match in (("app.py", "must look like"), ("nope.py:X", "not found"), ("app:Nope", "has no")):
        with pytest.raises(ValueError, match=match):
            cli.load_schema(bad)


def test_errors_map_to_exit_codes(capsys, monkeypatch, tmp_path):
    code, _, err = run(capsys, "bench", str(tmp_path / "x.csv"), "--schema", SCHEMA, "--engines", "fake")
    assert code == cli.INVALID and "no such file" in err

    def boom(args):
        raise RuntimeError("kaboom")

    def eng(args):
        raise ds.EngineError("jev", "down", "retry")

    def stop(args):
        raise KeyboardInterrupt

    for fn, want in ((boom, cli.ERROR), (eng, cli.ENGINE), (stop, cli.ERROR)):
        monkeypatch.setattr(cli, "_doctor", fn)
        p = cli.parser()
        monkeypatch.setattr(cli, "parser", lambda p=p: p)
        for action in p._subparsers._group_actions[0].choices.values():
            if action.prog.endswith("doctor"):
                action.set_defaults(run=fn)
        code, out, err = run(capsys, "doctor", "--json")
        assert code == want
        monkeypatch.undo()


def test_mcp_tools(tmp_path, toy_csv, monkeypatch):
    db = str(tmp_path / "m.db")
    json_schema = {
        "title": "T",
        "properties": {
            "team": {"enum": ["billing", "technical", "sales"]},
            "wants_refund": {"type": "boolean", "description": "Refund?"},
        },
    }
    r = mcp_tools.decide("you charged me twice", json_schema, teacher="fake", log=db)
    assert set(r["value"]) == {"team", "wants_refund"} and r["id"]
    assert mcp_tools.model_from(json_schema) is mcp_tools.model_from(json_schema)
    assert mcp_tools.label(r["id"], {"team": "sales"}, json_schema, log=db) == {"ok": True}
    st = mcp_tools.status(json_schema, log=db)
    assert st["decisions"] == 1 and st["fields"]["team"]["human_labels"] == 1
    assert mcp_tools.export(json_schema, str(tmp_path / "e.jsonl"), log=db)["rows"] == 1
    with pytest.raises(ValueError, match="no log"):
        mcp_tools.export(json_schema, "x", log=str(tmp_path / "none.db"))
    assert mcp_tools.bench(toy_csv, SCHEMA, ["fake"])["kind"] == "bench"
    assert mcp_tools.schema_compile(SCHEMA)["questions"]["team"]["type"] == "choice"
    many = {"properties": {"x": {"enum": ["o%d" % i for i in range(21)]}}}
    assert mcp_tools.schema_compile(many)["warnings"]
    assert mcp_tools.engines_check(["fake"])["checks"][-1]["ok"] is True
    for bad in ({"properties": {}}, {"properties": {"x": {"type": "string"}}}, 3):
        with pytest.raises(ValueError):
            mcp_tools.model_from(bad)
    calls = {}
    monkeypatch.setattr(
        "decisionsmith.training.finetuning.finetune",
        lambda *a, **k: calls.update(a=a, k=k) or ds.Report("finetune", "x", [], go=False),
    )
    assert mcp_tools.finetune(toy_csv, SCHEMA, out="o")["go"] is False and calls["k"]["verbose"] is False
    assert mcp_tools.finetune(toy_csv)["kind"] == "finetune" and calls["a"][1] is None
    app = mcp_tools.server()
    assert app.name == "decisionsmith"


def test_worker_rank_exits_zero(capsys, monkeypatch):
    monkeypatch.setattr(
        "decisionsmith.training.finetuning.finetune",
        lambda *a, **k: ds.Report("finetune", "worker rank 1 finished", []),
    )
    code, out, _ = run(capsys, "finetune", "x.csv")
    assert code == cli.OK and "worker rank 1" in out


def test_load_schema_adds_cwd_to_path(tmp_path, monkeypatch):
    import os
    import sys

    (tmp_path / "cwd_schema.py").write_text("from pydantic import BaseModel\nclass Egg(BaseModel):\n    egg: bool\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != os.getcwd()])
    assert cli.load_schema("cwd_schema:Egg").__name__ == "Egg" and sys.path[0] == os.getcwd()


def test_doctor_without_torch_and_mcp_command(capsys, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "decisionsmith.training.train", None)
    code, out, _ = run(capsys, "doctor", "--json")
    assert "device" not in [c["check"] for c in json.loads(out)["checks"]]
    served = []
    monkeypatch.setattr("decisionsmith.mcp.serve", lambda: served.append(1))
    assert cli.main(["mcp"]) == cli.OK and served == [1]


def test_mcp_reuses_harnesses_and_needs_the_extra(tmp_path, monkeypatch):
    import sys

    db = str(tmp_path / "m.db")
    a = mcp_tools._harness(SCHEMA, "fake", None, None, db)
    assert mcp_tools._harness(SCHEMA, "fake", None, None, db) is a
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    with pytest.raises(SystemExit, match="decisionsmith\\[mcp\\]"):
        mcp_tools.server()
