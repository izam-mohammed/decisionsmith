import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("changed_integrations", ROOT / "scripts" / "changed_integrations.py")
assert spec and spec.loader
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def run(monkeypatch, capsys, paths, argv=()):
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(paths)))
    assert ci.main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def test_names_come_from_the_pinned_groups():
    names = ci.names()
    assert "langchain" in names and "claude-agent-sdk" in names and "base" not in names


def test_docs_change_runs_nothing(monkeypatch, capsys):
    assert run(monkeypatch, capsys, ["README.md", "docs/guide.md"]) == []


def test_one_integration_runs_only_that_one(monkeypatch, capsys):
    paths = [
        "src/decisionsmith/integrations/pydantic_ai.py",
        "examples/04-integrations/langchain/teacher_openai.py",
        "tests\\integrations\\test_pydantic_ai.py",
    ]
    assert run(monkeypatch, capsys, paths) == ["langchain", "pydantic-ai"]


def test_core_change_runs_everything(monkeypatch, capsys):
    assert run(monkeypatch, capsys, ["src/decisionsmith/core.py"]) == ci.names()
    assert run(monkeypatch, capsys, ["uv.lock", "README.md"]) == ci.names()
    assert run(monkeypatch, capsys, [], ["--all"]) == ci.names()


def test_main_reads_argv_by_default(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["changed_integrations.py", "--all"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert ci.main() == 0
    assert json.loads(capsys.readouterr().out) == ci.names()
