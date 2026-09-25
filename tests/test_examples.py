import os
import runpy
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


@pytest.fixture
def example_env(monkeypatch, tiny, tmp_path):
    monkeypatch.syspath_prepend(EXAMPLES)
    monkeypatch.setenv("DS_STUDENT", "laya:%s" % tiny)
    monkeypatch.setenv("DS_BASE", str(tiny))
    monkeypatch.setenv("DS_ENGINES", "fake,laya:%s" % tiny)
    monkeypatch.setenv("DS_LOG", str(tmp_path / "e.db"))
    monkeypatch.setenv("DS_OUT", str(tmp_path / "run"))
    monkeypatch.delenv("DS_TEACHER", raising=False)
    for mod in ("schema", "_common"):
        sys.modules.pop(mod, None)


@pytest.mark.parametrize("name", ["01_quickstart.py", "03_bench.py", "04_finetune.py"])
def test_example_runs(example_env, name, capsys):
    runpy.run_path(os.path.join(EXAMPLES, name), run_name="__main__")
    out = capsys.readouterr().out
    assert {"01_quickstart.py": "adapt: Ticket", "03_bench.py": "bench: Ticket", "04_finetune.py": "go:"}[name] in out


def test_jev_example_needs_a_key(example_env, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        runpy.run_path(os.path.join(EXAMPLES, "02_jev_teacher_laya_student.py"), run_name="__main__")


def test_stand_in_teacher_uses_env(example_env, monkeypatch):
    import _common

    monkeypatch.setenv("DS_TEACHER", "claude-haiku-4-5")
    assert _common.teacher() == "claude-haiku-4-5"
